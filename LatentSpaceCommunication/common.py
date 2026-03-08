import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class TrainConfig:
    model_a_id: str = "gpt2"
    model_b_id: str = "gpt2-medium"
    output_dir: str = "./outputs/lsc_toy_topn"
    max_steps: int = 1000
    batch_size: int = 1
    grad_accum_steps: int = 16
    total_tokens: int = 128
    prefix_tokens: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 50
    grad_clip_norm: float = 1.0
    log_every: int = 25
    seed: int = 42
    shuffle_buffer: int = 50_000
    top_layers_to_translate: int = 6
    translator_dim: int = 512
    translator_heads: int = 8
    translator_depth: int = 2
    translator_mlp_ratio: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"


@dataclass
class ModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def freeze_model(model: PreTrainedModel) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def load_frozen_model(model_id: str, device: str, dtype: str = "float32") -> PreTrainedModel:
    torch_dtype = get_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    freeze_model(model)
    return model


def get_model_spec(model: PreTrainedModel) -> ModelSpec:
    config = model.config
    num_heads = getattr(config, "n_head", None)
    hidden_size = getattr(config, "n_embd", None)
    num_layers = getattr(config, "n_layer", None)
    if num_heads is None or hidden_size is None or num_layers is None:
        raise ValueError("This toy example expects GPT-2 style configs with n_head/n_embd/n_layer.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    return ModelSpec(
        model_id=getattr(config, "_name_or_path", "unknown"),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
    )


class OpenWebTextSequenceStream(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self) -> Iterable[torch.Tensor]:
        stream = load_dataset("openwebtext", split="train", streaming=True)
        stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        token_buffer: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for example in stream:
            text = example.get("text", "")
            if not text or text.isspace():
                continue
            token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(token_ids) < 8:
                continue
            token_buffer.extend(token_ids)
            token_buffer.append(eos_token_id)
            while len(token_buffer) >= self.sequence_length:
                chunk = token_buffer[: self.sequence_length]
                token_buffer = token_buffer[self.sequence_length :]
                yield torch.tensor(chunk, dtype=torch.long)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        norm_hidden = self.attn_norm(hidden)
        attn_out, _ = self.attn(norm_hidden, norm_hidden, norm_hidden, need_weights=False)
        hidden = hidden + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class PerLayerTranslator(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(src_hidden_size)
        self.input_proj = nn.Linear(src_hidden_size, translator_dim)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=translator_dim,
                    num_heads=translator_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(translator_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(translator_dim)
        self.output_proj = nn.Linear(translator_dim, dst_hidden_size)

    def forward(self, layer_cache: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_proj(self.input_norm(layer_cache)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_proj(self.output_norm(hidden))


class TopLayerDirectionalTranslator(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        top_layers_to_translate: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.top_layers_to_translate = top_layers_to_translate
        self.key_layers = nn.ModuleList(
            [
                PerLayerTranslator(
                    src_hidden_size=src_hidden_size,
                    dst_hidden_size=dst_hidden_size,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(top_layers_to_translate)
            ]
        )
        self.value_layers = nn.ModuleList(
            [
                PerLayerTranslator(
                    src_hidden_size=src_hidden_size,
                    dst_hidden_size=dst_hidden_size,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(top_layers_to_translate)
            ]
        )

    def forward(self, key_block: torch.Tensor, value_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_outputs = []
        value_outputs = []
        for layer_idx in range(self.top_layers_to_translate):
            key_outputs.append(self.key_layers[layer_idx](key_block[:, :, layer_idx, :]).unsqueeze(2))
            value_outputs.append(self.value_layers[layer_idx](value_block[:, :, layer_idx, :]).unsqueeze(2))
        translated_key = torch.cat(key_outputs, dim=2)
        translated_value = torch.cat(value_outputs, dim=2)
        return translated_key, translated_value


class TopLayerTranslatorPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        top_layers_to_translate: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        spec_a = model_specs["A"]
        spec_b = model_specs["B"]
        max_allowed = min(spec_a.num_layers, spec_b.num_layers)
        if top_layers_to_translate > max_allowed:
            raise ValueError(
                f"top_layers_to_translate={top_layers_to_translate} exceeds min layer count {max_allowed}."
            )
        self.model_specs = model_specs
        self.top_layers_to_translate = top_layers_to_translate
        self.adapters = nn.ModuleDict(
            {
                "A_to_B": TopLayerDirectionalTranslator(
                    src_hidden_size=spec_a.hidden_size,
                    dst_hidden_size=spec_b.hidden_size,
                    top_layers_to_translate=top_layers_to_translate,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                ),
                "B_to_A": TopLayerDirectionalTranslator(
                    src_hidden_size=spec_b.hidden_size,
                    dst_hidden_size=spec_a.hidden_size,
                    top_layers_to_translate=top_layers_to_translate,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                ),
            }
        )

    def translate_top_layer_blocks(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adapter_name = f"{src_name}_to_{dst_name}"
        return self.adapters[adapter_name](key_block, value_block)

    def translate_top_layers(
        self,
        past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        key_block, value_block = extract_top_layer_blocks(
            past_key_values=past_key_values,
            top_layers_to_translate=self.top_layers_to_translate,
        )
        translated_key, translated_value = self.translate_top_layer_blocks(
            key_block=key_block,
            value_block=value_block,
            src_name=src_name,
            dst_name=dst_name,
        )
        return blocks_to_partial_past_key_values(
            key_block=translated_key,
            value_block=translated_value,
            num_heads=dst_spec.num_heads,
            head_dim=dst_spec.head_dim,
        )


@torch.no_grad()
def extract_past_key_values(model: PreTrainedModel, input_ids: torch.Tensor) -> PastKeyValues:
    outputs = model(input_ids=input_ids, use_cache=True)
    return outputs.past_key_values


def past_key_values_to_blocks(past_key_values: PastKeyValues) -> Tuple[torch.Tensor, torch.Tensor]:
    key_layers = []
    value_layers = []
    for key, value in past_key_values:
        batch_size, num_heads, seq_len, head_dim = key.shape
        key_flat = key.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        value_flat = value.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        key_layers.append(key_flat)
        value_layers.append(value_flat)
    key_block = torch.stack(key_layers, dim=2)
    value_block = torch.stack(value_layers, dim=2)
    return key_block, value_block


def extract_top_layer_blocks(
    past_key_values: PastKeyValues,
    top_layers_to_translate: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if top_layers_to_translate < 1:
        raise ValueError("top_layers_to_translate must be >= 1")
    if top_layers_to_translate > len(past_key_values):
        raise ValueError(
            f"Cannot extract {top_layers_to_translate} layers from cache with only {len(past_key_values)} layers."
        )
    return past_key_values_to_blocks(past_key_values[-top_layers_to_translate:])


def blocks_to_partial_past_key_values(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> PastKeyValues:
    batch_size, seq_len, num_layers, hidden_size = key_block.shape
    expected_hidden = num_heads * head_dim
    if hidden_size != expected_hidden:
        raise ValueError(f"Hidden mismatch: block has {hidden_size}, expected {expected_hidden}.")

    past_key_values = []
    for layer_idx in range(num_layers):
        key_layer = key_block[:, :, layer_idx, :]
        value_layer = value_block[:, :, layer_idx, :]
        key_layer = key_layer.view(batch_size, seq_len, num_heads, head_dim)
        value_layer = value_layer.view(batch_size, seq_len, num_heads, head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
        past_key_values.append((key_layer, value_layer))
    return tuple(past_key_values)


def slice_top_layers(past_key_values: PastKeyValues, top_layers_to_translate: int) -> PastKeyValues:
    if top_layers_to_translate > len(past_key_values):
        raise ValueError(
            f"Cannot slice {top_layers_to_translate} layers from cache with only {len(past_key_values)} layers."
        )
    return tuple(past_key_values[-top_layers_to_translate:])


def replace_top_layers(
    base_past_key_values: PastKeyValues,
    translated_top_past_key_values: PastKeyValues,
) -> PastKeyValues:
    num_replaced = len(translated_top_past_key_values)
    if num_replaced < 1:
        raise ValueError("translated_top_past_key_values must contain at least one layer.")
    if num_replaced > len(base_past_key_values):
        raise ValueError(
            f"Cannot replace {num_replaced} layers in cache with only {len(base_past_key_values)} layers."
        )
    return tuple(base_past_key_values[:-num_replaced]) + tuple(translated_top_past_key_values)


def flatten_past_key_values(past_key_values: PastKeyValues) -> torch.Tensor:
    flat_parts = []
    for key, value in past_key_values:
        flat_parts.append(key.reshape(key.shape[0], -1))
        flat_parts.append(value.reshape(value.shape[0], -1))
    return torch.cat(flat_parts, dim=1)


def cosine_similarity_between_past(a: PastKeyValues, b: PastKeyValues) -> float:
    flat_a = flatten_past_key_values(a)
    flat_b = flatten_past_key_values(b)
    return F.cosine_similarity(flat_a, flat_b, dim=1).mean().item()


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def move_past_to_device(past_key_values: PastKeyValues, device: str) -> PastKeyValues:
    moved = []
    for key, value in past_key_values:
        moved.append((key.to(device), value.to(device)))
    return tuple(moved)


def compute_suffix_lm_loss(
    target_model: PreTrainedModel,
    translated_past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
) -> torch.Tensor:
    outputs = target_model(
        input_ids=lm_input_ids,
        past_key_values=translated_past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        lm_labels.reshape(-1),
        reduction="mean",
    )


class InfiniteDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def build_training_dataloader(tokenizer: PreTrainedTokenizerBase, config: TrainConfig) -> InfiniteDataLoader:
    dataset = OpenWebTextSequenceStream(
        tokenizer=tokenizer,
        sequence_length=config.total_tokens,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    return InfiniteDataLoader(dataloader)


def split_prefix_and_suffix_for_exact_next_token_loss(
    input_ids: torch.Tensor,
    prefix_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prefix_tokens < 2:
        raise ValueError("prefix_tokens must be >= 2")
    prefix_cache_ids = input_ids[:, : prefix_tokens - 1]
    lm_input_ids = input_ids[:, prefix_tokens - 1 : -1]
    lm_labels = input_ids[:, prefix_tokens:]
    return prefix_cache_ids, lm_input_ids, lm_labels


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_id = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_id += 1
        if self.step_id <= self.warmup_steps:
            multiplier = self.step_id / self.warmup_steps
        else:
            progress = (self.step_id - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def build_translator_pool(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    config: TrainConfig,
) -> Tuple[TopLayerTranslatorPool, Dict[str, ModelSpec]]:
    model_specs = {
        "A": get_model_spec(model_a),
        "B": get_model_spec(model_b),
    }
    translator_pool = TopLayerTranslatorPool(
        model_specs=model_specs,
        top_layers_to_translate=config.top_layers_to_translate,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs


def build_models_and_tokenizer(config: TrainConfig) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = load_tokenizer(config.model_a_id)
    model_a = load_frozen_model(config.model_a_id, device=config.device, dtype=config.dtype)
    model_b = load_frozen_model(config.model_b_id, device=config.device, dtype=config.dtype)
    return model_a, model_b, tokenizer


def save_checkpoint(
    output_path: str,
    translator_pool: TopLayerTranslatorPool,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    train_config: TrainConfig,
    step: int,
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "translator_pool": translator_pool.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_config": asdict(train_config),
        "scheduler_step": scheduler.step_id,
    }
    if extra is not None:
        payload["extra"] = extra
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))


def load_train_config_from_checkpoint(checkpoint_path: str) -> TrainConfig:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return TrainConfig(**payload["train_config"])


def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[TrainConfig, TopLayerTranslatorPool, Dict[str, ModelSpec], PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**payload["train_config"])
    if device_override is not None:
        config.device = device_override
    model_a, model_b, tokenizer = build_models_and_tokenizer(config)
    translator_pool, model_specs = build_translator_pool(model_a, model_b, config)
    translator_pool.load_state_dict(payload["translator_pool"])
    translator_pool.to(config.device)
    translator_pool.eval()
    return config, translator_pool, model_specs, model_a, model_b, tokenizer


@torch.no_grad()
def prepare_generation_prefix(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Prefix text must tokenize to at least 2 tokens.")
    cache_ids = input_ids[:, :-1]
    seed_token = input_ids[:, -1:]
    past_key_values = extract_past_key_values(model, cache_ids)
    return {
        "full_prefix_ids": input_ids,
        "cache_ids": cache_ids,
        "seed_token": seed_token,
        "past_key_values": past_key_values,
    }


@torch.no_grad()
def generate_from_past(
    model: PreTrainedModel,
    seed_token: torch.Tensor,
    past_key_values: PastKeyValues,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    current_input = seed_token
    current_past = past_key_values
    generated = []
    for _ in range(max_new_tokens):
        outputs = model(input_ids=current_input, past_key_values=current_past, use_cache=True)
        logits = outputs.logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        if do_sample:
            k = min(top_k, logits.shape[-1])
            values, indices = torch.topk(logits, k=k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, sampled)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        current_input = next_token
        current_past = outputs.past_key_values
        if eos_token_id is not None and torch.all(next_token.eq(eos_token_id)):
            break
    if not generated:
        return torch.empty(seed_token.shape[0], 0, dtype=seed_token.dtype, device=seed_token.device)
    return torch.cat(generated, dim=1)


def decode_full_generation(
    tokenizer: PreTrainedTokenizerBase,
    prefix_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> str:
    full_ids = torch.cat([prefix_ids.cpu(), generated_ids.cpu()], dim=1)
    return tokenizer.decode(full_ids[0], skip_special_tokens=True)


def write_json(path: str, payload: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
