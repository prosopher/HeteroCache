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
    output_dir: str = "./outputs/lsc_toy"
    max_steps: int = 500
    batch_size: int = 1
    grad_accum_steps: int = 16
    total_tokens: int = 128
    prefix_tokens: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 50
    grad_clip_norm: float = 1.0
    log_every: int = 25
    save_every: int = 100
    seed: int = 42
    shuffle_buffer: int = 50_000
    shared_slots: int = 32
    shared_dim: int = 128
    translator_dim: int = 256
    translator_heads: int = 4
    translator_mlp_ratio: int = 2
    top_layers_ratio: float = 1.0
    train_directions: str = "A_to_B,B_to_A"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"


@dataclass
class ModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int


@dataclass
class SharedCache:
    key: torch.Tensor
    value: torch.Tensor


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


def parse_train_directions(train_directions: str) -> List[str]:
    allowed = {"A_to_B", "B_to_A"}
    parsed = [item.strip() for item in train_directions.split(",") if item.strip()]
    if not parsed:
        raise ValueError("train_directions must contain at least one direction.")

    deduped = []
    seen = set()
    for item in parsed:
        if item not in allowed:
            raise ValueError(
                f"Unsupported train direction: {item}. "
                f"Allowed values are: {sorted(allowed)}"
            )
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


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


def resolve_top_layers_to_translate(num_layers: int, top_layers_ratio: float) -> int:
    if not (0.0 < top_layers_ratio <= 1.0):
        raise ValueError(f"top_layers_ratio must be in (0, 1], got {top_layers_ratio}")
    return max(1, min(num_layers, int(math.ceil(num_layers * top_layers_ratio))))


def build_translated_model_specs(
    full_model_specs: Dict[str, ModelSpec],
    top_layers_ratio: float,
) -> Dict[str, ModelSpec]:
    translated_specs = {}
    for name, spec in full_model_specs.items():
        translated_specs[name] = ModelSpec(
            model_id=spec.model_id,
            num_layers=resolve_top_layers_to_translate(spec.num_layers, top_layers_ratio),
            hidden_size=spec.hidden_size,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
        )
    return translated_specs


class OpenWebTextSequenceStream(IterableDataset):
    """
    Streams OpenWebText and yields fixed-length token chunks.

    This behaves like a rolling token buffer, which is usually a better fit for
    language-model style toy experiments than per-document truncation.
    """

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


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(hidden)
        kv = self.context_norm(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        hidden = hidden + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class LocalToSharedTranslator(nn.Module):
    """
    Input:  [batch, seq, local_layers, local_hidden]
    Output: [batch, seq, shared_slots, shared_dim]
    """

    def __init__(
        self,
        local_hidden_size: int,
        local_layers: int,
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.local_layers = local_layers
        self.shared_slots = shared_slots
        self.shared_dim = shared_dim
        self.input_norm = nn.LayerNorm(local_hidden_size)
        self.input_proj = nn.Linear(local_hidden_size, translator_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=translator_dim,
                    num_heads=translator_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(local_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(local_layers * translator_dim)
        self.output_proj = nn.Linear(local_layers * translator_dim, shared_slots * shared_dim)

    def forward(self, local_block: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = local_block.shape
        projected = F.gelu(self.input_proj(self.input_norm(local_block)))
        hidden = projected[:, :, 0, :]
        collected = []
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden, projected[:, :, layer_idx, :])
            collected.append(hidden)
        fused = torch.cat(collected, dim=-1)
        shared = F.gelu(self.output_proj(self.output_norm(fused)))
        return shared.view(batch_size, seq_len, self.shared_slots, self.shared_dim)


class SharedToLocalTranslator(nn.Module):
    """
    Input:  [batch, seq, shared_slots, shared_dim]
    Output: [batch, seq, local_layers, local_hidden]
    """

    def __init__(
        self,
        local_hidden_size: int,
        local_layers: int,
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.local_layers = local_layers
        self.local_hidden_size = local_hidden_size
        self.shared_slots = shared_slots
        self.shared_dim = shared_dim
        flat_shared = shared_slots * shared_dim
        self.input_norm = nn.LayerNorm(flat_shared)
        self.input_proj = nn.Linear(flat_shared, local_layers * translator_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=translator_dim,
                    num_heads=translator_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(local_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(local_layers * translator_dim)
        self.output_proj = nn.Linear(local_layers * translator_dim, local_layers * local_hidden_size)

    def forward(self, shared_block: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = shared_block.shape
        flat_shared = shared_block.reshape(batch_size, seq_len, self.shared_slots * self.shared_dim)
        expanded = F.gelu(self.input_proj(self.input_norm(flat_shared)))
        expanded = expanded.view(batch_size, seq_len, self.local_layers, -1)
        hidden = expanded[:, :, 0, :]
        collected = []
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden, expanded[:, :, layer_idx, :])
            collected.append(hidden)
        fused = torch.cat(collected, dim=-1)
        local = F.gelu(self.output_proj(self.output_norm(fused)))
        return local.view(batch_size, seq_len, self.local_layers, self.local_hidden_size)


class ModelLatentAdapter(nn.Module):
    def __init__(
        self,
        model_name: str,
        local_layers: int,
        local_hidden_size: int,
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.key_to_shared = LocalToSharedTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )
        self.value_to_shared = LocalToSharedTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )
        self.key_from_shared = SharedToLocalTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )
        self.value_from_shared = SharedToLocalTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )

    def to_shared(self, key_block: torch.Tensor, value_block: torch.Tensor) -> SharedCache:
        return SharedCache(
            key=self.key_to_shared(key_block),
            value=self.value_to_shared(value_block),
        )

    def from_shared(self, shared_cache: SharedCache) -> Tuple[torch.Tensor, torch.Tensor]:
        key_block = self.key_from_shared(shared_cache.key)
        value_block = self.value_from_shared(shared_cache.value)
        return key_block, value_block


class SharedKVTranslatorPool(nn.Module):
    def __init__(
        self,
        translated_model_specs: Dict[str, ModelSpec],
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.translated_model_specs = translated_model_specs
        self.adapters = nn.ModuleDict(
            {
                name: ModelLatentAdapter(
                    model_name=name,
                    local_layers=spec.num_layers,
                    local_hidden_size=spec.hidden_size,
                    shared_slots=shared_slots,
                    shared_dim=shared_dim,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    mlp_ratio=mlp_ratio,
                )
                for name, spec in translated_model_specs.items()
            }
        )

    def translate_blocks(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_cache = self.adapters[src_name].to_shared(key_block, value_block)
        return self.adapters[dst_name].from_shared(shared_cache)

    def translate_top_layers(
        self,
        past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        src_top_layers = self.translated_model_specs[src_name].num_layers
        src_top_past = slice_top_layers(
            past_key_values=past_key_values,
            top_layers_to_translate=src_top_layers,
        )
        key_block, value_block = past_key_values_to_blocks(src_top_past)
        translated_key, translated_value = self.translate_blocks(
            key_block=key_block,
            value_block=value_block,
            src_name=src_name,
            dst_name=dst_name,
        )
        return blocks_to_past_key_values(
            key_block=translated_key,
            value_block=translated_value,
            model_spec=dst_spec,
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


def blocks_to_past_key_values(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    model_spec: ModelSpec,
) -> PastKeyValues:
    batch_size, seq_len, num_layers, hidden_size = key_block.shape
    if num_layers != model_spec.num_layers:
        raise ValueError(f"Layer mismatch: block has {num_layers}, model expects {model_spec.num_layers}.")
    if hidden_size != model_spec.hidden_size:
        raise ValueError(f"Hidden mismatch: block has {hidden_size}, model expects {model_spec.hidden_size}.")

    past_key_values = []
    for layer_idx in range(model_spec.num_layers):
        key_layer = key_block[:, :, layer_idx, :]
        value_layer = value_block[:, :, layer_idx, :]
        key_layer = key_layer.view(batch_size, seq_len, model_spec.num_heads, model_spec.head_dim)
        value_layer = value_layer.view(batch_size, seq_len, model_spec.num_heads, model_spec.head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
        past_key_values.append((key_layer, value_layer))
    return tuple(past_key_values)


def slice_top_layers(
    past_key_values: PastKeyValues,
    top_layers_to_translate: int,
) -> PastKeyValues:
    if top_layers_to_translate < 1:
        raise ValueError("top_layers_to_translate must be >= 1")
    if top_layers_to_translate > len(past_key_values):
        raise ValueError(
            f"Requested top {top_layers_to_translate} layers, but cache only has {len(past_key_values)} layers."
        )
    return tuple(past_key_values[-top_layers_to_translate:])


def replace_top_layers(
    base_past_key_values: PastKeyValues,
    translated_top_past_key_values: PastKeyValues,
) -> PastKeyValues:
    num_replace = len(translated_top_past_key_values)
    if num_replace < 1:
        raise ValueError("translated_top_past_key_values must contain at least one layer.")
    if num_replace > len(base_past_key_values):
        raise ValueError(
            f"Cannot replace {num_replace} layers in cache with only {len(base_past_key_values)} layers."
        )

    base_list = list(base_past_key_values)
    start_idx = len(base_list) - num_replace

    for offset, translated_layer in enumerate(translated_top_past_key_values):
        base_key, base_value = base_list[start_idx + offset]
        translated_key, translated_value = translated_layer

        if base_key.shape != translated_key.shape:
            raise ValueError(
                f"Key shape mismatch at replaced layer {offset}: "
                f"base={tuple(base_key.shape)} vs translated={tuple(translated_key.shape)}"
            )
        if base_value.shape != translated_value.shape:
            raise ValueError(
                f"Value shape mismatch at replaced layer {offset}: "
                f"base={tuple(base_value.shape)} vs translated={tuple(translated_value.shape)}"
            )

        base_list[start_idx + offset] = (translated_key, translated_value)

    return tuple(base_list)


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
    past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
) -> torch.Tensor:
    outputs = target_model(
        input_ids=lm_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        lm_labels.reshape(-1),
        reduction="mean",
    )
    return loss


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
    """
    For exact next-token supervision on the suffix:
      - prefix_cache_ids: tokens [0, ..., prefix_tokens-2]
      - lm_input_ids:     tokens [prefix_tokens-1, ..., T-2]
      - lm_labels:        tokens [prefix_tokens,   ..., T-1]

    This lets the first suffix label be predicted from the last prefix token.
    """
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
) -> Tuple[SharedKVTranslatorPool, Dict[str, ModelSpec], Dict[str, ModelSpec]]:
    full_model_specs = {
        "A": get_model_spec(model_a),
        "B": get_model_spec(model_b),
    }
    translated_model_specs = build_translated_model_specs(
        full_model_specs=full_model_specs,
        top_layers_ratio=config.top_layers_ratio,
    )
    translator_pool = SharedKVTranslatorPool(
        translated_model_specs=translated_model_specs,
        shared_slots=config.shared_slots,
        shared_dim=config.shared_dim,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        mlp_ratio=config.translator_mlp_ratio,
    )
    translator_pool.to(config.device)
    return translator_pool, full_model_specs, translated_model_specs


def build_models_and_tokenizer(config: TrainConfig) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = load_tokenizer(config.model_a_id)
    model_a = load_frozen_model(config.model_a_id, device=config.device, dtype=config.dtype)
    model_b = load_frozen_model(config.model_b_id, device=config.device, dtype=config.dtype)
    return model_a, model_b, tokenizer


def save_checkpoint(
    output_path: str,
    translator_pool: SharedKVTranslatorPool,
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
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_train_config_from_checkpoint(checkpoint_path: str) -> TrainConfig:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return TrainConfig(**payload["train_config"])


def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    SharedKVTranslatorPool,
    Dict[str, ModelSpec],
    Dict[str, ModelSpec],
    PreTrainedModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**payload["train_config"])
    if device_override is not None:
        config.device = device_override
    model_a, model_b, tokenizer = build_models_and_tokenizer(config)
    translator_pool, full_model_specs, translated_model_specs = build_translator_pool(model_a, model_b, config)
    translator_pool.load_state_dict(payload["translator_pool"])
    translator_pool.to(config.device)
    translator_pool.eval()
    return config, translator_pool, full_model_specs, translated_model_specs, model_a, model_b, tokenizer


def write_json(path: str, payload: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
