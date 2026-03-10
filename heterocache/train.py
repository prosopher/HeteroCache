from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import *


@dataclass
class TrainConfig:
    alg: str = ""
    output_root: str = "outputs"
    timestamp: Optional[str] = None
    output_dir: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    model_a_id: str = "gpt2"
    model_b_id: str = "gpt2-medium"
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
    seed: int = 42
    shuffle_buffer: int = 50_000
    top_layers_to_translate: int = 6
    translator_dim: int = 1024
    translator_heads: int = 16
    translator_depth: int = 2
    translator_mlp_ratio: int = 2
    train_directions: str = "B_to_A"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        initialize_train_output_paths(self)


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
        active_directions: List[str],
    ) -> None:
        super().__init__()
        spec_a = model_specs["A"]
        spec_b = model_specs["B"]
        max_allowed = min(spec_a.num_layers, spec_b.num_layers)
        if top_layers_to_translate > max_allowed:
            raise ValueError(
                f"top_layers_to_translate={top_layers_to_translate} exceeds min layer count {max_allowed}."
            )
        if not active_directions:
            raise ValueError("active_directions must contain at least one direction.")

        self.model_specs = model_specs
        self.top_layers_to_translate = top_layers_to_translate
        self.active_directions = tuple(active_directions)

        adapters = {}
        if "A_to_B" in self.active_directions:
            adapters["A_to_B"] = TopLayerDirectionalTranslator(
                src_hidden_size=spec_a.hidden_size,
                dst_hidden_size=spec_b.hidden_size,
                top_layers_to_translate=top_layers_to_translate,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
            )
        if "B_to_A" in self.active_directions:
            adapters["B_to_A"] = TopLayerDirectionalTranslator(
                src_hidden_size=spec_b.hidden_size,
                dst_hidden_size=spec_a.hidden_size,
                top_layers_to_translate=top_layers_to_translate,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
            )

        self.adapters = nn.ModuleDict(adapters)

    def translate_top_layer_blocks(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adapter_name = f"{src_name}_to_{dst_name}"
        if adapter_name not in self.adapters:
            raise ValueError(
                f"Translator direction {adapter_name} is not available. "
                f"Active directions: {list(self.active_directions)}"
            )
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


def build_translator_pool(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    config: TrainConfig,
) -> Tuple[TopLayerTranslatorPool, Dict[str, ModelSpec]]:
    model_specs = {
        "A": get_model_spec(model_a),
        "B": get_model_spec(model_b),
    }
    active_directions = parse_train_directions(config.train_directions)
    translator_pool = TopLayerTranslatorPool(
        model_specs=model_specs,
        top_layers_to_translate=config.top_layers_to_translate,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        active_directions=active_directions,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs


def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    TopLayerTranslatorPool,
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
    translator_pool, model_specs = build_translator_pool(model_a, model_b, config)
    translator_pool.load_state_dict(payload["translator_pool"])
    translator_pool.to(config.device)
    translator_pool.eval()
    return config, translator_pool, model_specs, model_a, model_b, tokenizer


def run_train(config: TrainConfig) -> Path:
    if (
        config.output_dir is None
        or config.config_path is None
        or config.log_path is None
        or config.checkpoint_path is None
    ):
        raise ValueError("TrainConfig paths must be initialized before run_train.")

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.config_path, asdict(config))

    log_path = Path(config.log_path)
    logger = setup_logger(f"{config.alg}_train", log_path)
    logger.info("Starting training")
    logger.info("train_config=%s", asdict(config))

    train_directions = parse_train_directions(config.train_directions)
    logger.info("train_directions=%s", train_directions)

    logger.info("[Setup] device=%s", config.device)
    logger.info("[Setup] loading models: A=%s, B=%s", config.model_a_id, config.model_b_id)
    model_a, model_b, tokenizer = build_models_and_tokenizer(config)
    translator_pool, model_specs = build_translator_pool(model_a, model_b, config)
    translator_pool.train()

    logger.info("[Setup] full model specs")
    logger.info(
        "  A: layers=%d, hidden=%d, heads=%d",
        model_specs["A"].num_layers,
        model_specs["A"].hidden_size,
        model_specs["A"].num_heads,
    )
    logger.info(
        "  B: layers=%d, hidden=%d, heads=%d",
        model_specs["B"].num_layers,
        model_specs["B"].hidden_size,
        model_specs["B"].num_heads,
    )
    logger.info("[Setup] top_layers_to_translate = %d", config.top_layers_to_translate)
    logger.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")

    dataloader = build_training_dataloader(tokenizer, config)

    optimizer = torch.optim.AdamW(
        translator_pool.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    running_loss = 0.0
    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0

        for _ in range(config.grad_accum_steps):
            input_ids = next(dataloader).to(config.device)
            prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
                input_ids=input_ids,
                prefix_tokens=config.prefix_tokens,
            )

            with torch.no_grad():
                past_a = extract_past_key_values(model_a, prefix_cache_ids)
                past_b = extract_past_key_values(model_b, prefix_cache_ids)

            direction_contexts = {
                "A_to_B": {
                    "source_past": past_a,
                    "source_name": "A",
                    "target_name": "B",
                    "target_spec": model_specs["B"],
                    "target_full_past": past_b,
                    "target_model": model_b,
                },
                "B_to_A": {
                    "source_past": past_b,
                    "source_name": "B",
                    "target_name": "A",
                    "target_spec": model_specs["A"],
                    "target_full_past": past_a,
                    "target_model": model_a,
                },
            }

            total_direction_loss = 0.0
            for direction in train_directions:
                context = direction_contexts[direction]

                translated_top_past = translator_pool.translate_top_layers(
                    past_key_values=context["source_past"],
                    src_name=context["source_name"],
                    dst_name=context["target_name"],
                    dst_spec=context["target_spec"],
                )
                mixed_target_past = replace_top_layers(
                    base_past_key_values=context["target_full_past"],
                    translated_top_past_key_values=translated_top_past,
                )
                direction_loss = compute_suffix_lm_loss(
                    target_model=context["target_model"],
                    past_key_values=mixed_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                )
                total_direction_loss = total_direction_loss + direction_loss

            loss = total_direction_loss / config.grad_accum_steps
            loss.backward()
            step_loss_value += loss.item()

        torch.nn.utils.clip_grad_norm_(translator_pool.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += step_loss_value
        if step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            logger.info(
                "[Step %04d] total_suffix_lm_loss=%.4f | lr=%.2e | train_directions=%s",
                step,
                avg_loss,
                scheduler.lr,
                ",".join(train_directions),
            )
            running_loss = 0.0

    final_path = Path(config.checkpoint_path)
    save_checkpoint(
        output_path=str(final_path),
        translator_pool=translator_pool,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        step=config.max_steps,
        extra={
            "note": "Final checkpoint trained with suffix LM loss only.",
            "model_a": config.model_a_id,
            "model_b": config.model_b_id,
            "top_layers_to_translate": config.top_layers_to_translate,
            "train_directions": config.train_directions,
        },
    )
    logger.info("[Done] final checkpoint saved to %s", final_path)
    logger.info("Saved train log to %s", log_path)
    return final_path
