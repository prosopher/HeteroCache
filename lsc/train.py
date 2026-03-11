from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import *


@dataclass
class TrainConfig:
    alg: str = ""
    outputs_path: str = "outputs"
    timestamp: Optional[str] = None
    output_path: Optional[str] = None

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
    shared_slots: int = 32
    shared_dim: int = 128
    translator_dim: int = 256
    translator_heads: int = 4
    translator_mlp_ratio: int = 2
    top_layers_ratio: float = 1.0
    train_directions: str = "A_to_B,B_to_A"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        initialize_train_output_paths(self)


@dataclass
class SharedCache:
    key: torch.Tensor
    value: torch.Tensor


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


def run_train(config: TrainConfig) -> Path:
    if config.output_path is None:
        raise ValueError("TrainConfig.output_path must be initialized before run_train.")

    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logger = setup_logger(f"{config.alg}_train", log_path)
    logger.info("Starting training")
    logger.info("train_config=%s", asdict(config))

    train_directions = parse_train_directions(config.train_directions)
    logger.info("train_directions=%s", train_directions)

    logger.info("[Setup] device=%s", config.device)
    logger.info("[Setup] loading models: A=%s, B=%s", config.model_a_id, config.model_b_id)
    model_a, model_b, tokenizer = build_models_and_tokenizer(config)
    translator_pool, model_specs, translated_model_specs = build_translator_pool(model_a, model_b, config)
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
    logger.info("[Setup] translated top-layer specs")
    logger.info(
        "  A_top: layers=%d / %d",
        translated_model_specs["A"].num_layers,
        model_specs["A"].num_layers,
    )
    logger.info(
        "  B_top: layers=%d / %d",
        translated_model_specs["B"].num_layers,
        model_specs["B"].num_layers,
    )
    logger.info("[Setup] top_layers_ratio = %.4f", config.top_layers_ratio)
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

    gpu_memory_tracker = GPUMemoryTracker(config.device)

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
                    "target_top_spec": translated_model_specs["B"],
                    "target_full_past": past_b,
                    "target_model": model_b,
                },
                "B_to_A": {
                    "source_past": past_b,
                    "source_name": "B",
                    "target_name": "A",
                    "target_top_spec": translated_model_specs["A"],
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
                    dst_spec=context["target_top_spec"],
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
        gpu_memory_tracker.update()

        running_loss += step_loss_value
        if step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                "[Step %04d] total_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0

    final_path = get_train_checkpoint_path(output_path)
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
            "top_layers_ratio": config.top_layers_ratio,
            "train_directions": config.train_directions,
        },
    )
    final_gpu_memory = gpu_memory_tracker.summary()
    logger.info(
        "[Memory] avg_gpu_mem=%s | peak_gpu_mem=%s | samples=%d",
        final_gpu_memory["avg_allocated_pretty"],
        final_gpu_memory["peak_allocated_pretty"],
        final_gpu_memory["num_samples"],
    )
    logger.info("[Done] final checkpoint saved to %s", final_path)
    logger.info("Saved train log to %s", log_path)
    return final_path
