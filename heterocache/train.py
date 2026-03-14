from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from train_util import *


@dataclass
class TrainConfig:
    alg: str
    outputs_path: str
    timestamp: Optional[str]
    output_path: Optional[str]

    model_ids: str
    model_directions: str
    max_steps: int
    batch_size: int
    grad_accum_steps: int
    total_tokens: int
    prefix_tokens: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip_norm: float
    log_every: int
    seed: int
    shuffle_buffer: int
    bottom_layers_to_translate: int
    translator_dim: int
    translator_heads: int
    translator_depth: int
    translator_mlp_ratio: int
    device: str
    dtype: str

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        parse_model_ids_csv(self.model_ids)
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


class BottomLayerDirectionalTranslator(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        bottom_layers_to_translate: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.bottom_layers_to_translate = bottom_layers_to_translate
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
                for _ in range(bottom_layers_to_translate)
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
                for _ in range(bottom_layers_to_translate)
            ]
        )

    def forward(self, key_block: torch.Tensor, value_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_outputs = []
        value_outputs = []
        for layer_idx in range(self.bottom_layers_to_translate):
            key_outputs.append(self.key_layers[layer_idx](key_block[:, :, layer_idx, :]).unsqueeze(2))
            value_outputs.append(self.value_layers[layer_idx](value_block[:, :, layer_idx, :]).unsqueeze(2))
        translated_key = torch.cat(key_outputs, dim=2)
        translated_value = torch.cat(value_outputs, dim=2)
        return translated_key, translated_value


class BottomLayerTranslatorPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        bottom_layers_to_translate: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        active_directions: List[str],
    ) -> None:
        super().__init__()
        if bottom_layers_to_translate < 1:
            raise ValueError("bottom_layers_to_translate must be >= 1")
        if not active_directions:
            raise ValueError("active_directions must contain at least one direction.")

        self.model_specs = model_specs
        self.bottom_layers_to_translate = bottom_layers_to_translate
        self.active_directions = tuple(active_directions)
        self.edges_by_id = build_edge_map(edges)

        adapters = {}
        for direction in self.active_directions:
            if direction not in self.edges_by_id:
                raise ValueError(f"Unknown direction: {direction}")
            edge = self.edges_by_id[direction]
            src_spec = model_specs[edge.src_id]
            dst_spec = model_specs[edge.dst_id]
            max_allowed = min(src_spec.num_layers, dst_spec.num_layers) - 1
            if bottom_layers_to_translate > max_allowed:
                raise ValueError(
                    f"bottom_layers_to_translate={bottom_layers_to_translate} exceeds replaceable layer count "
                    f"{max_allowed} above layer 0 for direction {direction}."
                )
            adapters[direction] = BottomLayerDirectionalTranslator(
                src_hidden_size=src_spec.hidden_size,
                dst_hidden_size=dst_spec.hidden_size,
                bottom_layers_to_translate=bottom_layers_to_translate,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
            )

        self.adapters = nn.ModuleDict(adapters)

    def translate_bottom_layer_blocks(
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

    def translate_bottom_layers(
        self,
        past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        key_block, value_block = extract_bottom_layer_blocks(
            past_key_values=past_key_values,
            bottom_layers_to_translate=self.bottom_layers_to_translate,
        )
        translated_key, translated_value = self.translate_bottom_layer_blocks(
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


def extract_bottom_layer_blocks(
    past_key_values: PastKeyValues,
    bottom_layers_to_translate: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bottom_layers_to_translate < 1:
        raise ValueError("bottom_layers_to_translate must be >= 1")
    max_allowed = len(past_key_values) - 1
    if bottom_layers_to_translate > max_allowed:
        raise ValueError(
            f"Cannot extract {bottom_layers_to_translate} bottom layers from cache with only {len(past_key_values)} layers "
            f"when layer 0 is excluded."
        )
    return past_key_values_to_blocks(past_key_values[1 : 1 + bottom_layers_to_translate])


def slice_bottom_layers(
    past_key_values: PastKeyValues,
    bottom_layers_to_translate: int,
) -> PastKeyValues:
    if bottom_layers_to_translate < 1:
        raise ValueError("bottom_layers_to_translate must be >= 1")
    max_allowed = len(past_key_values) - 1
    if bottom_layers_to_translate > max_allowed:
        raise ValueError(
            f"Cannot slice {bottom_layers_to_translate} bottom layers from cache with only {len(past_key_values)} layers "
            f"when layer 0 is excluded."
        )
    return tuple(past_key_values[1 : 1 + bottom_layers_to_translate])


def replace_bottom_layers(
    base_past_key_values: PastKeyValues,
    translated_bottom_past_key_values: PastKeyValues,
) -> PastKeyValues:
    num_replace = len(translated_bottom_past_key_values)
    if num_replace < 1:
        raise ValueError("translated_bottom_past_key_values must contain at least one layer.")
    max_allowed = len(base_past_key_values) - 1
    if num_replace > max_allowed:
        raise ValueError(
            f"Cannot replace {num_replace} bottom layers in cache with only {len(base_past_key_values)} layers "
            f"when layer 0 is excluded."
        )

    base_list = list(base_past_key_values)

    for offset, translated_layer in enumerate(translated_bottom_past_key_values, start=1):
        base_key, base_value = base_list[offset]
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

        base_list[offset] = (translated_key, translated_value)

    return tuple(base_list)


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
    models: Dict[str, PreTrainedModel],
    config: TrainConfig,
) -> Tuple[BottomLayerTranslatorPool, Dict[str, ModelSpec], List[Node], List[Edge]]:
    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    model_specs = {
        node.id: get_model_spec(models[node.id])
        for node in nodes
    }
    active_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    translator_pool = BottomLayerTranslatorPool(
        model_specs=model_specs,
        edges=edges,
        bottom_layers_to_translate=config.bottom_layers_to_translate,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        active_directions=active_directions,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs, nodes, edges


def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    BottomLayerTranslatorPool,
    Dict[str, ModelSpec],
    Dict[str, PreTrainedModel],
    PreTrainedTokenizerBase,
    List[Node],
    List[Edge],
]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**payload["train_config"])
    if device_override is not None:
        config.device = device_override
    models, tokenizer, nodes, edges = build_models_and_tokenizer(config)
    translator_pool, model_specs, _, _ = build_translator_pool(models, config)
    translator_pool.load_state_dict(payload["translator_pool"])
    translator_pool.to(config.device)
    translator_pool.eval()
    return config, translator_pool, model_specs, models, tokenizer, nodes, edges


def run_train(config: TrainConfig) -> Path:
    if config.output_path is None:
        raise ValueError("TrainConfig.output_path must be initialized before run_train.")

    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    edge_map = build_edge_map(edges)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logger = setup_logger(f"{config.alg}_train", log_path)
    logger.info("Starting training")
    logger.info("train_config=%s", asdict(config))

    model_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    logger.info("nodes=%s", [asdict(node) for node in nodes])
    logger.info("model_directions=%s", model_directions)

    logger.info("[Setup] device=%s", config.device)
    logger.info("[Setup] loading models: %s", {node.id: node.model_id for node in nodes})
    models, tokenizer, _, _ = build_models_and_tokenizer(config)
    translator_pool, model_specs, _, _ = build_translator_pool(models, config)
    translator_pool.train()

    logger.info("[Setup] full model specs")
    for node in nodes:
        spec = model_specs[node.id]
        logger.info(
            "  %s (%s): layers=%d, hidden=%d, heads=%d",
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
        )
    logger.info("[Setup] bottom_layers_to_translate = %d", config.bottom_layers_to_translate)
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
                past_by_node_id = {
                    node.id: extract_past_key_values(models[node.id], prefix_cache_ids)
                    for node in nodes
                }

            total_direction_loss = 0.0
            for direction in model_directions:
                edge = edge_map[direction]
                translated_bottom_past = translator_pool.translate_bottom_layers(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
                )
                mixed_target_past = replace_bottom_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_bottom_past_key_values=translated_bottom_past,
                )
                direction_loss = compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
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
            "model_ids": config.model_ids,
            "bottom_layers_to_translate": config.bottom_layers_to_translate,
            "model_directions": config.model_directions,
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
