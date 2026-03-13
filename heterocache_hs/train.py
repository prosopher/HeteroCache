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
    top_layers_to_translate: int
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

    def forward(self, hidden_block: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_proj(self.input_norm(hidden_block)))
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
        self.hidden_layers = nn.ModuleList(
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

    def forward(self, hidden_block: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer_idx in range(self.top_layers_to_translate):
            outputs.append(self.hidden_layers[layer_idx](hidden_block[:, :, layer_idx, :]).unsqueeze(2))
        return torch.cat(outputs, dim=2)


class TopLayerTranslatorPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        top_layers_to_translate: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        active_directions: List[str],
    ) -> None:
        super().__init__()
        if top_layers_to_translate < 1:
            raise ValueError("top_layers_to_translate must be >= 1")
        if not active_directions:
            raise ValueError("active_directions must contain at least one direction.")

        self.model_specs = model_specs
        self.top_layers_to_translate = top_layers_to_translate
        self.active_directions = tuple(active_directions)
        self.edges_by_id = build_edge_map(edges)

        adapters = {}
        for direction in self.active_directions:
            if direction not in self.edges_by_id:
                raise ValueError(f"Unknown direction: {direction}")
            edge = self.edges_by_id[direction]
            src_spec = model_specs[edge.src_id]
            dst_spec = model_specs[edge.dst_id]
            max_allowed = min(src_spec.num_layers, dst_spec.num_layers)
            if top_layers_to_translate > max_allowed:
                raise ValueError(
                    f"top_layers_to_translate={top_layers_to_translate} exceeds min layer count {max_allowed} "
                    f"for direction {direction}."
                )
            adapters[direction] = TopLayerDirectionalTranslator(
                src_hidden_size=src_spec.hidden_size,
                dst_hidden_size=dst_spec.hidden_size,
                top_layers_to_translate=top_layers_to_translate,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
            )

        self.adapters = nn.ModuleDict(adapters)

    def translate_top_hidden_states(
        self,
        hidden_block: torch.Tensor,
        src_name: str,
        dst_name: str,
    ) -> torch.Tensor:
        adapter_name = f"{src_name}_to_{dst_name}"
        if adapter_name not in self.adapters:
            raise ValueError(
                f"Translator direction {adapter_name} is not available. "
                f"Active directions: {list(self.active_directions)}"
            )
        return self.adapters[adapter_name](hidden_block)


def extract_top_hidden_state_block(
    hidden_states: Tuple[torch.Tensor, ...],
    num_layers: int,
    top_layers_to_translate: int,
) -> torch.Tensor:
    if top_layers_to_translate < 1:
        raise ValueError("top_layers_to_translate must be >= 1")
    if len(hidden_states) < num_layers:
        raise ValueError(
            f"Expected at least {num_layers} hidden-state entries, got {len(hidden_states)}."
        )
    if top_layers_to_translate > num_layers:
        raise ValueError(
            f"Cannot extract {top_layers_to_translate} layers from model with only {num_layers} layers."
        )

    start_layer = num_layers - top_layers_to_translate
    selected_layers = [hidden_states[layer_idx] for layer_idx in range(start_layer, num_layers)]
    return torch.stack(selected_layers, dim=2)


@torch.no_grad()
def extract_past_key_values_and_top_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    top_layers_to_translate: int,
) -> Tuple[PastKeyValues, torch.Tensor]:
    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    spec = get_model_spec(model)
    hidden_block = extract_top_hidden_state_block(
        hidden_states=outputs.hidden_states,
        num_layers=spec.num_layers,
        top_layers_to_translate=top_layers_to_translate,
    )
    return outputs.past_key_values, hidden_block


def _project_hidden_states_to_kv(
    hidden_states: torch.Tensor,
    transformer_block: nn.Module,
    num_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(transformer_block, "attn") or not hasattr(transformer_block.attn, "c_attn"):
        raise ValueError("Expected a GPT-2 style transformer block with attn.c_attn.")

    pre_attn_hidden = hidden_states
    if hasattr(transformer_block, "ln_1"):
        pre_attn_hidden = transformer_block.ln_1(pre_attn_hidden)

    qkv = transformer_block.attn.c_attn(pre_attn_hidden)
    split_size = num_heads * head_dim
    _, key, value = qkv.split(split_size, dim=2)

    batch_size, seq_len, _ = key.shape
    key = key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    value = value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return key, value


def restore_top_layer_past_from_hidden_states(
    model: PreTrainedModel,
    hidden_block: torch.Tensor,
    dst_spec: ModelSpec,
    top_layers_to_translate: int,
) -> PastKeyValues:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("This implementation expects a GPT-2 style model with transformer.h blocks.")

    batch_size, seq_len, num_layers, hidden_size = hidden_block.shape
    if num_layers != top_layers_to_translate:
        raise ValueError(
            f"Hidden block contains {num_layers} layers, expected {top_layers_to_translate}."
        )
    if hidden_size != dst_spec.hidden_size:
        raise ValueError(
            f"Hidden size mismatch: hidden block has {hidden_size}, target model expects {dst_spec.hidden_size}."
        )

    start_layer = dst_spec.num_layers - top_layers_to_translate
    restored_past = []
    for offset in range(top_layers_to_translate):
        layer_idx = start_layer + offset
        transformer_block = model.transformer.h[layer_idx]
        key, value = _project_hidden_states_to_kv(
            hidden_states=hidden_block[:, :, offset, :],
            transformer_block=transformer_block,
            num_heads=dst_spec.num_heads,
            head_dim=dst_spec.head_dim,
        )
        restored_past.append((key, value))
    return tuple(restored_past)


def build_translator_pool(
    models: Dict[str, PreTrainedModel],
    config: TrainConfig,
) -> Tuple[TopLayerTranslatorPool, Dict[str, ModelSpec], List[Node], List[Edge]]:
    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    model_specs = {
        node.id: get_model_spec(models[node.id])
        for node in nodes
    }
    active_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    translator_pool = TopLayerTranslatorPool(
        model_specs=model_specs,
        edges=edges,
        top_layers_to_translate=config.top_layers_to_translate,
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
    TopLayerTranslatorPool,
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
    logger.info("[Setup] top_layers_to_translate = %d", config.top_layers_to_translate)
    logger.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")
    logger.info("[Setup] translation_mode = hidden_states_to_kv_restore_for_top_layers")

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
                prefix_state_by_node_id = {
                    node.id: extract_past_key_values_and_top_hidden_states(
                        model=models[node.id],
                        input_ids=prefix_cache_ids,
                        top_layers_to_translate=config.top_layers_to_translate,
                    )
                    for node in nodes
                }
                past_by_node_id = {
                    node_id: state[0]
                    for node_id, state in prefix_state_by_node_id.items()
                }
                hidden_by_node_id = {
                    node_id: state[1]
                    for node_id, state in prefix_state_by_node_id.items()
                }

            total_direction_loss = 0.0
            for direction in model_directions:
                edge = edge_map[direction]
                translated_top_hidden = translator_pool.translate_top_hidden_states(
                    hidden_block=hidden_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                translated_top_past = restore_top_layer_past_from_hidden_states(
                    model=models[edge.dst_id],
                    hidden_block=translated_top_hidden,
                    dst_spec=model_specs[edge.dst_id],
                    top_layers_to_translate=config.top_layers_to_translate,
                )
                mixed_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_top_past_key_values=translated_top_past,
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
            "note": "Final checkpoint trained by translating top-layer hidden states and restoring KV during evaluation.",
            "model_ids": config.model_ids,
            "top_layers_to_translate": config.top_layers_to_translate,
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
