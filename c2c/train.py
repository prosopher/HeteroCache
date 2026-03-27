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
    top_layers_to_fuse: int
    fuser_dim: int
    fuser_heads: int
    fuser_depth: int
    fuser_mlp_ratio: int
    gate_temperature_start: float
    gate_temperature_end: float
    hard_gate_eval: bool
    device: str
    dtype: str

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        parse_model_ids_csv(self.model_ids)
        initialize_train_output_paths(self)


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


class ResidualLayerCacheFuser(nn.Module):
    """
    C2C-style per-direction residual cache fuser.

    Inputs:
      receiver_block: [batch, seq, num_layers, receiver_hidden]
      sharer_block:   [batch, seq, num_layers, sharer_hidden]

    Outputs:
      delta_block:    [batch, seq, num_layers, receiver_hidden]
      gate_probs:     [num_layers]
    """

    def __init__(
        self,
        sharer_hidden_size: int,
        receiver_hidden_size: int,
        receiver_num_heads: int,
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        hard_gate_eval: bool,
    ) -> None:
        super().__init__()
        if top_layers_to_fuse < 1:
            raise ValueError("top_layers_to_fuse must be >= 1")
        if fuser_depth < 1:
            raise ValueError("fuser_depth must be >= 1")
        if receiver_hidden_size % receiver_num_heads != 0:
            raise ValueError("receiver_hidden_size must be divisible by receiver_num_heads")

        self.top_layers_to_fuse = top_layers_to_fuse
        self.receiver_hidden_size = receiver_hidden_size
        self.receiver_num_heads = receiver_num_heads
        self.receiver_head_dim = receiver_hidden_size // receiver_num_heads
        self.hard_gate_eval = hard_gate_eval

        self.receiver_norm = nn.LayerNorm(receiver_hidden_size)
        self.sharer_norm = nn.LayerNorm(sharer_hidden_size)
        self.receiver_proj = nn.Linear(receiver_hidden_size, fuser_dim)
        self.sharer_proj = nn.Linear(sharer_hidden_size, fuser_dim)
        self.pre_fuse = nn.Linear(fuser_dim * 2, fuser_dim)

        self.recurrent_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            dim=fuser_dim,
                            num_heads=fuser_heads,
                            mlp_ratio=mlp_ratio,
                        )
                        for _ in range(top_layers_to_fuse)
                    ]
                )
                for _ in range(fuser_depth)
            ]
        )

        self.output_norm = nn.LayerNorm(top_layers_to_fuse * fuser_dim)
        self.output_proj = nn.Linear(top_layers_to_fuse * fuser_dim, top_layers_to_fuse * receiver_hidden_size)

        self.head_modulator = nn.Sequential(
            nn.LayerNorm(receiver_hidden_size),
            nn.Linear(receiver_hidden_size, receiver_num_heads),
        )
        self.gate_logits = nn.Parameter(torch.zeros(top_layers_to_fuse))

    def _sample_gate(self, temperature: float) -> torch.Tensor:
        logits = self.gate_logits
        if self.training:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            gate = torch.sigmoid((logits + logistic) / max(temperature, 1e-6))
        else:
            gate = torch.sigmoid(logits / max(temperature, 1e-6))
            if self.hard_gate_eval:
                gate = (gate >= 0.5).to(gate.dtype)
        return gate

    def forward(
        self,
        receiver_block: torch.Tensor,
        sharer_block: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if receiver_block.ndim != 4 or sharer_block.ndim != 4:
            raise ValueError("Expected receiver_block and sharer_block to be rank-4 tensors.")
        if receiver_block.shape[:3] != sharer_block.shape[:3]:
            raise ValueError(
                "receiver_block and sharer_block must match in [batch, seq, layers], got "
                f"{tuple(receiver_block.shape)} vs {tuple(sharer_block.shape)}"
            )
        if receiver_block.shape[2] != self.top_layers_to_fuse:
            raise ValueError(
                f"Expected {self.top_layers_to_fuse} layers, got {receiver_block.shape[2]}"
            )

        receiver_proj = F.gelu(self.receiver_proj(self.receiver_norm(receiver_block)))
        sharer_proj = F.gelu(self.sharer_proj(self.sharer_norm(sharer_block)))
        fused_context = F.gelu(self.pre_fuse(torch.cat([receiver_proj, sharer_proj], dim=-1)))

        hidden = receiver_proj[:, :, 0, :]
        collected = []
        for stage_blocks in self.recurrent_blocks:
            stage_hidden = hidden
            stage_collected = []
            for layer_idx, block in enumerate(stage_blocks):
                context = fused_context[:, :, layer_idx, :]
                stage_hidden = block(stage_hidden, context)
                stage_collected.append(stage_hidden)
            hidden = stage_hidden
            collected = stage_collected

        fused = torch.cat(collected, dim=-1)
        delta = self.output_proj(self.output_norm(fused))
        batch_size, seq_len, _, _ = receiver_block.shape
        delta = delta.view(batch_size, seq_len, self.top_layers_to_fuse, self.receiver_hidden_size)

        head_weights = torch.sigmoid(self.head_modulator(receiver_block))
        head_weights = head_weights.repeat_interleave(self.receiver_head_dim, dim=-1)
        delta = delta * head_weights

        gate = self._sample_gate(temperature=temperature).view(1, 1, self.top_layers_to_fuse, 1)
        delta = delta * gate
        return delta, gate.view(-1)


class DirectionalCacheFuser(nn.Module):
    def __init__(
        self,
        sharer_hidden_size: int,
        receiver_hidden_size: int,
        receiver_num_heads: int,
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        hard_gate_eval: bool,
    ) -> None:
        super().__init__()
        self.key_fuser = ResidualLayerCacheFuser(
            sharer_hidden_size=sharer_hidden_size,
            receiver_hidden_size=receiver_hidden_size,
            receiver_num_heads=receiver_num_heads,
            top_layers_to_fuse=top_layers_to_fuse,
            fuser_dim=fuser_dim,
            fuser_heads=fuser_heads,
            fuser_depth=fuser_depth,
            mlp_ratio=mlp_ratio,
            hard_gate_eval=hard_gate_eval,
        )
        self.value_fuser = ResidualLayerCacheFuser(
            sharer_hidden_size=sharer_hidden_size,
            receiver_hidden_size=receiver_hidden_size,
            receiver_num_heads=receiver_num_heads,
            top_layers_to_fuse=top_layers_to_fuse,
            fuser_dim=fuser_dim,
            fuser_heads=fuser_heads,
            fuser_depth=fuser_depth,
            mlp_ratio=mlp_ratio,
            hard_gate_eval=hard_gate_eval,
        )

    def forward(
        self,
        receiver_key_block: torch.Tensor,
        receiver_value_block: torch.Tensor,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_key, key_gate = self.key_fuser(receiver_key_block, sharer_key_block, temperature=temperature)
        delta_value, value_gate = self.value_fuser(receiver_value_block, sharer_value_block, temperature=temperature)
        return delta_key, delta_value, key_gate, value_gate


class CacheFuserPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        hard_gate_eval: bool,
        active_directions: List[str],
    ) -> None:
        super().__init__()
        if not active_directions:
            raise ValueError("active_directions must contain at least one direction")
        self.model_specs = model_specs
        self.top_layers_to_fuse = top_layers_to_fuse
        self.active_directions = tuple(active_directions)
        self.edges_by_id = build_edge_map(edges)

        modules = {}
        for direction in self.active_directions:
            if direction not in self.edges_by_id:
                raise ValueError(f"Unknown direction: {direction}")
            edge = self.edges_by_id[direction]
            sharer_spec = model_specs[edge.src_id]
            receiver_spec = model_specs[edge.dst_id]
            max_allowed = min(sharer_spec.num_layers, receiver_spec.num_layers)
            if top_layers_to_fuse > max_allowed:
                raise ValueError(
                    f"top_layers_to_fuse={top_layers_to_fuse} exceeds min layer count {max_allowed} for {direction}"
                )
            modules[direction] = DirectionalCacheFuser(
                sharer_hidden_size=sharer_spec.hidden_size,
                receiver_hidden_size=receiver_spec.hidden_size,
                receiver_num_heads=receiver_spec.num_heads,
                top_layers_to_fuse=top_layers_to_fuse,
                fuser_dim=fuser_dim,
                fuser_heads=fuser_heads,
                fuser_depth=fuser_depth,
                mlp_ratio=mlp_ratio,
                hard_gate_eval=hard_gate_eval,
            )
        self.fusers = nn.ModuleDict(modules)

    def fuse_top_layer_blocks(
        self,
        receiver_key_block: torch.Tensor,
        receiver_value_block: torch.Tensor,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
        src_name: str,
        dst_name: str,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        module_name = f"{src_name}_to_{dst_name}"
        if module_name not in self.fusers:
            raise ValueError(
                f"Fuser direction {module_name} is not available. Active directions: {list(self.active_directions)}"
            )
        return self.fusers[module_name](
            receiver_key_block=receiver_key_block,
            receiver_value_block=receiver_value_block,
            sharer_key_block=sharer_key_block,
            sharer_value_block=sharer_value_block,
            temperature=temperature,
        )


def extract_top_layer_blocks(
    past_key_values: PastKeyValues,
    top_layers_to_fuse: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if top_layers_to_fuse < 1:
        raise ValueError("top_layers_to_fuse must be >= 1")
    if top_layers_to_fuse > len(past_key_values):
        raise ValueError(
            f"Cannot extract {top_layers_to_fuse} layers from cache with only {len(past_key_values)} layers."
        )
    return past_key_values_to_blocks(past_key_values[-top_layers_to_fuse:])



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



def build_fuser_pool(
    models: Dict[str, PreTrainedModel],
    config: TrainConfig,
) -> Tuple[CacheFuserPool, Dict[str, ModelSpec], List[Node], List[Edge]]:
    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    model_specs = {node.id: get_model_spec(models[node.id]) for node in nodes}
    active_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    fuser_pool = CacheFuserPool(
        model_specs=model_specs,
        edges=edges,
        top_layers_to_fuse=config.top_layers_to_fuse,
        fuser_dim=config.fuser_dim,
        fuser_heads=config.fuser_heads,
        fuser_depth=config.fuser_depth,
        mlp_ratio=config.fuser_mlp_ratio,
        hard_gate_eval=config.hard_gate_eval,
        active_directions=active_directions,
    )
    fuser_pool.to(config.device)
    return fuser_pool, model_specs, nodes, edges



def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    CacheFuserPool,
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
    fuser_pool, model_specs, _, _ = build_fuser_pool(models, config)
    fuser_pool.load_state_dict(payload["translator_pool"])
    fuser_pool.to(config.device)
    fuser_pool.eval()
    return config, fuser_pool, model_specs, models, tokenizer, nodes, edges



def compute_gate_temperature(config: TrainConfig, step: int) -> float:
    if config.max_steps <= 1:
        return config.gate_temperature_end
    progress = (step - 1) / (config.max_steps - 1)
    return (
        config.gate_temperature_start
        + progress * (config.gate_temperature_end - config.gate_temperature_start)
    )



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
    fuser_pool, model_specs, _, _ = build_fuser_pool(models, config)
    fuser_pool.train()

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
    logger.info("[Setup] top_layers_to_fuse = %d", config.top_layers_to_fuse)
    logger.info("[Setup] trainable fuser params = %s", f"{count_trainable_parameters(fuser_pool):,}")

    dataloader = build_training_dataloader(tokenizer, config)

    optimizer = torch.optim.AdamW(
        fuser_pool.parameters(),
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
    running_key_gate = 0.0
    running_value_gate = 0.0
    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0
        step_key_gate_value = 0.0
        step_value_gate_value = 0.0
        gate_temperature = compute_gate_temperature(config, step)

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
            total_key_gate = 0.0
            total_value_gate = 0.0
            for direction in model_directions:
                edge = edge_map[direction]
                sharer_key_block, sharer_value_block = extract_top_layer_blocks(
                    past_key_values=past_by_node_id[edge.src_id],
                    top_layers_to_fuse=config.top_layers_to_fuse,
                )
                receiver_key_block, receiver_value_block = extract_top_layer_blocks(
                    past_key_values=past_by_node_id[edge.dst_id],
                    top_layers_to_fuse=config.top_layers_to_fuse,
                )

                delta_key_block, delta_value_block, key_gate, value_gate = fuser_pool.fuse_top_layer_blocks(
                    receiver_key_block=receiver_key_block,
                    receiver_value_block=receiver_value_block,
                    sharer_key_block=sharer_key_block,
                    sharer_value_block=sharer_value_block,
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    temperature=gate_temperature,
                )

                fused_key_block = receiver_key_block + delta_key_block
                fused_value_block = receiver_value_block + delta_value_block
                fused_top_past = blocks_to_partial_past_key_values(
                    key_block=fused_key_block,
                    value_block=fused_value_block,
                    num_heads=model_specs[edge.dst_id].num_heads,
                    head_dim=model_specs[edge.dst_id].head_dim,
                )
                mixed_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_top_past_key_values=fused_top_past,
                )
                direction_loss = compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=mixed_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                )
                total_direction_loss = total_direction_loss + direction_loss
                total_key_gate = total_key_gate + key_gate.mean()
                total_value_gate = total_value_gate + value_gate.mean()

            loss = total_direction_loss / config.grad_accum_steps
            loss.backward()
            step_loss_value += loss.item()
            step_key_gate_value += (total_key_gate / len(model_directions)).item()
            step_value_gate_value += (total_value_gate / len(model_directions)).item()

        torch.nn.utils.clip_grad_norm_(fuser_pool.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        gpu_memory_tracker.update()

        running_loss += step_loss_value
        running_key_gate += step_key_gate_value / config.grad_accum_steps
        running_value_gate += step_value_gate_value / config.grad_accum_steps
        if step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            avg_key_gate = running_key_gate / config.log_every
            avg_value_gate = running_value_gate / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.lr:.2e}",
                temp=f"{gate_temperature:.3f}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                "[Step %04d] total_suffix_lm_loss=%.4f | gate_temp=%.4f | key_gate_mean=%.4f | value_gate_mean=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                gate_temperature,
                avg_key_gate,
                avg_value_gate,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0
            running_key_gate = 0.0
            running_value_gate = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_checkpoint(
        output_path=str(final_path),
        translator_pool=fuser_pool,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        step=config.max_steps,
        extra={
            "note": "Final checkpoint trained with C2C residual cache fusion and suffix LM loss.",
            "model_ids": config.model_ids,
            "model_directions": config.model_directions,
            "top_layers_to_fuse": config.top_layers_to_fuse,
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
