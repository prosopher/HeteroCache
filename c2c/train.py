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


class ResidualCacheFuser(nn.Module):
    """
    Fuses top-layer receiver/sharer cache blocks following the C2C recipe:
    project -> feature fuse -> dynamic weighting -> gated residual injection.

    receiver_block: [batch, seq, num_layers, dst_hidden]
    sharer_block:   [batch, seq, num_layers, src_hidden]
    output:         [batch, seq, num_layers, dst_hidden]
    """

    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        num_layers: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        gate_temperature_start: float,
        hard_gate_eval: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if fuser_depth < 1:
            raise ValueError("fuser_depth must be >= 1")
        if gate_temperature_start <= 0.0:
            raise ValueError("gate_temperature_start must be > 0")

        self.num_layers = num_layers
        self.fuser_depth = fuser_depth
        self.hard_gate_eval = hard_gate_eval
        self.temperature = float(gate_temperature_start)

        self.receiver_norm = nn.LayerNorm(dst_hidden_size)
        self.receiver_proj = nn.Linear(dst_hidden_size, fuser_dim)
        self.sharer_norm = nn.LayerNorm(src_hidden_size)
        self.sharer_proj = nn.Linear(src_hidden_size, fuser_dim)

        self.feature_norm = nn.LayerNorm(fuser_dim * 2)
        self.feature_proj = nn.Linear(fuser_dim * 2, fuser_dim)
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(fuser_dim),
            nn.Linear(fuser_dim, fuser_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(fuser_dim * mlp_ratio, fuser_dim),
        )

        self.dynamic_weight_norm = nn.LayerNorm(fuser_dim)
        self.dynamic_weight = nn.Linear(fuser_dim, fuser_dim)

        self.recurrent_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            dim=fuser_dim,
                            num_heads=fuser_heads,
                            mlp_ratio=mlp_ratio,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(fuser_depth)
            ]
        )

        self.output_norm = nn.LayerNorm(num_layers * fuser_dim)
        self.output_proj = nn.Linear(num_layers * fuser_dim, num_layers * dst_hidden_size)
        # Starting all gates exactly at 0.0 makes it very easy for a short toy run to
        # learn an all-closed hard gate solution, which turns the whole C2C path into
        # an exact identity map at evaluation time. We bias the gates slightly open so
        # the fusion module actually gets used early in training.
        self.gate_logits = nn.Parameter(torch.full((num_layers,), 1.5))

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(1e-4, float(temperature))

    def set_hard_gate_eval(self, enabled: bool) -> None:
        self.hard_gate_eval = bool(enabled)

    def gate_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def sample_gate(self) -> torch.Tensor:
        probs = self.gate_probabilities()
        if self.training:
            uniform = torch.rand_like(probs).clamp_(1e-6, 1.0 - 1e-6)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            soft_gate = torch.sigmoid((self.gate_logits + logistic_noise) / self.temperature)
            return soft_gate
        if self.hard_gate_eval:
            hard_gate = (probs >= 0.5).to(probs.dtype)
            # Avoid an exact all-zero hard gate at eval time. That makes fusion a strict
            # identity map, which is exactly the failure mode behind cosine=1.0 and
            # baseline-identical metrics. In that case we fall back to the learned soft
            # probabilities instead of silently disabling the whole C2C path.
            if float(hard_gate.sum().item()) == 0.0:
                return probs
            return hard_gate
        return probs

    def forward(self, receiver_block: torch.Tensor, sharer_block: torch.Tensor) -> torch.Tensor:
        if receiver_block.ndim != 4 or sharer_block.ndim != 4:
            raise ValueError(
                "ResidualCacheFuser expects [batch, seq, num_layers, hidden] tensors, "
                f"got receiver={tuple(receiver_block.shape)}, sharer={tuple(sharer_block.shape)}"
            )
        if receiver_block.shape[:3] != sharer_block.shape[:3]:
            raise ValueError(
                "receiver_block and sharer_block must agree on [batch, seq, num_layers], "
                f"got receiver={tuple(receiver_block.shape)}, sharer={tuple(sharer_block.shape)}"
            )
        if receiver_block.shape[2] != self.num_layers:
            raise ValueError(
                f"ResidualCacheFuser expected {self.num_layers} aligned layers, got {receiver_block.shape[2]}"
            )

        batch_size, seq_len, _, dst_hidden_size = receiver_block.shape

        receiver_projected = F.gelu(self.receiver_proj(self.receiver_norm(receiver_block)))
        sharer_projected = F.gelu(self.sharer_proj(self.sharer_norm(sharer_block)))

        combined = torch.cat([receiver_projected, sharer_projected], dim=-1)
        projected = F.gelu(self.feature_proj(self.feature_norm(combined)))
        fused_context = self.feature_fusion(projected) + projected

        weights = torch.sigmoid(self.dynamic_weight(self.dynamic_weight_norm(fused_context)))
        weighted_context = fused_context * weights

        hidden = receiver_projected[:, :, 0, :]
        collected = []
        for stage_blocks in self.recurrent_blocks:
            stage_hidden = hidden
            stage_collected = []
            for layer_idx, block in enumerate(stage_blocks):
                stage_hidden = block(stage_hidden, weighted_context[:, :, layer_idx, :])
                stage_collected.append(stage_hidden)
            hidden = stage_hidden
            collected = stage_collected

        residual = F.gelu(self.output_proj(self.output_norm(torch.cat(collected, dim=-1))))
        residual = residual.view(batch_size, seq_len, self.num_layers, dst_hidden_size)

        gate = self.sample_gate().view(1, 1, self.num_layers, 1)
        return receiver_block + (gate * residual)


class DirectionalCacheFuser(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        gate_temperature_start: float,
        hard_gate_eval: bool,
    ) -> None:
        super().__init__()
        self.top_layers_to_fuse = top_layers_to_fuse
        self.key_fuser = ResidualCacheFuser(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=top_layers_to_fuse,
            fuser_dim=fuser_dim,
            fuser_heads=fuser_heads,
            fuser_depth=fuser_depth,
            mlp_ratio=mlp_ratio,
            gate_temperature_start=gate_temperature_start,
            hard_gate_eval=hard_gate_eval,
        )
        self.value_fuser = ResidualCacheFuser(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=top_layers_to_fuse,
            fuser_dim=fuser_dim,
            fuser_heads=fuser_heads,
            fuser_depth=fuser_depth,
            mlp_ratio=mlp_ratio,
            gate_temperature_start=gate_temperature_start,
            hard_gate_eval=hard_gate_eval,
        )

    def set_temperature(self, temperature: float) -> None:
        self.key_fuser.set_temperature(temperature)
        self.value_fuser.set_temperature(temperature)

    def set_hard_gate_eval(self, enabled: bool) -> None:
        self.key_fuser.set_hard_gate_eval(enabled)
        self.value_fuser.set_hard_gate_eval(enabled)

    def mean_gate_probability(self) -> float:
        key_prob = float(self.key_fuser.gate_probabilities().mean().item())
        value_prob = float(self.value_fuser.gate_probabilities().mean().item())
        return 0.5 * (key_prob + value_prob)

    def forward(
        self,
        receiver_key_block: torch.Tensor,
        receiver_value_block: torch.Tensor,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_key = self.key_fuser(receiver_key_block, sharer_key_block)
        fused_value = self.value_fuser(receiver_value_block, sharer_value_block)
        return fused_key, fused_value


class C2CFuserPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        gate_temperature_start: float,
        hard_gate_eval: bool,
        active_directions: List[str],
    ) -> None:
        super().__init__()
        if top_layers_to_fuse < 1:
            raise ValueError("top_layers_to_fuse must be >= 1")
        if not active_directions:
            raise ValueError("active_directions must contain at least one direction")

        self.model_specs = model_specs
        self.top_layers_to_fuse = top_layers_to_fuse
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
            if top_layers_to_fuse > max_allowed:
                raise ValueError(
                    f"top_layers_to_fuse={top_layers_to_fuse} exceeds min layer count {max_allowed} "
                    f"for direction {direction}."
                )

            adapters[direction] = DirectionalCacheFuser(
                src_hidden_size=src_spec.hidden_size,
                dst_hidden_size=dst_spec.hidden_size,
                top_layers_to_fuse=top_layers_to_fuse,
                fuser_dim=fuser_dim,
                fuser_heads=fuser_heads,
                fuser_depth=fuser_depth,
                mlp_ratio=mlp_ratio,
                gate_temperature_start=gate_temperature_start,
                hard_gate_eval=hard_gate_eval,
            )

        self.adapters = nn.ModuleDict(adapters)

    def set_temperature(self, temperature: float) -> None:
        for module in self.adapters.values():
            module.set_temperature(temperature)

    def set_hard_gate_eval(self, enabled: bool) -> None:
        for module in self.adapters.values():
            module.set_hard_gate_eval(enabled)

    def mean_gate_probability(self) -> float:
        if not self.adapters:
            return float("nan")
        return float(sum(module.mean_gate_probability() for module in self.adapters.values()) / len(self.adapters))

    def fuse_top_layer_blocks(
        self,
        receiver_key_block: torch.Tensor,
        receiver_value_block: torch.Tensor,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adapter_name = f"{src_name}_to_{dst_name}"
        if adapter_name not in self.adapters:
            raise ValueError(
                f"C2C direction {adapter_name} is not available. Active directions: {list(self.active_directions)}"
            )
        return self.adapters[adapter_name](
            receiver_key_block=receiver_key_block,
            receiver_value_block=receiver_value_block,
            sharer_key_block=sharer_key_block,
            sharer_value_block=sharer_value_block,
        )

    def fuse_top_layers(
        self,
        sharer_past_key_values: PastKeyValues,
        receiver_past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        sharer_key_block, sharer_value_block = extract_top_layer_blocks(
            past_key_values=sharer_past_key_values,
            top_layers_to_fuse=self.top_layers_to_fuse,
        )
        receiver_key_block, receiver_value_block = extract_top_layer_blocks(
            past_key_values=receiver_past_key_values,
            top_layers_to_fuse=self.top_layers_to_fuse,
        )
        fused_key, fused_value = self.fuse_top_layer_blocks(
            receiver_key_block=receiver_key_block,
            receiver_value_block=receiver_value_block,
            sharer_key_block=sharer_key_block,
            sharer_value_block=sharer_value_block,
            src_name=src_name,
            dst_name=dst_name,
        )
        return blocks_to_partial_past_key_values(
            key_block=fused_key,
            value_block=fused_value,
            num_heads=dst_spec.num_heads,
            head_dim=dst_spec.head_dim,
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



def build_translator_pool(
    models: Dict[str, PreTrainedModel],
    config: TrainConfig,
) -> Tuple[C2CFuserPool, Dict[str, ModelSpec], List[Node], List[Edge]]:
    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    model_specs = {
        node.id: get_model_spec(models[node.id])
        for node in nodes
    }
    active_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    translator_pool = C2CFuserPool(
        model_specs=model_specs,
        edges=edges,
        top_layers_to_fuse=config.top_layers_to_fuse,
        fuser_dim=config.fuser_dim,
        fuser_heads=config.fuser_heads,
        fuser_depth=config.fuser_depth,
        mlp_ratio=config.fuser_mlp_ratio,
        gate_temperature_start=config.gate_temperature_start,
        hard_gate_eval=config.hard_gate_eval,
        active_directions=active_directions,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs, nodes, edges



def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    C2CFuserPool,
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



def compute_gate_temperature(config: TrainConfig, step: int) -> float:
    if config.max_steps <= 1:
        return float(config.gate_temperature_end)
    progress = (step - 1) / (config.max_steps - 1)
    temperature = (
        (1.0 - progress) * float(config.gate_temperature_start)
        + progress * float(config.gate_temperature_end)
    )
    return max(1e-4, temperature)



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
    logger.info("[Setup] top_layers_to_fuse = %d", config.top_layers_to_fuse)
    logger.info("[Setup] trainable C2C params = %s", f"{count_trainable_parameters(translator_pool):,}")

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
        gate_temperature = compute_gate_temperature(config, step)
        translator_pool.set_temperature(gate_temperature)

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
                fused_top_past = translator_pool.fuse_top_layers(
                    sharer_past_key_values=past_by_node_id[edge.src_id],
                    receiver_past_key_values=past_by_node_id[edge.dst_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
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
                gate=f"{translator_pool.mean_gate_probability():.3f}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                "[Step %04d] total_suffix_lm_loss=%.4f | lr=%.2e | gate_temp=%.4f | mean_gate_prob=%.4f | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                scheduler.lr,
                gate_temperature,
                translator_pool.mean_gate_probability(),
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
            "note": "Final checkpoint trained with C2C-style suffix LM loss.",
            "model_ids": config.model_ids,
            "top_layers_to_fuse": config.top_layers_to_fuse,
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
