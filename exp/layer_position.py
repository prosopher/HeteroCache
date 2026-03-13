#!/usr/bin/env python3
import csv
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import *
from eval_util import *
from heterocache.train import *
from train_util import *
from transformers import AutoConfig


GENERATION_DATASET_SPECS = [
    HFDatasetSpec(
        name_for_log="SQuAD/validation",
        dataset_path="rajpurkar/squad",
        dataset_name=None,
        split="validation",
        answer_mode="squad",
        question_field="question",
        context_field="context",
        answers_field="answers",
        streaming=False,
    ),
]


@dataclass(frozen=True)
class LayerMapping:
    reference_direction: str
    reference_target_node_id: str
    reference_target_num_layers: int
    reference_target_layer_idx: int
    relative_depth: float
    src_layer_idx: int
    dst_layer_idx: int
    src_num_layers: int
    dst_num_layers: int


@dataclass
class LayerPositionConfig:
    model_ids: str = "gpt2,gpt2-medium"
    model_directions: str = "A_to_B"
    reference_direction: Optional[str] = None
    position_ratio: Optional[float] = None

    output_root: str = "outputs/layer_position"
    study_id: Optional[str] = None

    max_steps: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 1
    total_tokens: int = 128
    prefix_tokens: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip_norm: float = 1.0
    log_every: int = 10
    seed: int = 42
    shuffle_buffer: int = 10_000

    translator_dim: int = 768
    translator_heads: int = 12
    translator_depth: int = 2
    translator_mlp_ratio: int = 2

    device: str = "auto"
    dtype: str = "float32"

    eval_batch_size: int = 4
    eval_num_workers: int = 0
    eval_max_examples_per_dataset: int = 256
    eval_shuffle_stream: bool = False
    generation_max_new_tokens: int = 32

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.prefix_tokens < 2 or self.prefix_tokens >= self.total_tokens:
            raise ValueError("prefix_tokens must satisfy 2 <= prefix_tokens < total_tokens")
        if self.position_ratio is not None and not (0.0 <= self.position_ratio <= 1.0):
            raise ValueError("position_ratio must be in [0.0, 1.0]")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")


class SingleLayerDirectionalTranslator(nn.Module):
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
        self.key_layer = PerLayerTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
        self.value_layer = PerLayerTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, key_layer: torch.Tensor, value_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_layer(key_layer), self.value_layer(value_layer)


class SingleLayerTranslatorPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        layer_mappings: Dict[str, LayerMapping],
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        active_directions: List[str],
    ) -> None:
        super().__init__()
        self.model_specs = model_specs
        self.layer_mappings = layer_mappings
        self.active_directions = tuple(active_directions)
        self.edges_by_id = build_edge_map(edges)

        self.adapters = nn.ModuleDict(
            {
                direction: SingleLayerDirectionalTranslator(
                    src_hidden_size=model_specs[self.edges_by_id[direction].src_id].hidden_size,
                    dst_hidden_size=model_specs[self.edges_by_id[direction].dst_id].hidden_size,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                )
                for direction in self.active_directions
            }
        )

    def translate_single_layer(
        self,
        past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, LayerMapping]:
        direction = f"{src_name}_to_{dst_name}"
        mapping = self.layer_mappings[direction]
        key_layer, value_layer = extract_single_layer_block(past_key_values, mapping.src_layer_idx)
        translated_key = self.adapters[direction].key_layer(key_layer)
        translated_value = self.adapters[direction].value_layer(value_layer)
        return translated_key, translated_value, mapping


class F1Meter:
    def __init__(self) -> None:
        self.f1_sum = 0.0
        self.native_f1_sum = 0.0
        self.count = 0

    def update(self, f1_value: float, native_f1_value: float, n: int = 1) -> None:
        self.f1_sum += float(f1_value) * n
        self.native_f1_sum += float(native_f1_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "f1": float("nan"),
                "native_f1": float("nan"),
                "count": 0,
            }
        return {
            "f1": self.f1_sum / self.count,
            "native_f1": self.native_f1_sum / self.count,
            "count": self.count,
        }


class SimpleNamespaceConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def load_model_spec_from_pretrained_config(model_id: str) -> ModelSpec:
    config = AutoConfig.from_pretrained(model_id)
    num_heads = getattr(config, "n_head", None)
    hidden_size = getattr(config, "n_embd", None)
    num_layers = getattr(config, "n_layer", None)
    if num_heads is None or hidden_size is None or num_layers is None:
        raise ValueError("This experiment expects GPT-2 style configs with n_head/n_embd/n_layer.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    return ModelSpec(
        model_id=getattr(config, "_name_or_path", model_id),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
    )


def resolve_reference_direction_metadata(
    model_ids: str,
    model_directions: str,
    reference_direction: Optional[str],
) -> Tuple[List[Node], List[Edge], List[str], Edge]:
    nodes, edges = build_nodes_and_edges(model_ids, model_directions)
    active_directions = [edge.id for edge in edges]
    if not active_directions:
        raise ValueError("No active directions were resolved from model_directions")
    chosen_direction = reference_direction or active_directions[0]
    edge_map = build_edge_map(edges)
    if chosen_direction not in edge_map:
        raise ValueError(
            f"reference_direction={chosen_direction!r} is not available. Choices: {sorted(edge_map)}"
        )
    return nodes, edges, active_directions, edge_map[chosen_direction]


def relative_depth_from_layer_index(layer_idx: int, num_layers: int) -> float:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx={layer_idx} must be in [0, {num_layers - 1}]")
    if num_layers == 1:
        return 0.0
    return float(layer_idx) / float(num_layers - 1)


def relative_depth_to_layer_index(relative_depth: float, num_layers: int) -> int:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if num_layers == 1:
        return 0
    clamped = min(1.0, max(0.0, float(relative_depth)))
    return int(round(clamped * (num_layers - 1)))


def resolve_target_num_layers(
    model_ids: str,
    model_directions: str,
    reference_direction: Optional[str],
) -> int:
    nodes, _, _, reference_edge = resolve_reference_direction_metadata(
        model_ids=model_ids,
        model_directions=model_directions,
        reference_direction=reference_direction,
    )
    node_map = build_node_map(nodes)
    target_model_id = node_map[reference_edge.dst_id].model_id
    return load_model_spec_from_pretrained_config(target_model_id).num_layers


def build_layer_mappings(
    config: LayerPositionConfig,
    model_specs: Dict[str, ModelSpec],
    edges: List[Edge],
    active_directions: List[str],
    reference_edge: Edge,
) -> Dict[str, LayerMapping]:
    reference_target_spec = model_specs[reference_edge.dst_id]

    if config.position_ratio is None:
        raise ValueError("position_ratio must be set before building layer mappings")
    relative_depth = float(config.position_ratio)

    reference_target_layer_idx = relative_depth_to_layer_index(relative_depth, reference_target_spec.num_layers)
    edge_map = build_edge_map(edges)
    mappings: Dict[str, LayerMapping] = {}
    for direction in active_directions:
        edge = edge_map[direction]
        src_spec = model_specs[edge.src_id]
        dst_spec = model_specs[edge.dst_id]
        mappings[direction] = LayerMapping(
            reference_direction=reference_edge.id,
            reference_target_node_id=reference_edge.dst_id,
            reference_target_num_layers=reference_target_spec.num_layers,
            reference_target_layer_idx=reference_target_layer_idx,
            relative_depth=relative_depth,
            src_layer_idx=relative_depth_to_layer_index(relative_depth, src_spec.num_layers),
            dst_layer_idx=relative_depth_to_layer_index(relative_depth, dst_spec.num_layers),
            src_num_layers=src_spec.num_layers,
            dst_num_layers=dst_spec.num_layers,
        )
    return mappings


def extract_single_layer_block(
    past_key_values: PastKeyValues,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (0 <= layer_idx < len(past_key_values)):
        raise ValueError(f"layer_idx={layer_idx} must be in [0, {len(past_key_values) - 1}]")
    key, value = past_key_values[layer_idx]
    batch_size, num_heads, seq_len, head_dim = key.shape
    key_flat = key.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    value_flat = value.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    return key_flat, value_flat


def single_layer_blocks_to_past(
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> PastKeyValues:
    batch_size, seq_len, hidden_size = key_layer.shape
    expected_hidden_size = num_heads * head_dim
    if hidden_size != expected_hidden_size:
        raise ValueError(f"Hidden mismatch: got {hidden_size}, expected {expected_hidden_size}")
    key = key_layer.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    value = value_layer.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return ((key, value),)


def require_gpt2_transformer(model: PreTrainedModel):
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        raise ValueError(
            "layer_position.py currently supports GPT-2 style decoder stacks only "
            "(expected model.transformer.h to exist)."
        )
    return transformer


def build_gpt2_input_hidden_states(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    transformer = require_gpt2_transformer(model)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
    drop = getattr(transformer, "drop", None)
    if drop is not None:
        hidden_states = drop(hidden_states)
    return hidden_states


def unpack_block_outputs(outputs: Any) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(outputs, tuple):
        if len(outputs) < 2:
            raise ValueError("Expected GPT-2 block outputs to include present key/value cache.")
        hidden_states = outputs[0]
        present = outputs[1]
    else:
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = getattr(outputs, "hidden_states", None)
        present = getattr(outputs, "past_key_value", None)
        if hidden_states is None or present is None:
            raise ValueError("Unsupported block output type for GPT-2 layer replay.")
    if not isinstance(present, tuple) or len(present) != 2:
        raise ValueError("Expected present cache to be a (key, value) tuple.")
    return hidden_states, present


def run_gpt2_block_with_cache(block: nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    outputs = block(
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=True,
        output_attentions=False,
    )
    return unpack_block_outputs(outputs)


def run_gpt2_block_with_injected_layer(
    block: nn.Module,
    hidden_states: torch.Tensor,
    injected_key: torch.Tensor,
    injected_value: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if injected_key.shape != injected_value.shape:
        raise ValueError(
            "Injected key/value must have identical shapes, "
            f"got {tuple(injected_key.shape)} vs {tuple(injected_value.shape)}"
        )

    attn = block.attn
    residual = hidden_states
    attn_input = block.ln_1(hidden_states)

    qkv = attn.c_attn(attn_input)
    split_size = getattr(attn, "split_size", qkv.shape[-1] // 3)
    query, _, _ = qkv.split(split_size, dim=2)

    batch_size, seq_len, _ = query.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    expected_cache_shape = (batch_size, num_heads, seq_len, head_dim)
    if tuple(injected_key.shape) != expected_cache_shape:
        raise ValueError(
            "Injected cache shape mismatch for GPT-2 layer replay: "
            f"expected {expected_cache_shape}, got {tuple(injected_key.shape)}"
        )

    query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    if getattr(attn, "reorder_and_upcast_attn", False) and hasattr(attn, "_upcast_and_reordered_attn"):
        attn_output, _ = attn._upcast_and_reordered_attn(
            query,
            injected_key,
            injected_value,
            attention_mask=None,
            head_mask=None,
        )
    else:
        attn_output, _ = attn._attn(
            query,
            injected_key,
            injected_value,
            attention_mask=None,
            head_mask=None,
        )

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    attn_output = attn.c_proj(attn_output)
    attn_output = attn.resid_dropout(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
    return hidden_states, (injected_key, injected_value)


def replay_target_prefill_with_single_layer(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_layer_idx: int,
    injected_key_layer: torch.Tensor,
    injected_value_layer: torch.Tensor,
    dst_spec: ModelSpec,
) -> PastKeyValues:
    translated_key, translated_value = single_layer_blocks_to_past(
        injected_key_layer,
        injected_value_layer,
        dst_spec.num_heads,
        dst_spec.head_dim,
    )[0]

    transformer = require_gpt2_transformer(target_model)
    if not (0 <= target_layer_idx < len(transformer.h)):
        raise ValueError(f"target_layer_idx={target_layer_idx} must be in [0, {len(transformer.h) - 1}]")

    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if torch.is_grad_enabled():
        with torch.no_grad():
            hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
            for lower_idx in range(target_layer_idx):
                hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
                rebuilt_past.append((present[0].detach(), present[1].detach()))
        hidden_states = hidden_states.detach()
    else:
        hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
        for lower_idx in range(target_layer_idx):
            hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
            rebuilt_past.append(present)

    hidden_states, present = run_gpt2_block_with_injected_layer(
        transformer.h[target_layer_idx],
        hidden_states,
        translated_key,
        translated_value,
    )
    rebuilt_past.append(present)

    for upper_idx in range(target_layer_idx + 1, len(transformer.h)):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[upper_idx], hidden_states)
        rebuilt_past.append(present)

    return tuple(rebuilt_past)


def build_run_output_dir(config: LayerPositionConfig) -> Path:
    study_id = config.study_id or f"run_{sanitize_slug(config.model_directions)}"
    if config.position_ratio is None:
        raise ValueError("position_ratio must be set before building the run directory")
    position_label = f"ratio_{int(round(config.position_ratio * 100)):03d}"
    return Path(config.output_root) / study_id / position_label


def build_train_log_path(run_dir: Path) -> Path:
    return run_dir / "train.log"


def build_eval_log_path(run_dir: Path) -> Path:
    return run_dir / "eval.log"


def build_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "final_checkpoint.pt"


def build_config_path(run_dir: Path) -> Path:
    return run_dir / "experiment_config.json"


def build_layer_mapping_path(run_dir: Path) -> Path:
    return run_dir / "layer_mapping.json"


def build_metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics.json"


def build_summary_markdown_path(run_dir: Path) -> Path:
    return run_dir / "summary.md"


def build_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_position_f1.png"


def log_layer_mappings(
    logger: logging.Logger,
    nodes: List[Node],
    model_specs: Dict[str, ModelSpec],
    layer_mappings: Dict[str, LayerMapping],
) -> None:
    node_map = build_node_map(nodes)
    logger.info("[LayerMapping] reference direction = %s", next(iter(layer_mappings.values())).reference_direction)
    for direction, mapping in layer_mappings.items():
        src_id, dst_id = direction.split("_to_")
        logger.info(
            "[LayerMapping] %s | %s(%s): layer %d/%d -> %s(%s): layer %d/%d | relative_depth=%.4f",
            direction,
            src_id,
            node_map[src_id].model_id,
            mapping.src_layer_idx,
            model_specs[src_id].num_layers - 1,
            dst_id,
            node_map[dst_id].model_id,
            mapping.dst_layer_idx,
            model_specs[dst_id].num_layers - 1,
            mapping.relative_depth,
        )


def save_checkpoint(
    checkpoint_path: Path,
    translator_pool: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    config: LayerPositionConfig,
    step: int,
    layer_mappings: Dict[str, LayerMapping],
    model_specs: Dict[str, ModelSpec],
) -> None:
    payload = {
        "translator_pool": translator_pool.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": scheduler.step_id,
        "step": step,
        "experiment_config": asdict(config),
        "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        "model_specs": {node_id: asdict(spec) for node_id, spec in model_specs.items()},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def build_models_for_experiment(
    config: LayerPositionConfig,
) -> Tuple[Dict[str, PreTrainedModel], PreTrainedTokenizerBase, List[Node], List[Edge]]:
    return build_models_and_tokenizer(
        SimpleNamespaceConfig(
            model_ids=config.model_ids,
            model_directions=config.model_directions,
            device=config.device,
            dtype=config.dtype,
        )
    )


def build_translator_pool(
    models: Dict[str, PreTrainedModel],
    config: LayerPositionConfig,
    active_directions: List[str],
    edges: List[Edge],
    reference_edge: Edge,
) -> Tuple[SingleLayerTranslatorPool, Dict[str, ModelSpec], Dict[str, LayerMapping]]:
    model_specs = {node_id: get_model_spec(model) for node_id, model in models.items()}
    layer_mappings = build_layer_mappings(config, model_specs, edges, active_directions, reference_edge)
    translator_pool = SingleLayerTranslatorPool(
        model_specs=model_specs,
        edges=edges,
        layer_mappings=layer_mappings,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        active_directions=active_directions,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs, layer_mappings


def run_train(
    config: LayerPositionConfig,
    run_dir: Path,
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
    reference_edge: Edge,
) -> Tuple[SingleLayerTranslatorPool, Dict[str, ModelSpec], Dict[str, LayerMapping]]:
    logger = setup_logger(f"layer_position_train_{run_dir.name}", build_train_log_path(run_dir))
    logger.info("Starting single-layer position training with target-layer replay")
    logger.info("experiment_config=%s", asdict(config))

    translator_pool, model_specs, layer_mappings = build_translator_pool(
        models=models,
        config=config,
        active_directions=active_directions,
        edges=edges,
        reference_edge=reference_edge,
    )
    translator_pool.train()
    log_layer_mappings(logger, nodes, model_specs, layer_mappings)
    logger.info("[Setup] translator trainable params = %s", f"{count_trainable_parameters(translator_pool):,}")

    dataloader = build_training_dataloader(
        tokenizer=tokenizer,
        config=SimpleNamespaceConfig(
            total_tokens=config.total_tokens,
            batch_size=config.batch_size,
            shuffle_buffer=config.shuffle_buffer,
            seed=config.seed,
        ),
    )

    optimizer = torch.optim.AdamW(
        translator_pool.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_steps, config.max_steps)
    edge_map = build_edge_map(edges)
    gpu_memory_tracker = GPUMemoryTracker(config.device)
    running_loss = 0.0

    progress_bar = tqdm(range(1, config.max_steps + 1), desc="LayerPositionTrain")
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
            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_single_layer(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                mixed_target_past = replay_target_prefill_with_single_layer(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_layer_idx=mapping.dst_layer_idx,
                    injected_key_layer=translated_key,
                    injected_value_layer=translated_value,
                    dst_spec=model_specs[edge.dst_id],
                )
                total_direction_loss = total_direction_loss + compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=mixed_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                )

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
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.lr:.2e}")
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                "[Step %04d] single_layer_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0

    save_checkpoint(
        checkpoint_path=build_checkpoint_path(run_dir),
        translator_pool=translator_pool,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=config.max_steps,
        layer_mappings=layer_mappings,
        model_specs=model_specs,
    )
    logger.info("[Done] checkpoint saved to %s", build_checkpoint_path(run_dir))
    return translator_pool, model_specs, layer_mappings


@torch.inference_mode()
def evaluate_generation_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    config: LayerPositionConfig,
    translator_pool: SingleLayerTranslatorPool,
    model_specs: Dict[str, ModelSpec],
    models: Dict[str, PreTrainedModel],
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    edge_map = build_edge_map(edges)
    path_metrics = {direction: F1Meter() for direction in active_directions}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            context_prefix = prepare_generation_context_inputs(
                tokenizer=tokenizer,
                context=context_text,
                device=config.device,
            )
            cache_input_ids = context_prefix["input_ids"]

            question_prefix = prepare_generation_question_prefix(
                tokenizer=tokenizer,
                question=question,
                device=config.device,
            )
            question_cache_ids = question_prefix["cache_ids"]
            seed_token = question_prefix["seed_token"]

            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], cache_input_ids)
                for node in nodes
            }

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_single_layer(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                mixed_target_past = replay_target_prefill_with_single_layer(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=cache_input_ids,
                    target_layer_idx=mapping.dst_layer_idx,
                    injected_key_layer=translated_key,
                    injected_value_layer=translated_value,
                    dst_spec=model_specs[edge.dst_id],
                )

                translated_generation_past = append_input_ids_to_past(
                    model=models[edge.dst_id],
                    past_key_values=mixed_target_past,
                    input_ids=question_cache_ids,
                )
                native_generation_past = append_input_ids_to_past(
                    model=models[edge.dst_id],
                    past_key_values=past_by_node_id[edge.dst_id],
                    input_ids=question_cache_ids,
                )

                translated_answer = generate_greedy_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=translated_generation_past,
                    seed_token=seed_token,
                    max_new_tokens=config.generation_max_new_tokens,
                )
                native_answer = generate_greedy_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=native_generation_past,
                    seed_token=seed_token,
                    max_new_tokens=config.generation_max_new_tokens,
                )

                f1 = compute_generation_f1(translated_answer, gold_answers)
                native_f1 = compute_generation_f1(native_answer, gold_answers)
                path_metrics[direction].update(
                    f1_value=f1,
                    native_f1_value=native_f1,
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 25 == 0:
            logger.info("[%s] progress: %d/%d examples", spec.name_for_log, processed_examples, config.eval_max_examples_per_dataset)

    summarized = {direction: meter.summary() for direction, meter in path_metrics.items()}
    return {
        direction: {
            "f1": row["f1"],
            "native_f1": row["native_f1"],
            "count": row["count"],
        }
        for direction, row in summarized.items()
    }


def compute_average_metric(
    logit_results: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
) -> float:
    values = []
    for dataset_results in logit_results.values():
        for direction_results in dataset_results.values():
            value = direction_results.get(metric_key)
            if value is not None and value == value:
                values.append(float(value))
    if not values:
        return float("nan")
    return sum(values) / len(values)



def run_eval(
    config: LayerPositionConfig,
    run_dir: Path,
    translator_pool: SingleLayerTranslatorPool,
    model_specs: Dict[str, ModelSpec],
    layer_mappings: Dict[str, LayerMapping],
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
) -> Dict[str, Any]:
    logger = setup_logger(f"layer_position_eval_{run_dir.name}", build_eval_log_path(run_dir))
    logger.info("Starting single-layer position evaluation with target-layer replay")
    logger.info("experiment_config=%s", asdict(config))
    log_layer_mappings(logger, nodes, model_specs, layer_mappings)

    translator_pool.eval()
    for model in models.values():
        model.eval()

    eval_config = SimpleNamespaceConfig(
        batch_size=config.eval_batch_size,
        num_workers=config.eval_num_workers,
        max_examples_per_dataset=config.eval_max_examples_per_dataset,
        seed=config.seed,
        shuffle_eval_stream=config.eval_shuffle_stream,
        shuffle_buffer=config.shuffle_buffer,
    )

    generation_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in GENERATION_DATASET_SPECS:
        dataloader = build_eval_dataloader(spec=spec, eval_config=eval_config)
        dataset_results = evaluate_generation_dataset(
            spec=spec,
            dataloader=dataloader,
            tokenizer=tokenizer,
            config=config,
            translator_pool=translator_pool,
            model_specs=model_specs,
            models=models,
            nodes=nodes,
            edges=edges,
            active_directions=active_directions,
            logger=logger,
        )
        generation_results[spec.name_for_log] = dataset_results
        for direction in active_directions:
            row = dataset_results[direction]
            logger.info(
                "[%s] %s | f1=%.6f | native_f1=%.6f | count=%d",
                spec.name_for_log,
                direction,
                row["f1"],
                row["native_f1"],
                int(row["count"]),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    average_f1 = compute_average_metric(generation_results, "f1")
    average_native_f1 = compute_average_metric(generation_results, "native_f1")
    logger.info(
        "[Summary] average_f1=%.6f | average_native_f1=%.6f",
        average_f1,
        average_native_f1,
    )
    return {
        "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        "dataset_f1": generation_results,
        "average_f1": average_f1,
        "average_native_f1": average_native_f1,
    }



def build_summary_markdown(metrics: Dict[str, Any]) -> str:
    lines = [
        "# Layer Position Result",
        "",
        f"- average_f1: {metrics['average_f1']:.6f}",
        f"- average_native_f1: {metrics['average_native_f1']:.6f}",
        "",
        "## Dataset F1",
        "",
        "| Dataset | Direction | F1 | Native F1 | Count |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_name, dataset_results in metrics["dataset_f1"].items():
        for direction, row in dataset_results.items():
            lines.append(
                f"| {dataset_name} | {direction} | {row['f1']:.6f} | {row['native_f1']:.6f} | {int(row['count'])} |"
            )
    return "\n".join(lines)



def build_study_summary_row(
    config: LayerPositionConfig,
    run_dir: Path,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    reference_mapping = next(iter(metrics["layer_mappings"].values()))
    return {
        "study_id": config.study_id or "",
        "position_ratio": float(reference_mapping["relative_depth"]),
        "source_layer_idx": int(reference_mapping["src_layer_idx"]),
        "target_layer_idx": int(reference_mapping["dst_layer_idx"]),
        "average_f1": float(metrics["average_f1"]),
        "average_native_f1": float(metrics["average_native_f1"]),
        "run_dir": str(run_dir),
    }



def update_study_summary(
    config: LayerPositionConfig,
    run_dir: Path,
    metrics: Dict[str, Any],
) -> Path:
    study_dir = run_dir.parent
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_path = study_dir / "summary.csv"
    row = build_study_summary_row(config, run_dir, metrics)

    rows = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))

    filtered_rows = [
        existing
        for existing in rows
        if float(existing.get("position_ratio", "nan")) != row["position_ratio"]
    ]
    filtered_rows.append({key: str(value) for key, value in row.items()})
    filtered_rows.sort(key=lambda item: float(item["position_ratio"]))

    fieldnames = list(row.keys())
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    return summary_path



def plot_study_summary(study_dir: Path) -> Path:
    summary_path = study_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Study summary not found: {summary_path}")

    rows = []
    native_values = []
    with summary_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            try:
                rows.append((int(row["target_layer_idx"]), float(row["average_f1"])))
                native_values.append(float(row["average_native_f1"]))
            except (KeyError, ValueError):
                continue

    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda item: item[0])
    x_values = [item[0] for item in rows]
    y_values = [item[1] for item in rows]
    average_native_f1 = sum(native_values) / len(native_values) if native_values else float("nan")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, marker="o", label="Translated F1")
    if average_native_f1 == average_native_f1:
        ax.axhline(
            average_native_f1,
            linestyle="--",
            linewidth=1.5,
            label=f"Native F1 Mean ({average_native_f1:.4f})",
        )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("F1")
    ax.set_title("Layer Position vs F1")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    chart_path = build_chart_path(study_dir)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path



def save_run_artifacts(
    config: LayerPositionConfig,
    run_dir: Path,
    layer_mappings: Dict[str, LayerMapping],
    metrics: Dict[str, Any],
) -> Tuple[Path, Path]:
    write_json(str(build_config_path(run_dir)), asdict(config))
    write_json(str(build_layer_mapping_path(run_dir)), {direction: asdict(mapping) for direction, mapping in layer_mappings.items()})
    write_json(str(build_metrics_path(run_dir)), metrics)
    build_summary_markdown_path(run_dir).write_text(build_summary_markdown(metrics), encoding="utf-8")
    summary_path = update_study_summary(config, run_dir, metrics)
    return summary_path, build_metrics_path(run_dir)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single translated layer by replaying target prefill before the layer, injecting it, and continuing above it."
    )
    parser.add_argument("--model-ids", default="gpt2,gpt2-medium")
    parser.add_argument("--model-directions", default="A_to_B")
    parser.add_argument("--reference-direction", default=None)
    parser.add_argument("--position-ratio", type=float, default=None)
    parser.add_argument("--print-target-num-layers", action="store_true")
    parser.add_argument("--plot-study-summary", action="store_true")

    parser.add_argument("--output-root", default="outputs/layer_position")
    parser.add_argument("--study-id", default=None)

    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--total-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=50_000)

    parser.add_argument("--translator-dim", type=int, default=1024)
    parser.add_argument("--translator-heads", type=int, default=16)
    parser.add_argument("--translator-depth", type=int, default=2)
    parser.add_argument("--translator-mlp-ratio", type=int, default=2)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-max-examples-per-dataset", type=int, default=200)
    parser.add_argument("--eval-shuffle-stream", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if args.print_target_num_layers:
        print(resolve_target_num_layers(args.model_ids, args.model_directions, args.reference_direction))
        return

    study_id = args.study_id or f"run_{sanitize_slug(args.model_directions)}"
    study_dir = Path(args.output_root) / study_id
    if args.plot_study_summary:
        chart_path = plot_study_summary(study_dir)
        print(chart_path)
        return

    config = LayerPositionConfig(
        model_ids=args.model_ids,
        model_directions=args.model_directions,
        reference_direction=args.reference_direction,
        position_ratio=args.position_ratio,
        output_root=args.output_root,
        study_id=args.study_id,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        total_tokens=args.total_tokens,
        prefix_tokens=args.prefix_tokens,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        translator_dim=args.translator_dim,
        translator_heads=args.translator_heads,
        translator_depth=args.translator_depth,
        translator_mlp_ratio=args.translator_mlp_ratio,
        device=args.device,
        dtype=args.dtype,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        eval_max_examples_per_dataset=args.eval_max_examples_per_dataset,
        eval_shuffle_stream=args.eval_shuffle_stream,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )

    if config.position_ratio is None:
        raise SystemExit("--position-ratio is required unless --print-target-num-layers or --plot-study-summary is used.")

    set_seed(config.seed)
    run_dir = build_run_output_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges, active_directions, reference_edge = resolve_reference_direction_metadata(
        model_ids=config.model_ids,
        model_directions=config.model_directions,
        reference_direction=config.reference_direction,
    )
    models, tokenizer, _, _ = build_models_for_experiment(config)
    translator_pool, model_specs, layer_mappings = run_train(
        config=config,
        run_dir=run_dir,
        models=models,
        tokenizer=tokenizer,
        nodes=nodes,
        edges=edges,
        active_directions=active_directions,
        reference_edge=reference_edge,
    )
    metrics = run_eval(
        config=config,
        run_dir=run_dir,
        translator_pool=translator_pool,
        model_specs=model_specs,
        layer_mappings=layer_mappings,
        models=models,
        tokenizer=tokenizer,
        nodes=nodes,
        edges=edges,
        active_directions=active_directions,
    )
    summary_path, metrics_path = save_run_artifacts(config, run_dir, layer_mappings, metrics)

    print(f"Run directory: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Study summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
