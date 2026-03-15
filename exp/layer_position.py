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


LOGIT_QA_DATASET_SPECS = [
    HFDatasetSpec(
        name_for_log="BoolQ/validation",
        dataset_path="google/boolq",
        dataset_name=None,
        split="validation",
        answer_mode="boolq",
        question_field="question",
        context_field="passage",
        streaming=False,
    ),
    HFDatasetSpec(
        name_for_log="PubMedQA/pqa_labeled/train",
        dataset_path="qiaojin/PubMedQA",
        dataset_name="pqa_labeled",
        split="train",
        answer_mode="pubmed_qa",
        question_field="question",
        context_field="context",
        streaming=False,
    ),
]

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
    reference_target_layer_end_idx: int
    relative_depth: float
    src_layer_idx: int
    src_layer_end_idx: int
    dst_layer_idx: int
    dst_layer_end_idx: int
    translated_num_layers: int
    src_num_layers: int
    dst_num_layers: int


@dataclass
class LayerPositionConfig:
    model_ids: str = "gpt2,gpt2-medium"
    model_directions: str = "A_to_B"
    reference_direction: Optional[str] = None
    position_layer_idx: Optional[int] = None
    injection_window_size: int = 1

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
    benchmark_mode: str = "squad_f1"
    generation_max_new_tokens: int = 32
    extractive_max_answer_tokens: int = 16

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.prefix_tokens < 2 or self.prefix_tokens >= self.total_tokens:
            raise ValueError("prefix_tokens must satisfy 2 <= prefix_tokens < total_tokens")
        if self.position_layer_idx is not None and self.position_layer_idx < 0:
            raise ValueError("position_layer_idx must be >= 0")
        if self.injection_window_size < 1:
            raise ValueError("injection_window_size must be >= 1")
        if self.benchmark_mode not in {"qa_accuracy", "squad_f1"}:
            raise ValueError("benchmark_mode must be one of {'qa_accuracy', 'squad_f1'}")
        if self.extractive_max_answer_tokens < 1:
            raise ValueError("extractive_max_answer_tokens must be >= 1")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")


class LayerWindowDirectionalTranslator(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        translated_num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if translated_num_layers < 1:
            raise ValueError("translated_num_layers must be >= 1")
        self.translated_num_layers = translated_num_layers
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
                for _ in range(translated_num_layers)
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
                for _ in range(translated_num_layers)
            ]
        )

    def forward(self, key_block: torch.Tensor, value_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_block.shape != value_block.shape:
            raise ValueError(
                "Layer-window key/value shapes must match, "
                f"got {tuple(key_block.shape)} vs {tuple(value_block.shape)}"
            )
        if key_block.ndim != 4:
            raise ValueError(
                "Layer-window tensors must have shape [batch, seq, num_layers, hidden], "
                f"got {tuple(key_block.shape)}"
            )
        if key_block.shape[2] != self.translated_num_layers:
            raise ValueError(
                f"Expected {self.translated_num_layers} layers in the translation window, got {key_block.shape[2]}"
            )

        translated_key_layers = []
        translated_value_layers = []
        for offset in range(self.translated_num_layers):
            translated_key_layers.append(self.key_layers[offset](key_block[:, :, offset, :]).unsqueeze(2))
            translated_value_layers.append(self.value_layers[offset](value_block[:, :, offset, :]).unsqueeze(2))
        return torch.cat(translated_key_layers, dim=2), torch.cat(translated_value_layers, dim=2)


class LayerWindowTranslatorPool(nn.Module):
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

        adapters = {}
        for direction in self.active_directions:
            mapping = self.layer_mappings[direction]
            adapters[direction] = LayerWindowDirectionalTranslator(
                src_hidden_size=model_specs[self.edges_by_id[direction].src_id].hidden_size,
                dst_hidden_size=model_specs[self.edges_by_id[direction].dst_id].hidden_size,
                translated_num_layers=mapping.translated_num_layers,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
            )
        self.adapters = nn.ModuleDict(adapters)

    def translate_layer_window(
        self,
        past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, LayerMapping]:
        direction = f"{src_name}_to_{dst_name}"
        mapping = self.layer_mappings[direction]
        key_block, value_block = extract_layer_window_blocks(
            past_key_values=past_key_values,
            start_layer_idx=mapping.src_layer_idx,
            num_layers=mapping.translated_num_layers,
        )
        translated_key, translated_value = self.adapters[direction](key_block, value_block)
        return translated_key, translated_value, mapping


class AccuracyMeter:
    def __init__(self) -> None:
        self.accuracy_sum = 0.0
        self.native_accuracy_sum = 0.0
        self.count = 0

    def update(self, accuracy_value: float, native_accuracy_value: float, n: int = 1) -> None:
        self.accuracy_sum += float(accuracy_value) * n
        self.native_accuracy_sum += float(native_accuracy_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "accuracy": float("nan"),
                "native_accuracy": float("nan"),
                "count": 0,
            }
        return {
            "accuracy": self.accuracy_sum / self.count,
            "native_accuracy": self.native_accuracy_sum / self.count,
            "count": self.count,
        }


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


def resolve_reference_target_layer_idx(config: LayerPositionConfig, reference_target_num_layers: int) -> int:
    if reference_target_num_layers < 1:
        raise ValueError("reference_target_num_layers must be >= 1")
    layer_idx = int(config.position_layer_idx)
    if not (0 <= layer_idx < reference_target_num_layers):
        raise ValueError(
            f"position_layer_idx={layer_idx} must be in [0, {reference_target_num_layers - 1}] for the reference target"
        )
    return layer_idx


def resolve_run_position_label(config: LayerPositionConfig) -> str:
    return f"layer_idx_{int(config.position_layer_idx):03d}"


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
    requested_window_size = int(config.injection_window_size)

    reference_target_layer_idx = resolve_reference_target_layer_idx(config, reference_target_spec.num_layers)
    reference_target_layer_end_idx = reference_target_layer_idx + requested_window_size - 1
    if reference_target_layer_end_idx >= reference_target_spec.num_layers:
        raise ValueError(
            "position_layer_idx with the requested injection_window_size would exceed the reference target stack: "
            f"start={reference_target_layer_idx}, end={reference_target_layer_end_idx}, "
            f"last_layer={reference_target_spec.num_layers - 1}"
        )
    reference_translated_num_layers = requested_window_size

    relative_depth = relative_depth_from_layer_index(reference_target_layer_idx, reference_target_spec.num_layers)

    edge_map = build_edge_map(edges)
    mappings: Dict[str, LayerMapping] = {}
    for direction in active_directions:
        edge = edge_map[direction]
        src_spec = model_specs[edge.src_id]
        dst_spec = model_specs[edge.dst_id]
        src_layer_idx = relative_depth_to_layer_index(relative_depth, src_spec.num_layers)
        dst_layer_idx = relative_depth_to_layer_index(relative_depth, dst_spec.num_layers)
        available_upper_layers = min(
            src_spec.num_layers - 1 - src_layer_idx,
            dst_spec.num_layers - 1 - dst_layer_idx,
        )
        translated_num_layers = min(requested_window_size, 1 + available_upper_layers)
        mappings[direction] = LayerMapping(
            reference_direction=reference_edge.id,
            reference_target_node_id=reference_edge.dst_id,
            reference_target_num_layers=reference_target_spec.num_layers,
            reference_target_layer_idx=reference_target_layer_idx,
            reference_target_layer_end_idx=reference_target_layer_end_idx,
            relative_depth=relative_depth,
            src_layer_idx=src_layer_idx,
            src_layer_end_idx=src_layer_idx + translated_num_layers - 1,
            dst_layer_idx=dst_layer_idx,
            dst_layer_end_idx=dst_layer_idx + translated_num_layers - 1,
            translated_num_layers=translated_num_layers,
            src_num_layers=src_spec.num_layers,
            dst_num_layers=dst_spec.num_layers,
        )
    return mappings


def extract_layer_window_blocks(
    past_key_values: PastKeyValues,
    start_layer_idx: int,
    num_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    end_layer_idx = start_layer_idx + num_layers
    if not (0 <= start_layer_idx < len(past_key_values)):
        raise ValueError(f"start_layer_idx={start_layer_idx} must be in [0, {len(past_key_values) - 1}]")
    if end_layer_idx > len(past_key_values):
        raise ValueError(
            f"Cannot extract layers [{start_layer_idx}, {end_layer_idx - 1}] from cache with {len(past_key_values)} layers"
        )
    return past_key_values_to_blocks(past_key_values[start_layer_idx:end_layer_idx])


def layer_window_blocks_to_past(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> PastKeyValues:
    return blocks_to_partial_past_key_values(
        key_block=key_block,
        value_block=value_block,
        num_heads=num_heads,
        head_dim=head_dim,
    )


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


def replay_target_prefill_with_injected_window(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_start_layer_idx: int,
    injected_key_block: torch.Tensor,
    injected_value_block: torch.Tensor,
    dst_spec: ModelSpec,
) -> PastKeyValues:
    injected_window = layer_window_blocks_to_past(
        injected_key_block,
        injected_value_block,
        dst_spec.num_heads,
        dst_spec.head_dim,
    )
    translated_num_layers = len(injected_window)

    transformer = require_gpt2_transformer(target_model)
    target_end_layer_idx = target_start_layer_idx + translated_num_layers - 1
    if not (0 <= target_start_layer_idx < len(transformer.h)):
        raise ValueError(f"target_start_layer_idx={target_start_layer_idx} must be in [0, {len(transformer.h) - 1}]")
    if target_end_layer_idx >= len(transformer.h):
        raise ValueError(
            f"Injected window ending at layer {target_end_layer_idx} exceeds target stack with {len(transformer.h)} layers"
        )

    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if torch.is_grad_enabled():
        with torch.no_grad():
            hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
            for lower_idx in range(target_start_layer_idx):
                hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
                rebuilt_past.append((present[0].detach(), present[1].detach()))
        hidden_states = hidden_states.detach()
    else:
        hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
        for lower_idx in range(target_start_layer_idx):
            hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
            rebuilt_past.append(present)

    for offset, injected_present in enumerate(injected_window):
        layer_idx = target_start_layer_idx + offset
        hidden_states, present = run_gpt2_block_with_injected_layer(
            transformer.h[layer_idx],
            hidden_states,
            injected_present[0],
            injected_present[1],
        )
        rebuilt_past.append(present)

    for upper_idx in range(target_end_layer_idx + 1, len(transformer.h)):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[upper_idx], hidden_states)
        rebuilt_past.append(present)

    return tuple(rebuilt_past)


def build_study_dir(config: LayerPositionConfig) -> Path:
    study_id = config.study_id or f"run_{sanitize_slug(config.model_directions)}"
    return Path(config.output_root) / study_id


def build_run_output_dir(config: LayerPositionConfig) -> Path:
    return build_study_dir(config) / resolve_run_position_label(config)


def format_layer_range(start_idx: int, end_idx: int) -> str:
    if start_idx == end_idx:
        return f"L{start_idx}"
    return f"L{start_idx}-L{end_idx}"


def format_window_title(injection_window_size: int) -> str:
    if injection_window_size < 1:
        raise ValueError("injection_window_size must be >= 1")
    return f"win={injection_window_size}"


def build_train_log_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_training.log"


def build_eval_log_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_evaluation.log"


def build_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "final_translator_checkpoint.pt"


def build_config_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_run_config.json"


def build_layer_mapping_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_layer_mapping.json"


def build_metrics_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_evaluation_metrics.json"

def build_chart_path(study_dir: Path, metric_name: str) -> Path:
    return study_dir / f"layer_idx_vs_{sanitize_slug(metric_name)}.png"


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
            "[LayerMapping] %s | ref_target=L%d-%d/%d | %s(%s): layers %d-%d/%d -> %s(%s): layers %d-%d/%d | translated_num_layers=%d | relative_depth=%.4f",
            direction,
            mapping.reference_target_layer_idx,
            mapping.reference_target_layer_end_idx,
            mapping.reference_target_num_layers - 1,
            src_id,
            node_map[src_id].model_id,
            mapping.src_layer_idx,
            mapping.src_layer_end_idx,
            model_specs[src_id].num_layers - 1,
            dst_id,
            node_map[dst_id].model_id,
            mapping.dst_layer_idx,
            mapping.dst_layer_end_idx,
            model_specs[dst_id].num_layers - 1,
            mapping.translated_num_layers,
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
) -> Tuple[LayerWindowTranslatorPool, Dict[str, ModelSpec], Dict[str, LayerMapping]]:
    model_specs = {node_id: get_model_spec(model) for node_id, model in models.items()}
    layer_mappings = build_layer_mappings(config, model_specs, edges, active_directions, reference_edge)
    translator_pool = LayerWindowTranslatorPool(
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
) -> Tuple[LayerWindowTranslatorPool, Dict[str, ModelSpec], Dict[str, LayerMapping]]:
    logger = setup_logger(f"layer_position_train_{run_dir.name}", build_train_log_path(run_dir))
    logger.info("Starting layer-window position training with target-layer replay")
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
                translated_key, translated_value, mapping = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                mixed_target_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=translated_key,
                    injected_value_block=translated_value,
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
                "[Step %04d] window_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
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
def evaluate_logit_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    config: LayerPositionConfig,
    translator_pool: LayerWindowTranslatorPool,
    model_specs: Dict[str, ModelSpec],
    models: Dict[str, PreTrainedModel],
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    edge_map = build_edge_map(edges)
    path_metrics = {direction: AccuracyMeter() for direction in active_directions}
    path_drift = {direction: DriftMeter() for direction in active_directions}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            prefix = prepare_question_prefix(
                tokenizer=tokenizer,
                question=example["question"],
                device=config.device,
                choices=example.get("choices"),
                subject=example.get("subject"),
                context=example.get("context"),
                answer_mode=spec.answer_mode,
            )
            candidate_token_ids = build_logit_answer_candidates(tokenizer=tokenizer, spec=spec, example=example)
            gold_answer = example["answer"]
            prefix_input_ids = prefix["cache_ids"]
            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], prefix_input_ids)
                for node in nodes
            }
            native_capture_by_target: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                translated_capture = replay_target_prefill_with_capture(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=prefix_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=translated_key,
                    injected_value_block=translated_value,
                    dst_spec=model_specs[edge.dst_id],
                )
                mixed_target_past = translated_capture["past_key_values"]

                native_capture_key = (edge.dst_id, mapping.dst_layer_end_idx)
                native_capture = native_capture_by_target.get(native_capture_key)
                if native_capture is None:
                    native_capture = run_native_prefill_capture(
                        target_model=models[edge.dst_id],
                        prefix_input_ids=prefix_input_ids,
                        capture_layer_idx=mapping.dst_layer_end_idx,
                    )
                    native_capture_by_target[native_capture_key] = native_capture

                translated_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=mixed_target_past,
                    seed_token=prefix["seed_token"],
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                native_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=past_by_node_id[edge.dst_id],
                    seed_token=prefix["seed_token"],
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )

                translated_pred = predict_answer_label(translated_scores)
                native_pred = predict_answer_label(native_scores)
                path_metrics[direction].update(
                    accuracy_value=1.0 if translated_pred == gold_answer else 0.0,
                    native_accuracy_value=1.0 if native_pred == gold_answer else 0.0,
                    n=1,
                )
                path_drift[direction].update(
                    injected_cosine=compute_hidden_cosine(
                        translated_capture["after_injected_window_hidden"],
                        native_capture["after_injected_window_hidden"],
                    ),
                    injected_l2=compute_hidden_l2(
                        translated_capture["after_injected_window_hidden"],
                        native_capture["after_injected_window_hidden"],
                    ),
                    final_cosine=compute_hidden_cosine(
                        translated_capture["final_hidden"],
                        native_capture["final_hidden"],
                    ),
                    final_l2=compute_hidden_l2(
                        translated_capture["final_hidden"],
                        native_capture["final_hidden"],
                    ),
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 50 == 0:
            logger.info(
                "[%s] progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                config.eval_max_examples_per_dataset,
            )

    summarized_metrics = {direction: meter.summary() for direction, meter in path_metrics.items()}
    summarized_drift = {direction: meter.summary() for direction, meter in path_drift.items()}
    return (
        {
            direction: {
                "accuracy": row["accuracy"],
                "native_accuracy": row["native_accuracy"],
                "count": row["count"],
            }
            for direction, row in summarized_metrics.items()
        },
        {direction: row for direction, row in summarized_drift.items()},
    )



@torch.inference_mode()
def evaluate_generation_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    config: LayerPositionConfig,
    translator_pool: LayerWindowTranslatorPool,
    model_specs: Dict[str, ModelSpec],
    models: Dict[str, PreTrainedModel],
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    edge_map = build_edge_map(edges)
    path_metrics = {direction: F1Meter() for direction in active_directions}
    path_drift = {direction: DriftMeter() for direction in active_directions}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            context_prefix = prepare_extractive_context_inputs(
                tokenizer=tokenizer,
                context=context_text,
                device=config.device,
            )
            cache_input_ids = context_prefix["input_ids"]

            question_prefix = prepare_extractive_question_prefix(
                tokenizer=tokenizer,
                question=question,
                device=config.device,
            )
            question_cache_ids = question_prefix["cache_ids"]
            seed_token = question_prefix["seed_token"]
            candidate_spans = build_extractive_span_candidates(
                tokenizer=tokenizer,
                context=context_text,
                max_answer_tokens=config.extractive_max_answer_tokens,
            )

            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], cache_input_ids)
                for node in nodes
            }
            native_capture_by_target: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                translated_capture = replay_target_prefill_with_capture(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=cache_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=translated_key,
                    injected_value_block=translated_value,
                    dst_spec=model_specs[edge.dst_id],
                )
                mixed_target_past = translated_capture["past_key_values"]

                native_capture_key = (edge.dst_id, mapping.dst_layer_end_idx)
                native_capture = native_capture_by_target.get(native_capture_key)
                if native_capture is None:
                    native_capture = run_native_prefill_capture(
                        target_model=models[edge.dst_id],
                        prefix_input_ids=cache_input_ids,
                        capture_layer_idx=mapping.dst_layer_end_idx,
                    )
                    native_capture_by_target[native_capture_key] = native_capture

                translated_answer_past = append_input_ids_to_past(
                    model=models[edge.dst_id],
                    past_key_values=mixed_target_past,
                    input_ids=question_cache_ids,
                )
                native_answer_past = append_input_ids_to_past(
                    model=models[edge.dst_id],
                    past_key_values=past_by_node_id[edge.dst_id],
                    input_ids=question_cache_ids,
                )

                translated_answer = predict_extractive_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=translated_answer_past,
                    seed_token=seed_token,
                    candidate_spans=candidate_spans,
                    max_answer_tokens=config.extractive_max_answer_tokens,
                )
                native_answer = predict_extractive_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=native_answer_past,
                    seed_token=seed_token,
                    candidate_spans=candidate_spans,
                    max_answer_tokens=config.extractive_max_answer_tokens,
                )

                f1 = compute_generation_f1(translated_answer, gold_answers)
                native_f1 = compute_generation_f1(native_answer, gold_answers)
                path_metrics[direction].update(
                    f1_value=f1,
                    native_f1_value=native_f1,
                    n=1,
                )
                path_drift[direction].update(
                    injected_cosine=compute_hidden_cosine(
                        translated_capture["after_injected_window_hidden"],
                        native_capture["after_injected_window_hidden"],
                    ),
                    injected_l2=compute_hidden_l2(
                        translated_capture["after_injected_window_hidden"],
                        native_capture["after_injected_window_hidden"],
                    ),
                    final_cosine=compute_hidden_cosine(
                        translated_capture["final_hidden"],
                        native_capture["final_hidden"],
                    ),
                    final_l2=compute_hidden_l2(
                        translated_capture["final_hidden"],
                        native_capture["final_hidden"],
                    ),
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 25 == 0:
            logger.info(
                "[%s] extractive progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                config.eval_max_examples_per_dataset,
            )

    summarized_metrics = {direction: meter.summary() for direction, meter in path_metrics.items()}
    summarized_drift = {direction: meter.summary() for direction, meter in path_drift.items()}
    return (
        {
            direction: {
                "f1": row["f1"],
                "native_f1": row["native_f1"],
                "count": row["count"],
            }
            for direction, row in summarized_metrics.items()
        },
        {direction: row for direction, row in summarized_drift.items()},
    )


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


@dataclass
class SummaryRow:
    study_id: str
    benchmark_mode: str
    metric_name: str
    position_layer_idx: int
    translated_num_layers: int
    source_layer_idx: int
    source_layer_end_idx: int
    target_layer_idx: int
    target_layer_end_idx: int
    average_metric: float
    average_native_metric: float
    injected_cosine: float
    injected_l2: float
    final_cosine: float
    final_l2: float
    run_dir: str


class DriftMeter:
    def __init__(self) -> None:
        self.injected_cosine_sum = 0.0
        self.injected_l2_sum = 0.0
        self.final_cosine_sum = 0.0
        self.final_l2_sum = 0.0
        self.count = 0

    def update(
        self,
        injected_cosine: float,
        injected_l2: float,
        final_cosine: float,
        final_l2: float,
        n: int = 1,
    ) -> None:
        self.injected_cosine_sum += float(injected_cosine) * n
        self.injected_l2_sum += float(injected_l2) * n
        self.final_cosine_sum += float(final_cosine) * n
        self.final_l2_sum += float(final_l2) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "injected_cosine": float("nan"),
                "injected_l2": float("nan"),
                "final_cosine": float("nan"),
                "final_l2": float("nan"),
                "count": 0,
            }
        return {
            "injected_cosine": self.injected_cosine_sum / self.count,
            "injected_l2": self.injected_l2_sum / self.count,
            "final_cosine": self.final_cosine_sum / self.count,
            "final_l2": self.final_l2_sum / self.count,
            "count": self.count,
        }

def build_drift_metrics_path(run_dir: Path) -> Path:
    return run_dir / "drift_metrics.json"


def build_summary_csv_path(study_dir: Path) -> Path:
    return study_dir / "summary.csv"


def build_drift_cosine_chart_path(study_dir: Path) -> Path:
    return study_dir / "drift_cosine.png"


def build_drift_l2_chart_path(study_dir: Path) -> Path:
    return study_dir / "drift_l2.png"


def resolve_dataset_specs(benchmark_mode: str) -> List[HFDatasetSpec]:
    if benchmark_mode == "qa_accuracy":
        return LOGIT_QA_DATASET_SPECS
    if benchmark_mode == "squad_f1":
        return GENERATION_DATASET_SPECS
    raise ValueError(f"Unsupported benchmark_mode: {benchmark_mode}")


@torch.inference_mode()
def run_native_prefill_capture(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    capture_layer_idx: int,
) -> Dict[str, torch.Tensor]:
    transformer = require_gpt2_transformer(target_model)
    if not (0 <= capture_layer_idx < len(transformer.h)):
        raise ValueError(f"capture_layer_idx={capture_layer_idx} must be in [0, {len(transformer.h) - 1}]")

    hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
    after_capture_hidden = None
    for layer_idx, block in enumerate(transformer.h):
        hidden_states, _ = run_gpt2_block_with_cache(block, hidden_states)
        if layer_idx == capture_layer_idx:
            after_capture_hidden = hidden_states.detach().clone()

    if after_capture_hidden is None:
        raise RuntimeError("Failed to capture native hidden states at the requested layer")

    final_hidden = hidden_states
    if getattr(transformer, "ln_f", None) is not None:
        final_hidden = transformer.ln_f(final_hidden)

    return {
        "after_injected_window_hidden": after_capture_hidden,
        "final_hidden": final_hidden.detach().clone(),
    }


@torch.inference_mode()
def replay_target_prefill_with_capture(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_start_layer_idx: int,
    injected_key_block: torch.Tensor,
    injected_value_block: torch.Tensor,
    dst_spec: ModelSpec,
) -> Dict[str, Any]:
    injected_window = layer_window_blocks_to_past(
        injected_key_block,
        injected_value_block,
        dst_spec.num_heads,
        dst_spec.head_dim,
    )
    translated_num_layers = len(injected_window)

    transformer = require_gpt2_transformer(target_model)
    target_end_layer_idx = target_start_layer_idx + translated_num_layers - 1
    if not (0 <= target_start_layer_idx < len(transformer.h)):
        raise ValueError(f"target_start_layer_idx={target_start_layer_idx} must be in [0, {len(transformer.h) - 1}]")
    if target_end_layer_idx >= len(transformer.h):
        raise ValueError(
            f"Injected window ending at layer {target_end_layer_idx} exceeds target stack with {len(transformer.h)} layers"
        )

    hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for lower_idx in range(target_start_layer_idx):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
        rebuilt_past.append((present[0].detach(), present[1].detach()))

    after_injected_window_hidden = None
    for offset, injected_present in enumerate(injected_window):
        layer_idx = target_start_layer_idx + offset
        hidden_states, present = run_gpt2_block_with_injected_layer(
            transformer.h[layer_idx],
            hidden_states,
            injected_present[0],
            injected_present[1],
        )
        rebuilt_past.append((present[0].detach(), present[1].detach()))
        if layer_idx == target_end_layer_idx:
            after_injected_window_hidden = hidden_states.detach().clone()

    if after_injected_window_hidden is None:
        raise RuntimeError("Failed to capture hidden states after the injected layer window")

    for upper_idx in range(target_end_layer_idx + 1, len(transformer.h)):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[upper_idx], hidden_states)
        rebuilt_past.append((present[0].detach(), present[1].detach()))

    final_hidden = hidden_states
    if getattr(transformer, "ln_f", None) is not None:
        final_hidden = transformer.ln_f(final_hidden)

    return {
        "past_key_values": tuple(rebuilt_past),
        "after_injected_window_hidden": after_injected_window_hidden,
        "final_hidden": final_hidden.detach().clone(),
    }


def compute_hidden_cosine(hidden_a: torch.Tensor, hidden_b: torch.Tensor) -> float:
    if hidden_a.shape != hidden_b.shape:
        raise ValueError(f"Hidden shapes must match, got {tuple(hidden_a.shape)} vs {tuple(hidden_b.shape)}")
    hidden_a = hidden_a.float()
    hidden_b = hidden_b.float()
    flat_a = hidden_a.reshape(-1, hidden_a.shape[-1])
    flat_b = hidden_b.reshape(-1, hidden_b.shape[-1])
    cosine = F.cosine_similarity(flat_a, flat_b, dim=-1)
    return float(cosine.mean().item())


def compute_hidden_l2(hidden_a: torch.Tensor, hidden_b: torch.Tensor) -> float:
    if hidden_a.shape != hidden_b.shape:
        raise ValueError(f"Hidden shapes must match, got {tuple(hidden_a.shape)} vs {tuple(hidden_b.shape)}")
    hidden_a = hidden_a.float()
    hidden_b = hidden_b.float()
    l2 = (hidden_a - hidden_b).pow(2).sum(dim=-1).sqrt()
    return float(l2.mean().item())




def extract_eval_metrics(combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
    metric_name = str(combined_metrics["metric_name"])
    dataset_results_key = "dataset_accuracies" if metric_name == "accuracy" else "dataset_f1"
    return {
        "benchmark_mode": combined_metrics["benchmark_mode"],
        "metric_name": metric_name,
        "layer_mappings": combined_metrics["layer_mappings"],
        dataset_results_key: combined_metrics[dataset_results_key],
        "average_metric": combined_metrics["average_metric"],
        "average_native_metric": combined_metrics["average_native_metric"],
        f"average_{metric_name}": combined_metrics[f"average_{metric_name}"],
        f"average_native_{metric_name}": combined_metrics[f"average_native_{metric_name}"],
    }


def extract_drift_metrics(combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark_mode": combined_metrics["benchmark_mode"],
        "layer_mappings": combined_metrics["layer_mappings"],
        "dataset_drift": combined_metrics["dataset_drift"],
        "average_injected_cosine": combined_metrics["average_injected_cosine"],
        "average_injected_l2": combined_metrics["average_injected_l2"],
        "average_final_cosine": combined_metrics["average_final_cosine"],
        "average_final_l2": combined_metrics["average_final_l2"],
    }


def build_summary_row(
    config: LayerPositionConfig,
    run_dir: Path,
    metrics: Dict[str, Any],
) -> SummaryRow:
    reference_mapping = next(iter(metrics["layer_mappings"].values()))
    return SummaryRow(
        study_id=config.study_id or "",
        benchmark_mode=str(metrics["benchmark_mode"]),
        metric_name=str(metrics["metric_name"]),
        position_layer_idx=int(reference_mapping["reference_target_layer_idx"]),
        translated_num_layers=int(reference_mapping["translated_num_layers"]),
        source_layer_idx=int(reference_mapping["src_layer_idx"]),
        source_layer_end_idx=int(reference_mapping["src_layer_end_idx"]),
        target_layer_idx=int(reference_mapping["dst_layer_idx"]),
        target_layer_end_idx=int(reference_mapping["dst_layer_end_idx"]),
        average_metric=float(metrics["average_metric"]),
        average_native_metric=float(metrics["average_native_metric"]),
        injected_cosine=float(metrics["average_injected_cosine"]),
        injected_l2=float(metrics["average_injected_l2"]),
        final_cosine=float(metrics["average_final_cosine"]),
        final_l2=float(metrics["average_final_l2"]),
        run_dir=str(run_dir),
    )


def read_summary_rows(summary_path: Path) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    if not summary_path.exists():
        return rows
    with summary_path.open("r", encoding="utf-8", newline="") as fp:
        for existing in csv.DictReader(fp):
            try:
                rows.append(
                    SummaryRow(
                        study_id=existing.get("study_id", ""),
                        benchmark_mode=existing["benchmark_mode"],
                        metric_name=existing["metric_name"],
                        position_layer_idx=int(existing["position_layer_idx"]),
                        translated_num_layers=int(existing["translated_num_layers"]),
                        source_layer_idx=int(existing["source_layer_idx"]),
                        source_layer_end_idx=int(existing["source_layer_end_idx"]),
                        target_layer_idx=int(existing["target_layer_idx"]),
                        target_layer_end_idx=int(existing["target_layer_end_idx"]),
                        average_metric=float(existing["average_metric"]),
                        average_native_metric=float(existing["average_native_metric"]),
                        injected_cosine=float(existing["injected_cosine"]),
                        injected_l2=float(existing["injected_l2"]),
                        final_cosine=float(existing["final_cosine"]),
                        final_l2=float(existing["final_l2"]),
                        run_dir=existing["run_dir"],
                    )
                )
            except (KeyError, ValueError):
                continue
    return rows


def write_summary(study_dir: Path, rows: List[SummaryRow]) -> Path:
    summary_path = build_summary_csv_path(study_dir)
    rows = sorted(rows, key=lambda row: row.position_layer_idx)
    fieldnames = list(SummaryRow.__dataclass_fields__.keys())
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})
    return summary_path


def update_summary(
    config: LayerPositionConfig,
    run_dir: Path,
    metrics: Dict[str, Any],
) -> Path:
    study_dir = run_dir.parent
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_path = build_summary_csv_path(study_dir)
    row = build_summary_row(config, run_dir, metrics)

    rows = [existing for existing in read_summary_rows(summary_path) if existing.position_layer_idx != row.position_layer_idx]
    rows.append(row)
    rows.sort(key=lambda item: item.position_layer_idx)
    return write_summary(study_dir, rows)


def annotate_injected_layer_ranges(ax, rows: List[Any], y_getter) -> None:
    for row in rows:
        x_value = float(row.position_layer_idx)
        ax.annotate(
            format_layer_range(int(row.target_layer_idx), int(row.target_layer_end_idx)),
            (x_value, float(y_getter(row))),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )


def plot_drift_summary(summary_path: Path) -> Tuple[Path, Path]:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda row: row.position_layer_idx)
    x_values = [row.position_layer_idx for row in rows]
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    cosine_fig = plt.figure(figsize=(9, 5.2))
    cosine_ax = cosine_fig.add_subplot(111)
    cosine_ax.plot(x_values, [row.injected_cosine for row in rows], marker="o", label="Injected window end")
    cosine_ax.plot(x_values, [row.final_cosine for row in rows], marker="s", label="Final layer")
    annotate_injected_layer_ranges(cosine_ax, rows, lambda row: row.injected_cosine)
    cosine_ax.set_xlabel("Reference target layer index")
    cosine_ax.set_ylabel("Cosine similarity")
    cosine_ax.set_title(f"Drift cosine vs layer index ({window_title})")
    cosine_ax.set_xticks(x_values)
    cosine_ax.grid(True, alpha=0.3)
    cosine_ax.legend()
    cosine_fig.tight_layout()
    cosine_path = build_drift_cosine_chart_path(study_dir)
    cosine_fig.savefig(cosine_path, dpi=200)
    plt.close(cosine_fig)

    l2_fig = plt.figure(figsize=(9, 5.2))
    l2_ax = l2_fig.add_subplot(111)
    l2_ax.plot(x_values, [row.injected_l2 for row in rows], marker="o", label="Injected window end")
    l2_ax.plot(x_values, [row.final_l2 for row in rows], marker="s", label="Final layer")
    annotate_injected_layer_ranges(l2_ax, rows, lambda row: row.injected_l2)
    l2_ax.set_xlabel("Reference target layer index")
    l2_ax.set_ylabel("L2 distance")
    l2_ax.set_title(f"Drift L2 vs layer index ({window_title})")
    l2_ax.set_xticks(x_values)
    l2_ax.grid(True, alpha=0.3)
    l2_ax.legend()
    l2_fig.tight_layout()
    l2_path = build_drift_l2_chart_path(study_dir)
    l2_fig.savefig(l2_path, dpi=200)
    plt.close(l2_fig)

    return cosine_path, l2_path


def remove_stale_summary_artifacts(study_dir: Path, run_dir: Path) -> None:
    stale_paths = [
        run_dir / "eval_summary.md",
        study_dir / "drift_summary.md",
        study_dir / "study_summary.csv",
        study_dir / "drift_summary.csv",
    ]
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()


def save_drift_artifacts(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    write_json(str(build_drift_metrics_path(run_dir)), metrics)
    return build_drift_metrics_path(run_dir)


def run_eval(
    config: LayerPositionConfig,
    run_dir: Path,
    translator_pool: LayerWindowTranslatorPool,
    model_specs: Dict[str, ModelSpec],
    layer_mappings: Dict[str, LayerMapping],
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
) -> Dict[str, Any]:
    logger = setup_logger(f"layer_position_eval_{run_dir.name}", build_eval_log_path(run_dir))
    logger.info("Starting layer-window position evaluation with target-layer replay")
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

    dataset_results_by_name: Dict[str, Dict[str, Dict[str, float]]] = {}
    dataset_drift_by_name: Dict[str, Dict[str, Dict[str, float]]] = {}

    if config.benchmark_mode == "qa_accuracy":
        metric_name = "accuracy"
        dataset_results_key = "dataset_accuracies"
        dataset_specs = LOGIT_QA_DATASET_SPECS
        dataset_evaluator = evaluate_logit_dataset
        progress_log_template = (
            "[%s] %s | accuracy=%.6f | native_accuracy=%.6f | injected_cosine=%.6f | "
            "injected_l2=%.6f | final_cosine=%.6f | final_l2=%.6f | count=%d"
        )
    elif config.benchmark_mode == "squad_f1":
        metric_name = "f1"
        dataset_results_key = "dataset_f1"
        dataset_specs = GENERATION_DATASET_SPECS
        dataset_evaluator = evaluate_generation_dataset
        progress_log_template = (
            "[%s] %s | f1=%.6f | native_f1=%.6f | injected_cosine=%.6f | "
            "injected_l2=%.6f | final_cosine=%.6f | final_l2=%.6f | count=%d"
        )
    else:
        raise ValueError(f"Unsupported benchmark_mode: {config.benchmark_mode}")

    for spec in dataset_specs:
        dataloader = build_eval_dataloader(spec=spec, eval_config=eval_config)
        dataset_results, dataset_drift = dataset_evaluator(
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
        dataset_results_by_name[spec.name_for_log] = dataset_results
        dataset_drift_by_name[spec.name_for_log] = dataset_drift
        for direction in active_directions:
            metric_row = dataset_results[direction]
            drift_row = dataset_drift[direction]
            logger.info(
                progress_log_template,
                spec.name_for_log,
                direction,
                metric_row[metric_name],
                metric_row[f"native_{metric_name}"],
                drift_row["injected_cosine"],
                drift_row["injected_l2"],
                drift_row["final_cosine"],
                drift_row["final_l2"],
                int(metric_row["count"]),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    average_metric = compute_average_metric(dataset_results_by_name, metric_name)
    average_native_metric = compute_average_metric(dataset_results_by_name, f"native_{metric_name}")
    average_injected_cosine = compute_average_metric(dataset_drift_by_name, "injected_cosine")
    average_injected_l2 = compute_average_metric(dataset_drift_by_name, "injected_l2")
    average_final_cosine = compute_average_metric(dataset_drift_by_name, "final_cosine")
    average_final_l2 = compute_average_metric(dataset_drift_by_name, "final_l2")

    logger.info(
        "[Summary] metric=%s | average_metric=%.6f | average_native_metric=%.6f",
        metric_name,
        average_metric,
        average_native_metric,
    )
    logger.info(
        "[Summary] avg_injected_cosine=%.6f | avg_injected_l2=%.6f | avg_final_cosine=%.6f | avg_final_l2=%.6f",
        average_injected_cosine,
        average_injected_l2,
        average_final_cosine,
        average_final_l2,
    )
    return {
        "benchmark_mode": config.benchmark_mode,
        "metric_name": metric_name,
        "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        dataset_results_key: dataset_results_by_name,
        "average_metric": average_metric,
        "average_native_metric": average_native_metric,
        f"average_{metric_name}": average_metric,
        f"average_native_{metric_name}": average_native_metric,
        "dataset_drift": dataset_drift_by_name,
        "average_injected_cosine": average_injected_cosine,
        "average_injected_l2": average_injected_l2,
        "average_final_cosine": average_final_cosine,
        "average_final_l2": average_final_l2,
    }




def plot_study_summary(summary_path: Path) -> Path:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda item: int(item.position_layer_idx))
    x_values = [int(item.position_layer_idx) for item in rows]
    y_values = [float(item.average_metric) for item in rows]
    native_values = [float(item.average_native_metric) for item in rows]
    average_native_metric = sum(native_values) / len(native_values) if native_values else float("nan")
    metric_name = rows[0].metric_name or "metric"
    metric_label = metric_name.upper() if metric_name == "f1" else metric_name.capitalize()
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, marker="o", label=f"Translated {metric_label}")
    if average_native_metric == average_native_metric:
        ax.axhline(
            average_native_metric,
            linestyle="--",
            linewidth=1.5,
            label=f"Native {metric_label} Mean ({average_native_metric:.4f})",
        )
    for row, x_value, y_value in zip(rows, x_values, y_values):
        ax.annotate(
            format_layer_range(int(row.target_layer_idx), int(row.target_layer_end_idx)),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Reference target layer index")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs layer index ({window_title})")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    chart_path = build_chart_path(study_dir, metric_name)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def save_run_artifacts(
    config: LayerPositionConfig,
    run_dir: Path,
    layer_mappings: Dict[str, LayerMapping],
    eval_metrics: Dict[str, Any],
    combined_metrics: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path, Path]:
    study_dir = run_dir.parent
    study_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_summary_artifacts(study_dir, run_dir)
    write_json(str(build_config_path(run_dir)), asdict(config))
    write_json(str(build_layer_mapping_path(run_dir)), {direction: asdict(mapping) for direction, mapping in layer_mappings.items()})
    write_json(str(build_metrics_path(run_dir)), eval_metrics)
    summary_path = update_summary(config, run_dir, combined_metrics)
    chart_path = plot_study_summary(summary_path)
    drift_cosine_chart_path, drift_l2_chart_path = plot_drift_summary(summary_path)
    return summary_path, build_metrics_path(run_dir), chart_path, drift_cosine_chart_path, drift_l2_chart_path



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train and evaluate a translated layer window anchored at a chosen reference-target layer position by replaying target prefill below the window, injecting the translated window, and continuing above it. Hidden-state drift evaluation is run automatically as part of the same execution."
    )
    parser.add_argument("--model-ids", default="gpt2,gpt2")
    parser.add_argument("--model-directions", default="A_to_B")
    parser.add_argument("--reference-direction", default=None)
    parser.add_argument("--position-layer-idx", type=int, default=None, help="Reference target layer index to use as the anchor layer for translation/injection sweeps.")
    parser.add_argument("--injection-window-size", type=int, default=1, help="Total number of consecutive layers to translate and inject, starting from the anchor layer selected by --position-layer-idx. For example, 1 injects only the anchor layer, and 3 injects the anchor layer plus the next two upper layers.")
    parser.add_argument("--print-target-num-layers", action="store_true")

    parser.add_argument("--output-root", default="outputs/layer_position")
    parser.add_argument("--study-id", default=None)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--total-tokens", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=50_000)

    parser.add_argument("--translator-dim", type=int, default=256)
    parser.add_argument("--translator-heads", type=int, default=4)
    parser.add_argument("--translator-depth", type=int, default=2)
    parser.add_argument("--translator-mlp-ratio", type=int, default=1)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-max-examples-per-dataset", type=int, default=200)
    parser.add_argument("--eval-shuffle-stream", action="store_true")
    parser.add_argument("--benchmark-mode", choices=["qa_accuracy", "squad_f1"], default="qa_accuracy")
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--extractive-max-answer-tokens", type=int, default=16)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if args.print_target_num_layers:
        print(resolve_target_num_layers(args.model_ids, args.model_directions, args.reference_direction))
        return

    config = LayerPositionConfig(
        model_ids=args.model_ids,
        model_directions=args.model_directions,
        reference_direction=args.reference_direction,
        position_layer_idx=args.position_layer_idx,
        injection_window_size=args.injection_window_size,
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
        benchmark_mode=args.benchmark_mode,
        generation_max_new_tokens=args.generation_max_new_tokens,
        extractive_max_answer_tokens=args.extractive_max_answer_tokens,
    )

    if config.position_layer_idx is None:
        raise SystemExit("--position-layer-idx is required unless --print-target-num-layers is used.")

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
    combined_metrics = run_eval(
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
    eval_metrics = extract_eval_metrics(combined_metrics)
    drift_metrics = extract_drift_metrics(combined_metrics)

    summary_path, metrics_path, chart_path, drift_cosine_chart_path, drift_l2_chart_path = save_run_artifacts(
        config=config,
        run_dir=run_dir,
        layer_mappings=layer_mappings,
        eval_metrics=eval_metrics,
        combined_metrics=combined_metrics,
    )
    drift_metrics_path = save_drift_artifacts(run_dir=run_dir, metrics=drift_metrics)

    print(f"Run directory: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Study chart: {chart_path}")
    print(f"Drift metrics: {drift_metrics_path}")
    print(f"Drift cosine chart: {drift_cosine_chart_path}")
    print(f"Drift L2 chart: {drift_l2_chart_path}")


if __name__ == "__main__":
    main()
