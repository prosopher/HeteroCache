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


LOGIT_DATASET_SPECS = [
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
    HFDatasetSpec(
        name_for_log="MMLU/all/validation",
        dataset_path="cais/mmlu",
        dataset_name="all",
        split="validation",
        answer_mode="mmlu",
        question_field="question",
        subject_field="subject",
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
    model_directions: str = "B_to_A"
    reference_direction: Optional[str] = None
    layer_to_translate: Optional[int] = None

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

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.prefix_tokens < 2 or self.prefix_tokens >= self.total_tokens:
            raise ValueError("prefix_tokens must satisfy 2 <= prefix_tokens < total_tokens")
        if self.layer_to_translate is not None and self.layer_to_translate < 0:
            raise ValueError("layer_to_translate must be >= 0")
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
    if config.layer_to_translate is None:
        raise ValueError("layer_to_translate must be set before building layer mappings")
    if not (0 <= config.layer_to_translate < reference_target_spec.num_layers):
        raise ValueError(
            f"layer_to_translate={config.layer_to_translate} must be in [0, {reference_target_spec.num_layers - 1}]"
        )

    relative_depth = relative_depth_from_layer_index(config.layer_to_translate, reference_target_spec.num_layers)
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
            reference_target_layer_idx=config.layer_to_translate,
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


def replace_single_layer(
    base_past_key_values: PastKeyValues,
    layer_idx: int,
    translated_key_layer: torch.Tensor,
    translated_value_layer: torch.Tensor,
    dst_spec: ModelSpec,
) -> PastKeyValues:
    translated_key, translated_value = single_layer_blocks_to_past(
        translated_key_layer,
        translated_value_layer,
        dst_spec.num_heads,
        dst_spec.head_dim,
    )[0]
    base = list(base_past_key_values)
    base_key, base_value = base[layer_idx]
    if base_key.shape != translated_key.shape or base_value.shape != translated_value.shape:
        raise ValueError("Translated KV shape does not match target layer shape")
    base[layer_idx] = (translated_key, translated_value)
    return tuple(base)


def build_run_output_dir(config: LayerPositionConfig) -> Path:
    study_id = config.study_id or f"run_{sanitize_slug(config.model_directions)}"
    return Path(config.output_root) / study_id / f"layer_{config.layer_to_translate:02d}"


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
    return study_dir / "layer_position_accuracy.png"


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
    logger.info("Starting single-layer position training")
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
                mixed_target_past = replace_single_layer(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    layer_idx=mapping.dst_layer_idx,
                    translated_key_layer=translated_key,
                    translated_value_layer=translated_value,
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
def evaluate_logit_dataset(
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
    path_metrics = {direction: AccuracyMeter() for direction in active_directions}
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
            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], prefix["cache_ids"])
                for node in nodes
            }

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_single_layer(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                mixed_target_past = replace_single_layer(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    layer_idx=mapping.dst_layer_idx,
                    translated_key_layer=translated_key,
                    translated_value_layer=translated_value,
                    dst_spec=model_specs[edge.dst_id],
                )

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

            processed_examples += 1

        if batch_idx % 50 == 0:
            logger.info("[%s] progress: %d/%d examples", spec.name_for_log, processed_examples, config.eval_max_examples_per_dataset)

    summarized = {direction: meter.summary() for direction, meter in path_metrics.items()}
    return {
        direction: {
            "accuracy": row["accuracy"],
            "native_accuracy": row["native_accuracy"],
            "count": row["count"],
        }
        for direction, row in summarized.items()
    }


def compute_average_accuracy(logit_results: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    accuracies = []
    for dataset_results in logit_results.values():
        for direction_results in dataset_results.values():
            value = direction_results.get("accuracy")
            if value is not None and value == value:
                accuracies.append(float(value))
    if not accuracies:
        return float("nan")
    return sum(accuracies) / len(accuracies)



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
    logger.info("Starting single-layer position evaluation")
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

    logit_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in LOGIT_DATASET_SPECS:
        dataloader = build_eval_dataloader(spec=spec, eval_config=eval_config)
        dataset_results = evaluate_logit_dataset(
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
        logit_results[spec.name_for_log] = dataset_results
        for direction in active_directions:
            row = dataset_results[direction]
            logger.info(
                "[%s] %s | accuracy=%.6f | native_accuracy=%.6f | count=%d",
                spec.name_for_log,
                direction,
                row["accuracy"],
                row["native_accuracy"],
                int(row["count"]),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    average_accuracy = compute_average_accuracy(logit_results)
    logger.info("[Summary] average_accuracy=%.6f", average_accuracy)
    return {
        "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        "dataset_accuracies": logit_results,
        "average_accuracy": average_accuracy,
    }



def build_summary_markdown(metrics: Dict[str, Any]) -> str:
    lines = [
        "# Layer Position Result",
        "",
        f"- average_accuracy: {metrics['average_accuracy']:.6f}",
        "",
        "## Dataset Accuracy",
        "",
        "| Dataset | Direction | Accuracy | Native Accuracy | Count |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_name, dataset_results in metrics["dataset_accuracies"].items():
        for direction, row in dataset_results.items():
            lines.append(
                f"| {dataset_name} | {direction} | {row['accuracy']:.6f} | {row['native_accuracy']:.6f} | {int(row['count'])} |"
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
        "layer_to_translate": int(config.layer_to_translate),
        "relative_depth": float(reference_mapping["relative_depth"]),
        "average_accuracy": float(metrics["average_accuracy"]),
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

    filtered_rows = [existing for existing in rows if int(existing["layer_to_translate"]) != row["layer_to_translate"]]
    filtered_rows.append({key: str(value) for key, value in row.items()})
    filtered_rows.sort(key=lambda item: int(item["layer_to_translate"]))

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
    with summary_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            try:
                rows.append((int(row["layer_to_translate"]), float(row["average_accuracy"])))
            except (KeyError, ValueError):
                continue

    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda item: item[0])
    x_values = [item[0] for item in rows]
    y_values = [item[1] for item in rows]

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Accuracy")
    ax.set_title("Layer Position vs Accuracy")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
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
        description="Train and evaluate a single translated layer at each target-relative layer position."
    )
    parser.add_argument("--model-ids", default="gpt2,gpt2-medium")
    parser.add_argument("--model-directions", default="B_to_A")
    parser.add_argument("--reference-direction", default=None)
    parser.add_argument("--layer-to-translate", type=int, default=None)
    parser.add_argument("--print-target-num-layers", action="store_true")
    parser.add_argument("--plot-study-summary", action="store_true")

    parser.add_argument("--output-root", default="outputs/layer_position")
    parser.add_argument("--study-id", default=None)

    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--total-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)

    parser.add_argument("--translator-dim", type=int, default=768)
    parser.add_argument("--translator-heads", type=int, default=12)
    parser.add_argument("--translator-depth", type=int, default=2)
    parser.add_argument("--translator-mlp-ratio", type=int, default=2)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-max-examples-per-dataset", type=int, default=256)
    parser.add_argument("--eval-shuffle-stream", action="store_true")
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
        layer_to_translate=args.layer_to_translate,
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
    )

    if config.layer_to_translate is None:
        raise SystemExit("--layer-to-translate is required unless --print-target-num-layers or --plot-study-summary is used.")

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
