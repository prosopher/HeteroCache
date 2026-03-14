#!/usr/bin/env python3
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import *
from eval_util import *
from layer_position import (
    GENERATION_DATASET_SPECS,
    LOGIT_QA_DATASET_SPECS,
    LayerPositionConfig,
    build_models_for_experiment,
    build_translator_pool,
    build_run_output_dir,
    build_summary_markdown_path,
    relative_depth_to_layer_index,
    resolve_reference_direction_metadata,
    replay_target_prefill_with_single_layer,
    sanitize_slug,
    setup_logger,
)


@dataclass
class DriftSummaryRow:
    study_id: str
    benchmark_mode: str
    position_ratio: float
    source_layer_idx: int
    target_layer_idx: int
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


class SimpleNamespaceConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def build_drift_log_path(run_dir: Path) -> Path:
    return run_dir / "drift_eval.log"


def build_drift_metrics_path(run_dir: Path) -> Path:
    return run_dir / "drift_metrics.json"


def build_drift_summary_csv_path(study_dir: Path) -> Path:
    return study_dir / "drift_summary.csv"


def build_drift_summary_md_path(study_dir: Path) -> Path:
    return study_dir / "drift_summary.md"


def build_drift_cosine_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_position_drift_cosine.png"


def build_drift_l2_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_position_drift_l2.png"


def read_study_summary_rows(study_dir: Path) -> List[Dict[str, str]]:
    summary_path = study_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Study summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"Study summary is empty: {summary_path}")
    rows.sort(key=lambda row: float(row["position_ratio"]))
    return rows


def infer_benchmark_mode_from_summary(rows: List[Dict[str, str]]) -> str:
    benchmark_mode = rows[0].get("benchmark_mode", "")
    if benchmark_mode not in {"qa_accuracy", "squad_f1"}:
        raise ValueError(
            "Could not infer benchmark_mode from summary.csv. "
            f"Found: {benchmark_mode!r}"
        )
    return benchmark_mode


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
    target_layer_idx: int,
) -> Dict[str, torch.Tensor]:
    transformer = require_gpt2_transformer(target_model)
    if not (0 <= target_layer_idx < len(transformer.h)):
        raise ValueError(f"target_layer_idx={target_layer_idx} must be in [0, {len(transformer.h) - 1}]")

    hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
    after_target_hidden = None
    for layer_idx, block in enumerate(transformer.h):
        hidden_states, _ = run_gpt2_block_with_cache(block, hidden_states)
        if layer_idx == target_layer_idx:
            after_target_hidden = hidden_states.detach().clone()

    if after_target_hidden is None:
        raise RuntimeError("Failed to capture native hidden states at target layer")

    final_hidden = hidden_states
    if getattr(transformer, "ln_f", None) is not None:
        final_hidden = transformer.ln_f(final_hidden)

    return {
        "after_target_hidden": after_target_hidden,
        "final_hidden": final_hidden.detach().clone(),
    }


@torch.inference_mode()
def replay_target_prefill_with_capture(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_layer_idx: int,
    injected_key_layer: torch.Tensor,
    injected_value_layer: torch.Tensor,
    dst_spec: ModelSpec,
) -> Dict[str, Any]:
    translated_key, translated_value = single_layer_blocks_to_past(
        injected_key_layer,
        injected_value_layer,
        dst_spec.num_heads,
        dst_spec.head_dim,
    )[0]

    transformer = require_gpt2_transformer(target_model)
    if not (0 <= target_layer_idx < len(transformer.h)):
        raise ValueError(f"target_layer_idx={target_layer_idx} must be in [0, {len(transformer.h) - 1}]")

    hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for lower_idx in range(target_layer_idx):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
        rebuilt_past.append((present[0].detach(), present[1].detach()))

    hidden_states, present = run_gpt2_block_with_injected_layer(
        transformer.h[target_layer_idx],
        hidden_states,
        translated_key,
        translated_value,
    )
    rebuilt_past.append((present[0].detach(), present[1].detach()))
    after_target_hidden = hidden_states.detach().clone()

    for upper_idx in range(target_layer_idx + 1, len(transformer.h)):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[upper_idx], hidden_states)
        rebuilt_past.append((present[0].detach(), present[1].detach()))

    final_hidden = hidden_states
    if getattr(transformer, "ln_f", None) is not None:
        final_hidden = transformer.ln_f(final_hidden)

    return {
        "past_key_values": tuple(rebuilt_past),
        "after_target_hidden": after_target_hidden,
        "final_hidden": final_hidden.detach().clone(),
    }


# Imported late to avoid circular confusion in static linters.
from layer_position import (
    build_gpt2_input_hidden_states,
    require_gpt2_transformer,
    run_gpt2_block_with_cache,
    run_gpt2_block_with_injected_layer,
    single_layer_blocks_to_past,
)


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


@torch.inference_mode()
def evaluate_drift_on_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    benchmark_mode: str,
    tokenizer: PreTrainedTokenizerBase,
    config: LayerPositionConfig,
    translator_pool,
    model_specs: Dict[str, ModelSpec],
    models: Dict[str, PreTrainedModel],
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    edge_map = build_edge_map(edges)
    path_metrics = {direction: DriftMeter() for direction in active_directions}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            if benchmark_mode == "qa_accuracy":
                prefix = prepare_question_prefix(
                    tokenizer=tokenizer,
                    question=example["question"],
                    device=config.device,
                    choices=example.get("choices"),
                    subject=example.get("subject"),
                    context=example.get("context"),
                    answer_mode=spec.answer_mode,
                )
                prefix_input_ids = prefix["cache_ids"]
            elif benchmark_mode == "squad_f1":
                prefix = prepare_generation_context_inputs(
                    tokenizer=tokenizer,
                    context=example["context"],
                    device=config.device,
                )
                prefix_input_ids = prefix["input_ids"]
            else:
                raise ValueError(f"Unsupported benchmark_mode: {benchmark_mode}")

            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], prefix_input_ids)
                for node in nodes
            }

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_single_layer(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )

                translated_capture = replay_target_prefill_with_capture(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=prefix_input_ids,
                    target_layer_idx=mapping.dst_layer_idx,
                    injected_key_layer=translated_key,
                    injected_value_layer=translated_value,
                    dst_spec=model_specs[edge.dst_id],
                )
                native_capture = run_native_prefill_capture(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=prefix_input_ids,
                    target_layer_idx=mapping.dst_layer_idx,
                )

                path_metrics[direction].update(
                    injected_cosine=compute_hidden_cosine(
                        translated_capture["after_target_hidden"],
                        native_capture["after_target_hidden"],
                    ),
                    injected_l2=compute_hidden_l2(
                        translated_capture["after_target_hidden"],
                        native_capture["after_target_hidden"],
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
            logger.info("[%s] drift progress: %d/%d examples", spec.name_for_log, processed_examples, config.eval_max_examples_per_dataset)

    return {direction: meter.summary() for direction, meter in path_metrics.items()}


def average_metric_across_datasets(
    dataset_results_by_name: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
) -> float:
    values: List[float] = []
    for dataset_results in dataset_results_by_name.values():
        for direction_results in dataset_results.values():
            value = direction_results.get(metric_key)
            if value is not None and value == value:
                values.append(float(value))
    if not values:
        return float("nan")
    return sum(values) / len(values)


@torch.inference_mode()
def evaluate_run_drift(
    benchmark_mode: str,
    config: LayerPositionConfig,
    run_dir: Path,
    translator_pool,
    model_specs: Dict[str, ModelSpec],
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
) -> Dict[str, Any]:
    logger = setup_logger(f"layer_drift_{run_dir.name}", build_drift_log_path(run_dir))
    logger.info("Starting drift evaluation")
    logger.info("benchmark_mode=%s", benchmark_mode)
    logger.info("run_dir=%s", run_dir)

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
    for spec in resolve_dataset_specs(benchmark_mode):
        dataloader = build_eval_dataloader(spec=spec, eval_config=eval_config)
        dataset_results = evaluate_drift_on_dataset(
            spec=spec,
            dataloader=dataloader,
            benchmark_mode=benchmark_mode,
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
        for direction in active_directions:
            row = dataset_results[direction]
            logger.info(
                "[%s] %s | injected_cosine=%.6f | injected_l2=%.6f | final_cosine=%.6f | final_l2=%.6f | count=%d",
                spec.name_for_log,
                direction,
                row["injected_cosine"],
                row["injected_l2"],
                row["final_cosine"],
                row["final_l2"],
                int(row["count"]),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {
        "benchmark_mode": benchmark_mode,
        "dataset_drift": dataset_results_by_name,
        "average_injected_cosine": average_metric_across_datasets(dataset_results_by_name, "injected_cosine"),
        "average_injected_l2": average_metric_across_datasets(dataset_results_by_name, "injected_l2"),
        "average_final_cosine": average_metric_across_datasets(dataset_results_by_name, "final_cosine"),
        "average_final_l2": average_metric_across_datasets(dataset_results_by_name, "final_l2"),
    }
    write_json(str(build_drift_metrics_path(run_dir)), metrics)
    logger.info(
        "[Summary] avg_injected_cosine=%.6f | avg_injected_l2=%.6f | avg_final_cosine=%.6f | avg_final_l2=%.6f",
        metrics["average_injected_cosine"],
        metrics["average_injected_l2"],
        metrics["average_final_cosine"],
        metrics["average_final_l2"],
    )
    return metrics


def build_translator_pool_from_run(
    models: Dict[str, PreTrainedModel],
    run_config: LayerPositionConfig,
    edges: List[Edge],
    active_directions: List[str],
    reference_edge: Edge,
    checkpoint_path: Path,
):
    translator_pool, model_specs, layer_mappings = build_translator_pool(
        models=models,
        config=run_config,
        active_directions=active_directions,
        edges=edges,
        reference_edge=reference_edge,
    )
    checkpoint = torch.load(checkpoint_path, map_location=run_config.device)
    translator_pool.load_state_dict(checkpoint["translator_pool"])
    translator_pool.to(run_config.device)
    translator_pool.eval()
    return translator_pool, model_specs, layer_mappings


def build_run_config_from_json(payload: Dict[str, Any]) -> LayerPositionConfig:
    valid_field_names = {field.name for field in fields(LayerPositionConfig)}
    filtered_payload = {key: value for key, value in payload.items() if key in valid_field_names}
    return LayerPositionConfig(**filtered_payload)


def build_drift_summary_row(
    study_id: str,
    benchmark_mode: str,
    run_dir: Path,
    mapping: Dict[str, Any],
    metrics: Dict[str, Any],
) -> DriftSummaryRow:
    return DriftSummaryRow(
        study_id=study_id,
        benchmark_mode=benchmark_mode,
        position_ratio=float(mapping["relative_depth"]),
        source_layer_idx=int(mapping["src_layer_idx"]),
        target_layer_idx=int(mapping["dst_layer_idx"]),
        injected_cosine=float(metrics["average_injected_cosine"]),
        injected_l2=float(metrics["average_injected_l2"]),
        final_cosine=float(metrics["average_final_cosine"]),
        final_l2=float(metrics["average_final_l2"]),
        run_dir=str(run_dir),
    )


def write_drift_summary(study_dir: Path, rows: List[DriftSummaryRow]) -> Path:
    summary_path = build_drift_summary_csv_path(study_dir)
    rows = sorted(rows, key=lambda row: row.position_ratio)
    fieldnames = list(DriftSummaryRow.__dataclass_fields__.keys())
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})
    return summary_path


def build_drift_summary_markdown(rows: List[DriftSummaryRow]) -> str:
    lines = [
        "# Layer Position Drift Summary",
        "",
        "| Ratio | Src Layer | Tgt Layer | Injected Cosine | Injected L2 | Final Cosine | Final L2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item.position_ratio):
        lines.append(
            f"| {row.position_ratio:.1f} | {row.source_layer_idx} | {row.target_layer_idx} | "
            f"{row.injected_cosine:.6f} | {row.injected_l2:.6f} | {row.final_cosine:.6f} | {row.final_l2:.6f} |"
        )
    return "\n".join(lines)


def save_drift_summary_markdown(study_dir: Path, rows: List[DriftSummaryRow]) -> Path:
    markdown_path = build_drift_summary_md_path(study_dir)
    markdown_path.write_text(build_drift_summary_markdown(rows), encoding="utf-8")
    return markdown_path


def plot_drift_summary(study_dir: Path, rows: List[DriftSummaryRow]) -> Tuple[Path, Path]:
    rows = sorted(rows, key=lambda row: row.position_ratio)
    x_values = [row.position_ratio for row in rows]

    import matplotlib.pyplot as plt

    cosine_fig = plt.figure(figsize=(8, 4.5))
    cosine_ax = cosine_fig.add_subplot(111)
    cosine_ax.plot(x_values, [row.injected_cosine for row in rows], marker="o", label="Injected-layer cosine")
    cosine_ax.plot(x_values, [row.final_cosine for row in rows], marker="s", label="Final-layer cosine")
    cosine_ax.set_xlabel("Layer Index Ratio")
    cosine_ax.set_ylabel("Cosine Similarity")
    cosine_ax.set_title("Layer Ratio vs Hidden-State Cosine")
    cosine_ax.set_xticks(x_values)
    cosine_ax.set_xticklabels([f"{value:.1f}" for value in x_values])
    cosine_ax.grid(True, alpha=0.3)
    cosine_ax.legend()
    cosine_fig.tight_layout()
    cosine_path = build_drift_cosine_chart_path(study_dir)
    cosine_fig.savefig(cosine_path, dpi=200)
    plt.close(cosine_fig)

    l2_fig = plt.figure(figsize=(8, 4.5))
    l2_ax = l2_fig.add_subplot(111)
    l2_ax.plot(x_values, [row.injected_l2 for row in rows], marker="o", label="Injected-layer L2")
    l2_ax.plot(x_values, [row.final_l2 for row in rows], marker="s", label="Final-layer L2")
    l2_ax.set_xlabel("Layer Index Ratio")
    l2_ax.set_ylabel("Mean Token L2")
    l2_ax.set_title("Layer Ratio vs Hidden-State L2 Drift")
    l2_ax.set_xticks(x_values)
    l2_ax.set_xticklabels([f"{value:.1f}" for value in x_values])
    l2_ax.grid(True, alpha=0.3)
    l2_ax.legend()
    l2_fig.tight_layout()
    l2_path = build_drift_l2_chart_path(study_dir)
    l2_fig.savefig(l2_path, dpi=200)
    plt.close(l2_fig)

    return cosine_path, l2_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hidden-state drift at the injected layer and final layer for a finished layer_position study."
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--output-root", default="outputs/layer_position")
    parser.add_argument("--benchmark-mode", choices=["qa_accuracy", "squad_f1"], default=None)
    parser.add_argument("--model-ids", default=None)
    parser.add_argument("--model-directions", default=None)
    parser.add_argument("--reference-direction", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--eval-max-examples-per-dataset", type=int, default=256)
    parser.add_argument("--eval-shuffle-stream", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = Path(args.output_root) / args.study_id
    summary_rows = read_study_summary_rows(study_dir)
    benchmark_mode = args.benchmark_mode or infer_benchmark_mode_from_summary(summary_rows)

    first_run_dir = Path(summary_rows[0]["run_dir"])
    first_config_payload = read_json(first_run_dir / "experiment_config.json")
    model_ids = args.model_ids or first_config_payload.get("model_ids", "gpt2,gpt2-medium")
    model_directions = args.model_directions or first_config_payload.get("model_directions", "A_to_B")
    reference_direction = args.reference_direction or first_config_payload.get("reference_direction")
    device = resolve_device(args.device)
    dtype = args.dtype

    base_config = LayerPositionConfig(
        model_ids=model_ids,
        model_directions=model_directions,
        reference_direction=reference_direction,
        position_ratio=0.0,
        output_root=args.output_root,
        study_id=args.study_id,
        device=device,
        dtype=dtype,
        eval_batch_size=args.eval_batch_size or int(first_config_payload.get("eval_batch_size", 4)),
        eval_num_workers=args.eval_num_workers or int(first_config_payload.get("eval_num_workers", 0)),
        eval_max_examples_per_dataset=args.eval_max_examples_per_dataset,
        eval_shuffle_stream=args.eval_shuffle_stream,
        seed=args.seed,
        benchmark_mode=benchmark_mode,
        generation_max_new_tokens=int(first_config_payload.get("generation_max_new_tokens", 32)),
        max_steps=1,
    )

    set_seed(base_config.seed)
    nodes, edges, active_directions, reference_edge = resolve_reference_direction_metadata(
        model_ids=base_config.model_ids,
        model_directions=base_config.model_directions,
        reference_direction=base_config.reference_direction,
    )
    models, tokenizer, _, _ = build_models_for_experiment(base_config)

    drift_rows: List[DriftSummaryRow] = []
    for summary_row in summary_rows:
        run_dir = Path(summary_row["run_dir"])
        run_config_payload = read_json(run_dir / "experiment_config.json")
        run_config = build_run_config_from_json(run_config_payload)
        run_config.device = base_config.device
        run_config.dtype = base_config.dtype
        run_config.eval_batch_size = base_config.eval_batch_size
        run_config.eval_num_workers = base_config.eval_num_workers
        run_config.eval_max_examples_per_dataset = base_config.eval_max_examples_per_dataset
        run_config.eval_shuffle_stream = base_config.eval_shuffle_stream
        run_config.seed = base_config.seed
        run_config.benchmark_mode = benchmark_mode

        checkpoint_path = run_dir / "final_checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        translator_pool, model_specs, layer_mappings = build_translator_pool_from_run(
            models=models,
            run_config=run_config,
            edges=edges,
            active_directions=active_directions,
            reference_edge=reference_edge,
            checkpoint_path=checkpoint_path,
        )
        metrics = evaluate_run_drift(
            benchmark_mode=benchmark_mode,
            config=run_config,
            run_dir=run_dir,
            translator_pool=translator_pool,
            model_specs=model_specs,
            models=models,
            tokenizer=tokenizer,
            nodes=nodes,
            edges=edges,
            active_directions=active_directions,
        )
        reference_mapping = asdict(next(iter(layer_mappings.values())))
        drift_rows.append(
            build_drift_summary_row(
                study_id=args.study_id,
                benchmark_mode=benchmark_mode,
                run_dir=run_dir,
                mapping=reference_mapping,
                metrics=metrics,
            )
        )

    summary_csv_path = write_drift_summary(study_dir, drift_rows)
    summary_md_path = save_drift_summary_markdown(study_dir, drift_rows)
    cosine_chart_path, l2_chart_path = plot_drift_summary(study_dir, drift_rows)

    print(f"Drift summary CSV: {summary_csv_path}")
    print(f"Drift summary Markdown: {summary_md_path}")
    print(f"Cosine chart: {cosine_chart_path}")
    print(f"L2 chart: {l2_chart_path}")


if __name__ == "__main__":
    main()
