#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    OpenWebTextSequenceStream,
    add_dataclass_arguments,
    build_dataclass_kwargs_from_json_and_namespace,
    extract_past_key_values,
    get_model_spec,
    load_frozen_model,
    load_tokenizer,
    past_key_values_to_blocks,
    resolve_device,
    setup_logger,
    write_json,
)


PoolMode = Literal["mean", "last"]
MetricName = Literal["linear_cka"]


@dataclass
class LayerSimConfig:
    model_a_id: str
    model_b_id: str
    shared_tokenizer_model_id: Optional[str]

    output_root: str
    study_id: Optional[str]

    split: str
    num_samples: int
    batch_size: int
    num_workers: int
    prefix_tokens: int
    shuffle_stream: bool
    shuffle_buffer: int
    seed: int

    pool_mode: PoolMode
    metric: MetricName

    device: str
    dtype: str

    figure_dpi: int
    annotate_heatmap: bool

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.prefix_tokens < 2:
            raise ValueError("prefix_tokens must be >= 2")
        if self.pool_mode not in {"mean", "last"}:
            raise ValueError("pool_mode must be one of {'mean', 'last'}")
        if self.metric != "linear_cka":
            raise ValueError("metric currently supports only 'linear_cka'")
        if self.figure_dpi < 50:
            raise ValueError("figure_dpi must be >= 50")
        if self.shared_tokenizer_model_id is None:
            self.shared_tokenizer_model_id = self.model_a_id
        if self.study_id is None:
            self.study_id = f"{self.model_a_id.replace('/', '_')}__vs__{self.model_b_id.replace('/', '_')}"


class LayerFeatureStore:
    def __init__(self, num_layers: int) -> None:
        self.key_features: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.value_features: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.mean_key_block_sum: Optional[torch.Tensor] = None
        self.mean_value_block_sum: Optional[torch.Tensor] = None
        self.num_examples = 0

    def update(self, key_block: torch.Tensor, value_block: torch.Tensor, pool_mode: PoolMode) -> None:
        # key/value block: [batch, seq, layers, hidden]
        key_cpu = key_block.detach().to(device="cpu", dtype=torch.float32)
        value_cpu = value_block.detach().to(device="cpu", dtype=torch.float32)
        batch_size = key_cpu.shape[0]

        if self.mean_key_block_sum is None:
            self.mean_key_block_sum = torch.zeros_like(key_cpu[0], dtype=torch.float64)
            self.mean_value_block_sum = torch.zeros_like(value_cpu[0], dtype=torch.float64)

        self.mean_key_block_sum += key_cpu.to(dtype=torch.float64).sum(dim=0)
        self.mean_value_block_sum += value_cpu.to(dtype=torch.float64).sum(dim=0)
        self.num_examples += int(batch_size)

        if pool_mode == "mean":
            pooled_key = key_cpu.mean(dim=1)
            pooled_value = value_cpu.mean(dim=1)
        elif pool_mode == "last":
            pooled_key = key_cpu[:, -1, :, :]
            pooled_value = value_cpu[:, -1, :, :]
        else:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")

        for layer_idx in range(pooled_key.shape[1]):
            self.key_features[layer_idx].append(pooled_key[:, layer_idx, :].clone())
            self.value_features[layer_idx].append(pooled_value[:, layer_idx, :].clone())

    def finalize(self) -> Dict[str, object]:
        if self.num_examples < 1:
            raise ValueError("No examples were accumulated.")
        return {
            "key_features": [torch.cat(chunks, dim=0) for chunks in self.key_features],
            "value_features": [torch.cat(chunks, dim=0) for chunks in self.value_features],
            "mean_key_block": (self.mean_key_block_sum / float(self.num_examples)).to(dtype=torch.float32),
            "mean_value_block": (self.mean_value_block_sum / float(self.num_examples)).to(dtype=torch.float32),
            "num_examples": self.num_examples,
        }


@torch.no_grad()
def collect_layer_features(config: LayerSimConfig, logger) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    tokenizer = load_tokenizer(config.shared_tokenizer_model_id)
    model_a = load_frozen_model(config.model_a_id, device=config.device, dtype=config.dtype)
    model_b = load_frozen_model(config.model_b_id, device=config.device, dtype=config.dtype)

    spec_a = get_model_spec(model_a)
    spec_b = get_model_spec(model_b)
    logger.info(
        "Loaded models: A=%s (layers=%d, hidden=%d), B=%s (layers=%d, hidden=%d)",
        config.model_a_id,
        spec_a.num_layers,
        spec_a.hidden_size,
        config.model_b_id,
        spec_b.num_layers,
        spec_b.hidden_size,
    )

    dataset = OpenWebTextSequenceStream(
        tokenizer=tokenizer,
        sequence_length=config.prefix_tokens,
        split=config.split,
        shuffle=config.shuffle_stream,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    store_a = LayerFeatureStore(num_layers=spec_a.num_layers)
    store_b = LayerFeatureStore(num_layers=spec_b.num_layers)

    processed = 0
    for batch_idx, input_ids in enumerate(dataloader, start=1):
        input_ids = input_ids[:, : config.prefix_tokens]
        remaining = config.num_samples - processed
        if remaining <= 0:
            break
        if input_ids.shape[0] > remaining:
            input_ids = input_ids[:remaining]
        input_ids = input_ids.to(config.device)

        past_a = extract_past_key_values(model_a, input_ids)
        past_b = extract_past_key_values(model_b, input_ids)
        key_a, value_a = past_key_values_to_blocks(past_a)
        key_b, value_b = past_key_values_to_blocks(past_b)

        store_a.update(key_a, value_a, pool_mode=config.pool_mode)
        store_b.update(key_b, value_b, pool_mode=config.pool_mode)
        processed += int(input_ids.shape[0])

        if batch_idx == 1:
            logger.info(
                "First batch block shapes: A key=%s value=%s | B key=%s value=%s",
                tuple(key_a.shape),
                tuple(value_a.shape),
                tuple(key_b.shape),
                tuple(value_b.shape),
            )
        if batch_idx % 10 == 0 or processed >= config.num_samples:
            logger.info("Collected %d / %d samples", processed, config.num_samples)
        if processed >= config.num_samples:
            break

    if processed < config.num_samples:
        logger.warning(
            "OpenWebText stream ended early. Requested %d samples, collected %d.",
            config.num_samples,
            processed,
        )

    result_a = store_a.finalize()
    result_b = store_b.finalize()
    metadata = {
        "spec_a": asdict(spec_a),
        "spec_b": asdict(spec_b),
        "num_examples": min(result_a["num_examples"], result_b["num_examples"]),
        "tokenizer_model_id": config.shared_tokenizer_model_id,
    }
    return result_a, result_b, metadata


def center_features(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float64) - x.to(dtype=torch.float64).mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"linear_cka requires the same number of samples, got {x.shape[0]} and {y.shape[0]}")
    if x.shape[0] < 2:
        return float("nan")

    x_centered = center_features(x)
    y_centered = center_features(y)

    cross_cov = x_centered.transpose(0, 1) @ y_centered
    x_cov = x_centered.transpose(0, 1) @ x_centered
    y_cov = y_centered.transpose(0, 1) @ y_centered

    numerator = torch.linalg.matrix_norm(cross_cov, ord="fro").pow(2)
    denominator = torch.linalg.matrix_norm(x_cov, ord="fro") * torch.linalg.matrix_norm(y_cov, ord="fro")
    score = numerator / denominator.clamp_min(eps)
    return float(score.item())



def compute_similarity_matrix(
    a_layers: Sequence[torch.Tensor],
    b_layers: Sequence[torch.Tensor],
    metric: MetricName,
) -> torch.Tensor:
    matrix = torch.zeros((len(a_layers), len(b_layers)), dtype=torch.float64)
    for layer_a_idx, layer_a_repr in enumerate(a_layers):
        for layer_b_idx, layer_b_repr in enumerate(b_layers):
            if metric == "linear_cka":
                matrix[layer_a_idx, layer_b_idx] = linear_cka(layer_a_repr, layer_b_repr)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return matrix.to(dtype=torch.float32)



def build_best_alignment_summary(matrix: torch.Tensor) -> Dict[str, object]:
    per_row = []
    for row_idx in range(matrix.shape[0]):
        best_col = int(torch.argmax(matrix[row_idx]).item())
        per_row.append(
            {
                "src_layer_idx": row_idx,
                "best_dst_layer_idx": best_col,
                "score": float(matrix[row_idx, best_col].item()),
            }
        )
    global_flat_idx = int(torch.argmax(matrix).item())
    global_row = global_flat_idx // matrix.shape[1]
    global_col = global_flat_idx % matrix.shape[1]
    return {
        "best_per_src_layer": per_row,
        "global_best_pair": {
            "src_layer_idx": global_row,
            "dst_layer_idx": global_col,
            "score": float(matrix[global_row, global_col].item()),
        },
    }



def save_matrix_csv(path: Path, matrix: torch.Tensor, row_prefix: str, col_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([""] + [f"{col_prefix}_{idx}" for idx in range(matrix.shape[1])])
        for row_idx in range(matrix.shape[0]):
            writer.writerow([f"{row_prefix}_{row_idx}"] + [f"{float(v):.8f}" for v in matrix[row_idx].tolist()])



def plot_heatmap(
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    annotate: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    height = max(5.5, 0.42 * matrix.shape[0] + 2.0)
    width = max(7.0, 0.34 * matrix.shape[1] + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.cpu().numpy(), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([str(idx) for idx in range(matrix.shape[1])], rotation=45, ha="right")
    ax.set_yticklabels([str(idx) for idx in range(matrix.shape[0])])

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Similarity")

    if annotate and matrix.numel() <= 900:
        values = matrix.cpu().numpy()
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                text_color = "white" if value < 0.5 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def build_output_dir(config: LayerSimConfig) -> Path:
    output_dir = Path(config.output_root) / str(config.study_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir



def run_layer_similarity(config: LayerSimConfig) -> Path:
    output_dir = build_output_dir(config)
    logger = setup_logger("layer_sim", output_dir / "layer_sim.log")
    logger.info("Layer similarity config: %s", asdict(config))

    result_a, result_b, metadata = collect_layer_features(config=config, logger=logger)

    key_matrix = compute_similarity_matrix(
        a_layers=result_a["key_features"],
        b_layers=result_b["key_features"],
        metric=config.metric,
    )
    value_matrix = compute_similarity_matrix(
        a_layers=result_a["value_features"],
        b_layers=result_b["value_features"],
        metric=config.metric,
    )
    kv_matrix = 0.5 * (key_matrix + value_matrix)

    save_matrix_csv(output_dir / "key_similarity.csv", key_matrix, row_prefix="a_layer", col_prefix="b_layer")
    save_matrix_csv(output_dir / "value_similarity.csv", value_matrix, row_prefix="a_layer", col_prefix="b_layer")
    save_matrix_csv(output_dir / "kv_similarity.csv", kv_matrix, row_prefix="a_layer", col_prefix="b_layer")

    plot_heatmap(
        matrix=key_matrix,
        title=f"Key cache layer similarity ({config.model_a_id} vs {config.model_b_id})",
        xlabel=f"{config.model_b_id} layer",
        ylabel=f"{config.model_a_id} layer",
        output_path=output_dir / "key_similarity_heatmap.png",
        annotate=config.annotate_heatmap,
        dpi=config.figure_dpi,
    )
    plot_heatmap(
        matrix=value_matrix,
        title=f"Value cache layer similarity ({config.model_a_id} vs {config.model_b_id})",
        xlabel=f"{config.model_b_id} layer",
        ylabel=f"{config.model_a_id} layer",
        output_path=output_dir / "value_similarity_heatmap.png",
        annotate=config.annotate_heatmap,
        dpi=config.figure_dpi,
    )
    plot_heatmap(
        matrix=kv_matrix,
        title=f"Mean K/V layer similarity ({config.model_a_id} vs {config.model_b_id})",
        xlabel=f"{config.model_b_id} layer",
        ylabel=f"{config.model_a_id} layer",
        output_path=output_dir / "kv_similarity_heatmap.png",
        annotate=config.annotate_heatmap,
        dpi=config.figure_dpi,
    )

    torch.save(
        {
            "model_id": config.model_a_id,
            "mean_key_block": result_a["mean_key_block"],
            "mean_value_block": result_a["mean_value_block"],
        },
        output_dir / "model_a_mean_cache.pt",
    )
    torch.save(
        {
            "model_id": config.model_b_id,
            "mean_key_block": result_b["mean_key_block"],
            "mean_value_block": result_b["mean_value_block"],
        },
        output_dir / "model_b_mean_cache.pt",
    )

    summary = {
        "config": asdict(config),
        "metadata": metadata,
        "key_alignment": build_best_alignment_summary(key_matrix),
        "value_alignment": build_best_alignment_summary(value_matrix),
        "kv_alignment": build_best_alignment_summary(kv_matrix),
        "artifacts": {
            "key_similarity_csv": str(output_dir / "key_similarity.csv"),
            "value_similarity_csv": str(output_dir / "value_similarity.csv"),
            "kv_similarity_csv": str(output_dir / "kv_similarity.csv"),
            "key_similarity_heatmap_png": str(output_dir / "key_similarity_heatmap.png"),
            "value_similarity_heatmap_png": str(output_dir / "value_similarity_heatmap.png"),
            "kv_similarity_heatmap_png": str(output_dir / "kv_similarity_heatmap.png"),
            "model_a_mean_cache_pt": str(output_dir / "model_a_mean_cache.pt"),
            "model_b_mean_cache_pt": str(output_dir / "model_b_mean_cache.pt"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    logger.info(
        "Global best mean K/V pair: A layer %d <-> B layer %d (score=%.6f)",
        summary["kv_alignment"]["global_best_pair"]["src_layer_idx"],
        summary["kv_alignment"]["global_best_pair"]["dst_layer_idx"],
        summary["kv_alignment"]["global_best_pair"]["score"],
    )
    logger.info("Saved artifacts to %s", output_dir)
    return output_dir



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default-config-path",
        dest="default_config_path",
        default="configs/exp_layer_sim.json",
    )
    add_dataclass_arguments(parser, LayerSimConfig)
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_kwargs = build_dataclass_kwargs_from_json_and_namespace(
        config_cls=LayerSimConfig,
        default_config_path=args.default_config_path,
        args=args,
    )
    config = LayerSimConfig(**config_kwargs)
    output_dir = run_layer_similarity(config)
    print(f"Layer similarity outputs: {output_dir}")


if __name__ == "__main__":
    main()
