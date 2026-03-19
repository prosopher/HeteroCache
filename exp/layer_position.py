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
    model_ids: str = "gpt2,gpt2"
    model_directions: str = "A_to_B"
    reference_direction: Optional[str] = None
    position_layer_idx: Optional[int] = None
    injection_window_size: int = 1

    output_root: str = "outputs/layer_position"
    study_id: Optional[str] = None

    max_steps: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 4
    total_tokens: int = 128
    prefix_tokens: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 50
    grad_clip_norm: float = 1.0
    log_every: int = 25
    seed: int = 42
    shuffle_buffer: int = 50_000

    translator_dim: int = 1024
    translator_heads: int = 16
    translator_depth: int = 2
    translator_mlp_ratio: int = 4

    enable_principal_rotation: bool = True
    principal_rotation_streams: str = "k"
    principal_rotation_calibration_steps: int = 512

    device: str = "auto"
    dtype: str = "float32"

    eval_batch_size: int = 4
    eval_num_workers: int = 0
    eval_max_examples_per_dataset: int = 100
    eval_shuffle_stream: bool = False
    benchmark_mode: str = "gen_qa"
    generation_max_new_tokens: int = 64

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
        if self.benchmark_mode not in {"logit_qa", "gen_qa"}:
            raise ValueError("benchmark_mode must be one of {'logit_qa', 'gen_qa'}")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")
        normalized_streams = "".join(sorted(set(str(self.principal_rotation_streams).lower())))
        if normalized_streams not in {"", "k", "v", "kv"}:
            raise ValueError("principal_rotation_streams must be one of {'k', 'v', 'kv'}")
        self.principal_rotation_streams = normalized_streams or "kv"
        if self.enable_principal_rotation and self.principal_rotation_calibration_steps < 1:
            raise ValueError("principal_rotation_calibration_steps must be >= 1 when principal rotation is enabled")


@dataclass
class StreamPrincipalRotation:
    mean: torch.Tensor
    basis: torch.Tensor
    top_explained_variance_ratio: float


@dataclass
class DirectionalPrincipalRotations:
    src_key: List[StreamPrincipalRotation]
    dst_key: List[StreamPrincipalRotation]
    src_value: List[StreamPrincipalRotation]
    dst_value: List[StreamPrincipalRotation]


class RunningCovarianceAccumulator:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = int(hidden_size)
        self.count = 0
        self.sum = torch.zeros(self.hidden_size, dtype=torch.float64)
        self.xtx = torch.zeros(self.hidden_size, self.hidden_size, dtype=torch.float64)

    def update(self, samples: torch.Tensor) -> None:
        flattened = samples.detach().reshape(-1, samples.shape[-1]).to(device="cpu", dtype=torch.float64)
        if flattened.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RunningCovarianceAccumulator expected hidden size {self.hidden_size}, got {flattened.shape[-1]}"
            )
        self.count += int(flattened.shape[0])
        self.sum += flattened.sum(dim=0)
        self.xtx += flattened.transpose(0, 1) @ flattened

    def finalize(self) -> StreamPrincipalRotation:
        if self.count < 1:
            identity = torch.eye(self.hidden_size, dtype=torch.float32)
            return StreamPrincipalRotation(
                mean=torch.zeros(self.hidden_size, dtype=torch.float32),
                basis=identity,
                top_explained_variance_ratio=0.0,
            )
        mean = self.sum / float(self.count)
        covariance = self.xtx / float(self.count) - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        basis = eigenvectors[:, order]
        for column_idx in range(basis.shape[1]):
            pivot_idx = int(torch.argmax(basis[:, column_idx].abs()).item())
            if basis[pivot_idx, column_idx] < 0:
                basis[:, column_idx] = -basis[:, column_idx]
        total_variance = float(eigenvalues.clamp_min(0).sum().item())
        top_ratio = 0.0 if total_variance <= 0.0 else float(eigenvalues[0].clamp_min(0).item() / total_variance)
        return StreamPrincipalRotation(
            mean=mean.to(dtype=torch.float32),
            basis=basis.to(dtype=torch.float32),
            top_explained_variance_ratio=top_ratio,
        )


def make_identity_principal_rotation(hidden_size: int) -> StreamPrincipalRotation:
    return StreamPrincipalRotation(
        mean=torch.zeros(hidden_size, dtype=torch.float32),
        basis=torch.eye(hidden_size, dtype=torch.float32),
        top_explained_variance_ratio=0.0,
    )


def rotate_into_principal_basis(layer_cache: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    centered = layer_cache.float() - mean.to(device=layer_cache.device, dtype=torch.float32)
    rotated = torch.matmul(centered, basis.to(device=layer_cache.device, dtype=torch.float32))
    return rotated.to(dtype=layer_cache.dtype)


def inverse_rotate_from_principal_basis(rotated_cache: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    restored = torch.matmul(
        rotated_cache.float(),
        basis.to(device=rotated_cache.device, dtype=torch.float32).transpose(0, 1),
    )
    restored = restored + mean.to(device=rotated_cache.device, dtype=torch.float32)
    return restored.to(dtype=rotated_cache.dtype)


def summarize_principal_rotation_stats(stats: List[StreamPrincipalRotation]) -> Dict[str, Any]:
    ratios = [float(stat.top_explained_variance_ratio) for stat in stats]
    return {
        "num_layers": len(stats),
        "top_explained_variance_ratio": ratios,
        "average_top_explained_variance_ratio": float(sum(ratios) / len(ratios)) if ratios else float("nan"),
    }


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
        self.src_hidden_size = src_hidden_size
        self.dst_hidden_size = dst_hidden_size
        self.key_translator = CrossLayerWindowTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=translated_num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
        self.value_translator = CrossLayerWindowTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=translated_num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
        self.register_buffer("src_key_means", torch.zeros(translated_num_layers, src_hidden_size, dtype=torch.float32))
        self.register_buffer("dst_key_means", torch.zeros(translated_num_layers, dst_hidden_size, dtype=torch.float32))
        self.register_buffer(
            "src_key_bases",
            torch.eye(src_hidden_size, dtype=torch.float32).unsqueeze(0).repeat(translated_num_layers, 1, 1),
        )
        self.register_buffer(
            "dst_key_bases",
            torch.eye(dst_hidden_size, dtype=torch.float32).unsqueeze(0).repeat(translated_num_layers, 1, 1),
        )
        self.register_buffer("src_value_means", torch.zeros(translated_num_layers, src_hidden_size, dtype=torch.float32))
        self.register_buffer("dst_value_means", torch.zeros(translated_num_layers, dst_hidden_size, dtype=torch.float32))
        self.register_buffer(
            "src_value_bases",
            torch.eye(src_hidden_size, dtype=torch.float32).unsqueeze(0).repeat(translated_num_layers, 1, 1),
        )
        self.register_buffer(
            "dst_value_bases",
            torch.eye(dst_hidden_size, dtype=torch.float32).unsqueeze(0).repeat(translated_num_layers, 1, 1),
        )

    def set_principal_rotations(self, rotations: DirectionalPrincipalRotations) -> None:
        expected = self.translated_num_layers
        for stream_name, stats in {
            "src_key": rotations.src_key,
            "dst_key": rotations.dst_key,
            "src_value": rotations.src_value,
            "dst_value": rotations.dst_value,
        }.items():
            if len(stats) != expected:
                raise ValueError(
                    f"Directional principal rotation {stream_name} expected {expected} layers, got {len(stats)}"
                )

        self._copy_principal_rotation_list(self.src_key_means, self.src_key_bases, rotations.src_key)
        self._copy_principal_rotation_list(self.dst_key_means, self.dst_key_bases, rotations.dst_key)
        self._copy_principal_rotation_list(self.src_value_means, self.src_value_bases, rotations.src_value)
        self._copy_principal_rotation_list(self.dst_value_means, self.dst_value_bases, rotations.dst_value)

    def _copy_principal_rotation_list(
        self,
        mean_buffer: torch.Tensor,
        basis_buffer: torch.Tensor,
        stats: List[StreamPrincipalRotation],
    ) -> None:
        means = torch.stack([stat.mean for stat in stats], dim=0).to(device=mean_buffer.device, dtype=mean_buffer.dtype)
        bases = torch.stack([stat.basis for stat in stats], dim=0).to(device=basis_buffer.device, dtype=basis_buffer.dtype)
        mean_buffer.copy_(means)
        basis_buffer.copy_(bases)

    def _rotate_layer_window_into_basis(
        self,
        layer_cache: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
    ) -> torch.Tensor:
        rotated_layers = []
        for offset in range(self.translated_num_layers):
            rotated_layers.append(
                rotate_into_principal_basis(
                    layer_cache[:, :, offset, :],
                    means[offset],
                    bases[offset],
                ).unsqueeze(2)
            )
        return torch.cat(rotated_layers, dim=2)

    def _inverse_rotate_layer_window_from_basis(
        self,
        rotated_cache: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
    ) -> torch.Tensor:
        restored_layers = []
        for offset in range(self.translated_num_layers):
            restored_layers.append(
                inverse_rotate_from_principal_basis(
                    rotated_cache[:, :, offset, :],
                    means[offset],
                    bases[offset],
                ).unsqueeze(2)
            )
        return torch.cat(restored_layers, dim=2)

    def _translate_stream(
        self,
        layer_cache: torch.Tensor,
        translator: nn.Module,
        src_means: torch.Tensor,
        src_bases: torch.Tensor,
        dst_means: torch.Tensor,
        dst_bases: torch.Tensor,
    ) -> torch.Tensor:
        rotated_input = self._rotate_layer_window_into_basis(layer_cache, src_means, src_bases)
        translated_rotated = translator(rotated_input)
        return self._inverse_rotate_layer_window_from_basis(translated_rotated, dst_means, dst_bases)

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

        translated_key = self._translate_stream(
            layer_cache=key_block,
            translator=self.key_translator,
            src_means=self.src_key_means,
            src_bases=self.src_key_bases,
            dst_means=self.dst_key_means,
            dst_bases=self.dst_key_bases,
        )
        translated_value = self._translate_stream(
            layer_cache=value_block,
            translator=self.value_translator,
            src_means=self.src_value_means,
            src_bases=self.src_value_bases,
            dst_means=self.dst_value_means,
            dst_bases=self.dst_value_bases,
        )
        return translated_key, translated_value


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

    def set_principal_rotations(self, direction: str, rotations: DirectionalPrincipalRotations) -> None:
        if direction not in self.adapters:
            raise ValueError(f"Unknown direction for principal rotations: {direction}")
        self.adapters[direction].set_principal_rotations(rotations)

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


def principal_rotation_stream_is_enabled(config: LayerPositionConfig, stream_name: str) -> bool:
    stream_name = stream_name.lower()
    if stream_name not in {"k", "v"}:
        raise ValueError(f"Unknown principal rotation stream: {stream_name}")
    return config.enable_principal_rotation and stream_name in config.principal_rotation_streams


def build_principal_rotation_metadata_path(run_dir: Path) -> Path:
    return run_dir / "principal_rotation_metadata.json"


def save_principal_rotation_metadata(run_dir: Path, metadata: Dict[str, Any]) -> Path:
    metadata_path = build_principal_rotation_metadata_path(run_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)
    return metadata_path


def calibrate_principal_rotations(
    config: LayerPositionConfig,
    logger: logging.Logger,
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    nodes: List[Node],
    edges: List[Edge],
    translator_pool: LayerWindowTranslatorPool,
    layer_mappings: Dict[str, LayerMapping],
) -> Dict[str, Any]:
    if not config.enable_principal_rotation:
        return {
            "enabled": False,
            "streams": config.principal_rotation_streams,
            "calibration_steps": 0,
            "directions": {},
        }

    logger.info(
        "[Rotation] calibrating principal-basis rotations with %d batches (streams=%s)",
        config.principal_rotation_calibration_steps,
        config.principal_rotation_streams,
    )
    calibration_loader = build_training_dataloader(
        tokenizer=tokenizer,
        config=SimpleNamespaceConfig(
            total_tokens=config.total_tokens,
            batch_size=config.batch_size,
            shuffle_buffer=config.shuffle_buffer,
            seed=config.seed,
        ),
    )
    edge_map = build_edge_map(edges)
    stats: Dict[str, Dict[str, List[RunningCovarianceAccumulator]]] = {}
    for direction in translator_pool.active_directions:
        edge = edge_map[direction]
        mapping = layer_mappings[direction]
        src_hidden = translator_pool.model_specs[edge.src_id].hidden_size
        dst_hidden = translator_pool.model_specs[edge.dst_id].hidden_size
        stats[direction] = {
            "src_key": [RunningCovarianceAccumulator(src_hidden) for _ in range(mapping.translated_num_layers)],
            "dst_key": [RunningCovarianceAccumulator(dst_hidden) for _ in range(mapping.translated_num_layers)],
            "src_value": [RunningCovarianceAccumulator(src_hidden) for _ in range(mapping.translated_num_layers)],
            "dst_value": [RunningCovarianceAccumulator(dst_hidden) for _ in range(mapping.translated_num_layers)],
        }

    progress_bar = tqdm(range(config.principal_rotation_calibration_steps), desc="PrincipalRotationCalib")
    for _ in progress_bar:
        input_ids = next(calibration_loader).to(config.device)
        prefix_cache_ids, _, _ = split_prefix_and_suffix_for_exact_next_token_loss(
            input_ids=input_ids,
            prefix_tokens=config.prefix_tokens,
        )
        with torch.no_grad():
            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], prefix_cache_ids)
                for node in nodes
            }

        for direction in translator_pool.active_directions:
            edge = edge_map[direction]
            mapping = layer_mappings[direction]
            src_key_block, src_value_block = extract_layer_window_blocks(
                past_key_values=past_by_node_id[edge.src_id],
                start_layer_idx=mapping.src_layer_idx,
                num_layers=mapping.translated_num_layers,
            )
            dst_key_block, dst_value_block = extract_layer_window_blocks(
                past_key_values=past_by_node_id[edge.dst_id],
                start_layer_idx=mapping.dst_layer_idx,
                num_layers=mapping.translated_num_layers,
            )
            for offset in range(mapping.translated_num_layers):
                if principal_rotation_stream_is_enabled(config, "k"):
                    stats[direction]["src_key"][offset].update(src_key_block[:, :, offset, :])
                    stats[direction]["dst_key"][offset].update(dst_key_block[:, :, offset, :])
                if principal_rotation_stream_is_enabled(config, "v"):
                    stats[direction]["src_value"][offset].update(src_value_block[:, :, offset, :])
                    stats[direction]["dst_value"][offset].update(dst_value_block[:, :, offset, :])

    metadata = {
        "enabled": True,
        "streams": config.principal_rotation_streams,
        "calibration_steps": int(config.principal_rotation_calibration_steps),
        "directions": {},
    }
    for direction in translator_pool.active_directions:
        edge = edge_map[direction]
        src_hidden = translator_pool.model_specs[edge.src_id].hidden_size
        dst_hidden = translator_pool.model_specs[edge.dst_id].hidden_size
        src_key_rotations = (
            [acc.finalize() for acc in stats[direction]["src_key"]]
            if principal_rotation_stream_is_enabled(config, "k")
            else [make_identity_principal_rotation(src_hidden) for _ in stats[direction]["src_key"]]
        )
        dst_key_rotations = (
            [acc.finalize() for acc in stats[direction]["dst_key"]]
            if principal_rotation_stream_is_enabled(config, "k")
            else [make_identity_principal_rotation(dst_hidden) for _ in stats[direction]["dst_key"]]
        )
        src_value_rotations = (
            [acc.finalize() for acc in stats[direction]["src_value"]]
            if principal_rotation_stream_is_enabled(config, "v")
            else [make_identity_principal_rotation(src_hidden) for _ in stats[direction]["src_value"]]
        )
        dst_value_rotations = (
            [acc.finalize() for acc in stats[direction]["dst_value"]]
            if principal_rotation_stream_is_enabled(config, "v")
            else [make_identity_principal_rotation(dst_hidden) for _ in stats[direction]["dst_value"]]
        )
        rotations = DirectionalPrincipalRotations(
            src_key=src_key_rotations,
            dst_key=dst_key_rotations,
            src_value=src_value_rotations,
            dst_value=dst_value_rotations,
        )
        translator_pool.set_principal_rotations(direction, rotations)
        metadata["directions"][direction] = {
            "src_key": summarize_principal_rotation_stats(src_key_rotations),
            "dst_key": summarize_principal_rotation_stats(dst_key_rotations),
            "src_value": summarize_principal_rotation_stats(src_value_rotations),
            "dst_value": summarize_principal_rotation_stats(dst_value_rotations),
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


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


class ControlMetricMeter:
    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self.native_sum = 0.0
        self.dir_only_sum = 0.0
        self.mag_only_sum = 0.0
        self.full_mix_sum = 0.0
        self.count = 0

    def update(
        self,
        native_value: float,
        dir_only_value: float,
        mag_only_value: float,
        full_mix_value: float,
        n: int = 1,
    ) -> None:
        self.native_sum += float(native_value) * n
        self.dir_only_sum += float(dir_only_value) * n
        self.mag_only_sum += float(mag_only_value) * n
        self.full_mix_sum += float(full_mix_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                self.metric_name: float("nan"),
                f"native_{self.metric_name}": float("nan"),
                f"dir_only_{self.metric_name}": float("nan"),
                f"mag_only_{self.metric_name}": float("nan"),
                f"full_mix_{self.metric_name}": float("nan"),
                "delta_dir_only": float("nan"),
                "delta_mag_only": float("nan"),
                "delta_full_mix": float("nan"),
                "count": 0,
            }
        native_value = self.native_sum / self.count
        dir_only_value = self.dir_only_sum / self.count
        mag_only_value = self.mag_only_sum / self.count
        full_mix_value = self.full_mix_sum / self.count
        delta_dir_only = dir_only_value - native_value
        delta_mag_only = mag_only_value - native_value
        delta_full_mix = full_mix_value - native_value
        return {
            self.metric_name: full_mix_value,
            f"native_{self.metric_name}": native_value,
            f"dir_only_{self.metric_name}": dir_only_value,
            f"mag_only_{self.metric_name}": mag_only_value,
            f"full_mix_{self.metric_name}": full_mix_value,
            "delta_dir_only": delta_dir_only,
            "delta_mag_only": delta_mag_only,
            "delta_full_mix": delta_full_mix,
            "count": self.count,
        }


class LogitKLMeter:
    def __init__(self) -> None:
        self.native_to_dir_only_sum = 0.0
        self.native_to_mag_only_sum = 0.0
        self.native_to_full_mix_sum = 0.0
        self.full_mix_to_dir_only_sum = 0.0
        self.full_mix_to_mag_only_sum = 0.0
        self.count = 0

    def update(
        self,
        native_to_dir_only: float,
        native_to_mag_only: float,
        native_to_full_mix: float,
        full_mix_to_dir_only: float,
        full_mix_to_mag_only: float,
        n: int = 1,
    ) -> None:
        self.native_to_dir_only_sum += float(native_to_dir_only) * n
        self.native_to_mag_only_sum += float(native_to_mag_only) * n
        self.native_to_full_mix_sum += float(native_to_full_mix) * n
        self.full_mix_to_dir_only_sum += float(full_mix_to_dir_only) * n
        self.full_mix_to_mag_only_sum += float(full_mix_to_mag_only) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "native_to_dir_only_logit_kl": float("nan"),
                "native_to_mag_only_logit_kl": float("nan"),
                "native_to_full_mix_logit_kl": float("nan"),
                "full_mix_to_dir_only_logit_kl": float("nan"),
                "full_mix_to_mag_only_logit_kl": float("nan"),
                "count": 0,
            }
        return {
            "native_to_dir_only_logit_kl": self.native_to_dir_only_sum / self.count,
            "native_to_mag_only_logit_kl": self.native_to_mag_only_sum / self.count,
            "native_to_full_mix_logit_kl": self.native_to_full_mix_sum / self.count,
            "full_mix_to_dir_only_logit_kl": self.full_mix_to_dir_only_sum / self.count,
            "full_mix_to_mag_only_logit_kl": self.full_mix_to_mag_only_sum / self.count,
            "count": self.count,
        }


def rescale_block_to_reference_norm(source_block: torch.Tensor, reference_block: torch.Tensor) -> torch.Tensor:
    if source_block.shape != reference_block.shape:
        raise ValueError(
            "Source/reference blocks must have identical shapes, "
            f"got {tuple(source_block.shape)} vs {tuple(reference_block.shape)}"
        )
    source_norm = source_block.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reference_norm = reference_block.float().norm(dim=-1, keepdim=True)
    scaled = source_block.float() * (reference_norm / source_norm)
    return scaled.to(dtype=source_block.dtype)


def build_direction_only_window(
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        rescale_block_to_reference_norm(translated_key_block, native_key_block),
        rescale_block_to_reference_norm(translated_value_block, native_value_block),
    )


def build_magnitude_only_window(
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        rescale_block_to_reference_norm(native_key_block, translated_key_block),
        rescale_block_to_reference_norm(native_value_block, translated_value_block),
    )


def compute_next_token_log_probs(
    model: PreTrainedModel,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=seed_token,
        past_key_values=past_key_values,
        use_cache=False,
    )
    return F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)


def compute_logit_kl(reference_log_probs: torch.Tensor, candidate_log_probs: torch.Tensor) -> float:
    if reference_log_probs.shape != candidate_log_probs.shape:
        raise ValueError(
            "Reference/candidate log-prob shapes must match, "
            f"got {tuple(reference_log_probs.shape)} vs {tuple(candidate_log_probs.shape)}"
        )
    reference_probs = reference_log_probs.exp()
    kl = torch.sum(reference_probs * (reference_log_probs - candidate_log_probs), dim=-1)
    return float(kl.mean().item())


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


def build_control_window_variants(
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    dir_only_key, dir_only_value = build_direction_only_window(
        translated_key_block=translated_key_block,
        translated_value_block=translated_value_block,
        native_key_block=native_key_block,
        native_value_block=native_value_block,
    )
    mag_only_key, mag_only_value = build_magnitude_only_window(
        translated_key_block=translated_key_block,
        translated_value_block=translated_value_block,
        native_key_block=native_key_block,
        native_value_block=native_value_block,
    )
    return {
        "dir_only": (dir_only_key, dir_only_value),
        "mag_only": (mag_only_key, mag_only_value),
        "full_mix": (translated_key_block, translated_value_block),
    }


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
    logger.info(
        "principal_rotation=%s | streams=%s | calibration_steps=%d",
        config.enable_principal_rotation,
        config.principal_rotation_streams,
        config.principal_rotation_calibration_steps,
    )
    logger.info("experiment_config=%s", asdict(config))

    translator_pool, model_specs, layer_mappings = build_translator_pool(
        models=models,
        config=config,
        active_directions=active_directions,
        edges=edges,
        reference_edge=reference_edge,
    )
    rotation_metadata = calibrate_principal_rotations(
        config=config,
        logger=logger,
        models=models,
        tokenizer=tokenizer,
        nodes=nodes,
        edges=edges,
        translator_pool=translator_pool,
        layer_mappings=layer_mappings,
    )
    rotation_metadata_path = save_principal_rotation_metadata(run_dir, rotation_metadata)
    logger.info("[Rotation] metadata saved to %s", rotation_metadata_path)

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
    path_metrics = {direction: ControlMetricMeter("accuracy") for direction in active_directions}
    path_logit_kl = {direction: LogitKLMeter() for direction in active_directions}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            prepared_inputs = prepare_logit_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=example.get("context"),
                question=example["question"],
                device=config.device,
            )
            candidate_token_ids = build_logit_answer_candidates(tokenizer=tokenizer, spec=spec)
            gold_answer = example["answer"]
            context_input_ids = prepared_inputs["cache_input_ids"]
            question_cache_ids = prepared_inputs["question_cache_ids"]
            seed_token = prepared_inputs["seed_token"]
            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], context_input_ids)
                for node in nodes
            }

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                native_target_past = past_by_node_id[edge.dst_id]
                native_key_block, native_value_block = extract_layer_window_blocks(
                    past_key_values=native_target_past,
                    start_layer_idx=mapping.dst_layer_idx,
                    num_layers=mapping.translated_num_layers,
                )
                control_windows = build_control_window_variants(
                    native_key_block=native_key_block,
                    native_value_block=native_value_block,
                    translated_key_block=translated_key,
                    translated_value_block=translated_value,
                )
                dir_only_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=context_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=control_windows["dir_only"][0],
                    injected_value_block=control_windows["dir_only"][1],
                    dst_spec=model_specs[edge.dst_id],
                )
                mag_only_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=context_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=control_windows["mag_only"][0],
                    injected_value_block=control_windows["mag_only"][1],
                    dst_spec=model_specs[edge.dst_id],
                )
                full_mix_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=context_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=control_windows["full_mix"][0],
                    injected_value_block=control_windows["full_mix"][1],
                    dst_spec=model_specs[edge.dst_id],
                )

                native_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=native_target_past,
                    question_cache_ids=question_cache_ids,
                )
                dir_only_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=dir_only_past,
                    question_cache_ids=question_cache_ids,
                )
                mag_only_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=mag_only_past,
                    question_cache_ids=question_cache_ids,
                )
                full_mix_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=full_mix_past,
                    question_cache_ids=question_cache_ids,
                )

                native_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                dir_only_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=dir_only_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                mag_only_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=mag_only_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                full_mix_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=full_mix_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )

                native_pred = predict_answer_label(native_scores)
                dir_only_pred = predict_answer_label(dir_only_scores)
                mag_only_pred = predict_answer_label(mag_only_scores)
                full_mix_pred = predict_answer_label(full_mix_scores)
                path_metrics[direction].update(
                    native_value=1.0 if native_pred == gold_answer else 0.0,
                    dir_only_value=1.0 if dir_only_pred == gold_answer else 0.0,
                    mag_only_value=1.0 if mag_only_pred == gold_answer else 0.0,
                    full_mix_value=1.0 if full_mix_pred == gold_answer else 0.0,
                    n=1,
                )

                native_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                )
                dir_only_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=dir_only_scoring_past,
                    seed_token=seed_token,
                )
                mag_only_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=mag_only_scoring_past,
                    seed_token=seed_token,
                )
                full_mix_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=full_mix_scoring_past,
                    seed_token=seed_token,
                )
                path_logit_kl[direction].update(
                    native_to_dir_only=compute_logit_kl(native_log_probs, dir_only_log_probs),
                    native_to_mag_only=compute_logit_kl(native_log_probs, mag_only_log_probs),
                    native_to_full_mix=compute_logit_kl(native_log_probs, full_mix_log_probs),
                    full_mix_to_dir_only=compute_logit_kl(full_mix_log_probs, dir_only_log_probs),
                    full_mix_to_mag_only=compute_logit_kl(full_mix_log_probs, mag_only_log_probs),
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
    summarized_logit_kl = {direction: meter.summary() for direction, meter in path_logit_kl.items()}
    return summarized_metrics, summarized_logit_kl


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
    path_metrics = {direction: ControlMetricMeter("f1") for direction in active_directions}
    path_logit_kl = {direction: LogitKLMeter() for direction in active_directions}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            context_budget = compute_benchmark_context_budget(
                tokenizer=tokenizer,
                spec=spec,
                question=question,
                eval_config=config,
                models=models,
            )
            prepared_inputs = prepare_generation_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=context_text,
                question=question,
                device=config.device,
                max_input_tokens=context_budget,
            )
            cache_input_ids = prepared_inputs["cache_input_ids"]
            question_cache_ids = prepared_inputs["question_cache_ids"]
            seed_token = prepared_inputs["seed_token"]

            if prepared_inputs.get("was_truncated") and processed_examples < 3:
                question_cache_tokens = 0 if question_cache_ids is None else int(question_cache_ids.shape[1])
                logger.info(
                    "[%s] truncated context to %d tokens to fit model context window (question_cache_tokens=%d, answer_token_budget=%d)",
                    spec.name_for_log,
                    int(cache_input_ids.shape[1]),
                    question_cache_tokens,
                    get_answer_token_budget(config),
                )

            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], cache_input_ids)
                for node in nodes
            }

            for direction in active_directions:
                edge = edge_map[direction]
                translated_key, translated_value, mapping = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                )
                native_target_past = past_by_node_id[edge.dst_id]
                native_key_block, native_value_block = extract_layer_window_blocks(
                    past_key_values=native_target_past,
                    start_layer_idx=mapping.dst_layer_idx,
                    num_layers=mapping.translated_num_layers,
                )
                control_windows = build_control_window_variants(
                    native_key_block=native_key_block,
                    native_value_block=native_value_block,
                    translated_key_block=translated_key,
                    translated_value_block=translated_value,
                )
                dir_only_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=cache_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=control_windows["dir_only"][0],
                    injected_value_block=control_windows["dir_only"][1],
                    dst_spec=model_specs[edge.dst_id],
                )
                mag_only_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=cache_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=control_windows["mag_only"][0],
                    injected_value_block=control_windows["mag_only"][1],
                    dst_spec=model_specs[edge.dst_id],
                )
                full_mix_past = replay_target_prefill_with_injected_window(
                    target_model=models[edge.dst_id],
                    prefix_input_ids=cache_input_ids,
                    target_start_layer_idx=mapping.dst_layer_idx,
                    injected_key_block=control_windows["full_mix"][0],
                    injected_value_block=control_windows["full_mix"][1],
                    dst_spec=model_specs[edge.dst_id],
                )

                native_answer = predict_generation_task_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=native_target_past,
                    seed_token=seed_token,
                    eval_config=config,
                    question_cache_ids=question_cache_ids,
                )
                dir_only_answer = predict_generation_task_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=dir_only_past,
                    seed_token=seed_token,
                    eval_config=config,
                    question_cache_ids=question_cache_ids,
                )
                mag_only_answer = predict_generation_task_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=mag_only_past,
                    seed_token=seed_token,
                    eval_config=config,
                    question_cache_ids=question_cache_ids,
                )
                full_mix_answer = predict_generation_task_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=full_mix_past,
                    seed_token=seed_token,
                    eval_config=config,
                    question_cache_ids=question_cache_ids,
                )

                path_metrics[direction].update(
                    native_value=compute_generation_f1(native_answer, gold_answers),
                    dir_only_value=compute_generation_f1(dir_only_answer, gold_answers),
                    mag_only_value=compute_generation_f1(mag_only_answer, gold_answers),
                    full_mix_value=compute_generation_f1(full_mix_answer, gold_answers),
                    n=1,
                )

                native_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=native_target_past,
                    question_cache_ids=question_cache_ids,
                )
                dir_only_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=dir_only_past,
                    question_cache_ids=question_cache_ids,
                )
                mag_only_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=mag_only_past,
                    question_cache_ids=question_cache_ids,
                )
                full_mix_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=full_mix_past,
                    question_cache_ids=question_cache_ids,
                )

                native_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                )
                dir_only_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=dir_only_scoring_past,
                    seed_token=seed_token,
                )
                mag_only_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=mag_only_scoring_past,
                    seed_token=seed_token,
                )
                full_mix_log_probs = compute_next_token_log_probs(
                    model=models[edge.dst_id],
                    past_key_values=full_mix_scoring_past,
                    seed_token=seed_token,
                )
                path_logit_kl[direction].update(
                    native_to_dir_only=compute_logit_kl(native_log_probs, dir_only_log_probs),
                    native_to_mag_only=compute_logit_kl(native_log_probs, mag_only_log_probs),
                    native_to_full_mix=compute_logit_kl(native_log_probs, full_mix_log_probs),
                    full_mix_to_dir_only=compute_logit_kl(full_mix_log_probs, dir_only_log_probs),
                    full_mix_to_mag_only=compute_logit_kl(full_mix_log_probs, mag_only_log_probs),
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 25 == 0:
            logger.info(
                "[%s] generation progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                config.eval_max_examples_per_dataset,
            )

    summarized_metrics = {direction: meter.summary() for direction, meter in path_metrics.items()}
    summarized_logit_kl = {direction: meter.summary() for direction, meter in path_logit_kl.items()}
    return summarized_metrics, summarized_logit_kl

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
    average_dir_only_metric: float
    average_mag_only_metric: float
    average_full_mix_metric: float
    average_delta_dir_only: float
    average_delta_mag_only: float
    average_delta_full_mix: float
    average_native_to_dir_only_logit_kl: float
    average_native_to_mag_only_logit_kl: float
    average_native_to_full_mix_logit_kl: float
    average_full_mix_to_dir_only_logit_kl: float
    average_full_mix_to_mag_only_logit_kl: float
    run_dir: str

def build_summary_csv_path(study_dir: Path) -> Path:
    return study_dir / "summary.csv"


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
        "average_dir_only_metric": combined_metrics["average_dir_only_metric"],
        "average_mag_only_metric": combined_metrics["average_mag_only_metric"],
        "average_full_mix_metric": combined_metrics["average_full_mix_metric"],
        "average_delta_dir_only": combined_metrics["average_delta_dir_only"],
        "average_delta_mag_only": combined_metrics["average_delta_mag_only"],
        "average_delta_full_mix": combined_metrics["average_delta_full_mix"],
        f"average_{metric_name}": combined_metrics[f"average_{metric_name}"],
        f"average_native_{metric_name}": combined_metrics[f"average_native_{metric_name}"],
    }


def extract_analysis_metrics(combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark_mode": combined_metrics["benchmark_mode"],
        "metric_name": combined_metrics["metric_name"],
        "layer_mappings": combined_metrics["layer_mappings"],
        "dataset_logit_kl": combined_metrics["dataset_logit_kl"],
        "average_native_to_dir_only_logit_kl": combined_metrics["average_native_to_dir_only_logit_kl"],
        "average_native_to_mag_only_logit_kl": combined_metrics["average_native_to_mag_only_logit_kl"],
        "average_native_to_full_mix_logit_kl": combined_metrics["average_native_to_full_mix_logit_kl"],
        "average_full_mix_to_dir_only_logit_kl": combined_metrics["average_full_mix_to_dir_only_logit_kl"],
        "average_full_mix_to_mag_only_logit_kl": combined_metrics["average_full_mix_to_mag_only_logit_kl"],
        "average_delta_dir_only": combined_metrics["average_delta_dir_only"],
        "average_delta_mag_only": combined_metrics["average_delta_mag_only"],
        "average_delta_full_mix": combined_metrics["average_delta_full_mix"],
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
        average_dir_only_metric=float(metrics["average_dir_only_metric"]),
        average_mag_only_metric=float(metrics["average_mag_only_metric"]),
        average_full_mix_metric=float(metrics["average_full_mix_metric"]),
        average_delta_dir_only=float(metrics["average_delta_dir_only"]),
        average_delta_mag_only=float(metrics["average_delta_mag_only"]),
        average_delta_full_mix=float(metrics["average_delta_full_mix"]),
        average_native_to_dir_only_logit_kl=float(metrics["average_native_to_dir_only_logit_kl"]),
        average_native_to_mag_only_logit_kl=float(metrics["average_native_to_mag_only_logit_kl"]),
        average_native_to_full_mix_logit_kl=float(metrics["average_native_to_full_mix_logit_kl"]),
        average_full_mix_to_dir_only_logit_kl=float(metrics["average_full_mix_to_dir_only_logit_kl"]),
        average_full_mix_to_mag_only_logit_kl=float(metrics["average_full_mix_to_mag_only_logit_kl"]),
        run_dir=str(run_dir),
    )

def read_summary_rows(summary_path: Path) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    if not summary_path.exists():
        return rows
    field_names = SummaryRow.__dataclass_fields__
    with summary_path.open("r", encoding="utf-8", newline="") as fp:
        for existing in csv.DictReader(fp):
            try:
                payload = {}
                for field_name, field_def in field_names.items():
                    raw_value = existing[field_name]
                    if field_def.type is int:
                        payload[field_name] = int(raw_value)
                    elif field_def.type is float:
                        payload[field_name] = float(raw_value)
                    else:
                        payload[field_name] = raw_value
                rows.append(SummaryRow(**payload))
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


def build_analysis_metrics_path(run_dir: Path) -> Path:
    return run_dir / "control_analysis_metrics.json"


def build_metric_controls_chart_path(study_dir: Path, metric_name: str) -> Path:
    return study_dir / f"layer_idx_vs_{sanitize_slug(metric_name)}_controls.png"


def build_logit_kl_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_logit_kl.png"


def plot_metric_controls_summary(summary_path: Path) -> Path:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda row: row.position_layer_idx)
    x_values = [row.position_layer_idx for row in rows]
    metric_name = rows[0].metric_name or "metric"
    metric_label = metric_name.upper() if metric_name == "f1" else metric_name.capitalize()
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(x_values, [row.average_native_metric for row in rows], marker="o", label=f"Native {metric_label}")
    ax.plot(x_values, [row.average_dir_only_metric for row in rows], marker="s", label=f"Dir-only {metric_label}")
    ax.plot(x_values, [row.average_mag_only_metric for row in rows], marker="^", label=f"Mag-only {metric_label}")
    ax.plot(x_values, [row.average_full_mix_metric for row in rows], marker="D", label=f"Full-mix {metric_label}")
    annotate_injected_layer_ranges(ax, rows, lambda row: row.average_full_mix_metric)
    ax.set_xlabel("Reference target layer index")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} decomposition vs layer index ({window_title})")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    chart_path = build_metric_controls_chart_path(study_dir, metric_name)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def plot_logit_kl_summary(summary_path: Path) -> Path:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda row: row.position_layer_idx)
    x_values = [row.position_layer_idx for row in rows]
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(x_values, [row.average_native_to_dir_only_logit_kl for row in rows], marker="o", label="KL(native || dir-only)")
    ax.plot(x_values, [row.average_native_to_mag_only_logit_kl for row in rows], marker="s", label="KL(native || mag-only)")
    ax.plot(x_values, [row.average_native_to_full_mix_logit_kl for row in rows], marker="^", label="KL(native || full-mix)")
    ax.plot(x_values, [row.average_full_mix_to_dir_only_logit_kl for row in rows], marker="x", label="KL(full-mix || dir-only)")
    ax.plot(x_values, [row.average_full_mix_to_mag_only_logit_kl for row in rows], marker="d", label="KL(full-mix || mag-only)")
    ax.set_xlabel("Reference target layer index")
    ax.set_ylabel("KL divergence")
    ax.set_title(f"Logit KL comparison vs layer index ({window_title})")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    chart_path = build_logit_kl_chart_path(study_dir)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def remove_stale_summary_artifacts(study_dir: Path, run_dir: Path) -> None:
    stale_paths = [
        run_dir / "eval_summary.md",
        run_dir / "drift_metrics.json",
        run_dir / "control_analysis_metrics.json",
        study_dir / "drift_summary.md",
        study_dir / "study_summary.csv",
        study_dir / "drift_summary.csv",
        study_dir / "drift_cosine.png",
        study_dir / "drift_l2.png",
    ]
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()


def save_analysis_artifacts(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    write_json(str(build_analysis_metrics_path(run_dir)), metrics)
    return build_analysis_metrics_path(run_dir)


def save_run_artifacts(
    config: LayerPositionConfig,
    run_dir: Path,
    layer_mappings: Dict[str, LayerMapping],
    eval_metrics: Dict[str, Any],
    combined_metrics: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path]:
    study_dir = run_dir.parent
    study_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_summary_artifacts(study_dir, run_dir)
    write_json(str(build_config_path(run_dir)), asdict(config))
    write_json(str(build_layer_mapping_path(run_dir)), {direction: asdict(mapping) for direction, mapping in layer_mappings.items()})
    write_json(str(build_metrics_path(run_dir)), eval_metrics)
    summary_path = update_summary(config, run_dir, combined_metrics)
    metric_controls_chart_path = plot_metric_controls_summary(summary_path)
    logit_kl_chart_path = plot_logit_kl_summary(summary_path)
    return summary_path, build_metrics_path(run_dir), metric_controls_chart_path, logit_kl_chart_path

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
    logger.info(
        "principal_rotation=%s | streams=%s | calibration_steps=%d",
        config.enable_principal_rotation,
        config.principal_rotation_streams,
        config.principal_rotation_calibration_steps,
    )
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
    dataset_logit_kl_by_name: Dict[str, Dict[str, Dict[str, float]]] = {}

    if config.benchmark_mode == "logit_qa":
        metric_name = "accuracy"
        dataset_results_key = "dataset_accuracies"
        dataset_specs = get_eval_spec_group("logit_qa")
        dataset_evaluator = evaluate_logit_dataset
        dataloader_builder = build_eval_dataloader
        progress_log_template = (
            "[%s] %s | native_%s=%.6f | dir_only_%s=%.6f | mag_only_%s=%.6f | "
            "full_mix_%s=%.6f | delta_dir_only=%.6f | delta_mag_only=%.6f | "
            "delta_full_mix=%.6f | kl(native||dir)=%.6f | "
            "kl(native||mag)=%.6f | kl(native||full)=%.6f | kl(full||dir)=%.6f | "
            "kl(full||mag)=%.6f | count=%d"
        )
    elif config.benchmark_mode == "gen_qa":
        metric_name = "f1"
        dataset_results_key = "dataset_f1"
        dataset_specs = get_eval_spec_group("gen_qa")
        dataset_evaluator = evaluate_generation_dataset
        dataloader_builder = build_generation_eval_dataloader
        progress_log_template = (
            "[%s] %s | native_%s=%.6f | dir_only_%s=%.6f | mag_only_%s=%.6f | "
            "full_mix_%s=%.6f | delta_dir_only=%.6f | delta_mag_only=%.6f | "
            "delta_full_mix=%.6f | kl(native||dir)=%.6f | "
            "kl(native||mag)=%.6f | kl(native||full)=%.6f | kl(full||dir)=%.6f | "
            "kl(full||mag)=%.6f | count=%d"
        )
    else:
        raise ValueError(f"Unsupported benchmark_mode: {config.benchmark_mode}")

    for spec in dataset_specs:
        dataloader = dataloader_builder(spec=spec, eval_config=eval_config)
        dataset_results, dataset_logit_kl = dataset_evaluator(
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
        dataset_logit_kl_by_name[spec.name_for_log] = dataset_logit_kl
        for direction in active_directions:
            metric_row = dataset_results[direction]
            logit_row = dataset_logit_kl[direction]
            logger.info(
                progress_log_template,
                spec.name_for_log,
                direction,
                metric_name, metric_row[f"native_{metric_name}"],
                metric_name, metric_row[f"dir_only_{metric_name}"],
                metric_name, metric_row[f"mag_only_{metric_name}"],
                metric_name, metric_row[f"full_mix_{metric_name}"],
                metric_row["delta_dir_only"],
                metric_row["delta_mag_only"],
                metric_row["delta_full_mix"],
                logit_row["native_to_dir_only_logit_kl"],
                logit_row["native_to_mag_only_logit_kl"],
                logit_row["native_to_full_mix_logit_kl"],
                logit_row["full_mix_to_dir_only_logit_kl"],
                logit_row["full_mix_to_mag_only_logit_kl"],
                int(metric_row["count"]),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    average_native_metric = compute_average_metric(dataset_results_by_name, f"native_{metric_name}")
    average_dir_only_metric = compute_average_metric(dataset_results_by_name, f"dir_only_{metric_name}")
    average_mag_only_metric = compute_average_metric(dataset_results_by_name, f"mag_only_{metric_name}")
    average_full_mix_metric = compute_average_metric(dataset_results_by_name, f"full_mix_{metric_name}")
    average_delta_dir_only = compute_average_metric(dataset_results_by_name, "delta_dir_only")
    average_delta_mag_only = compute_average_metric(dataset_results_by_name, "delta_mag_only")
    average_delta_full_mix = compute_average_metric(dataset_results_by_name, "delta_full_mix")
    average_native_to_dir_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "native_to_dir_only_logit_kl")
    average_native_to_mag_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "native_to_mag_only_logit_kl")
    average_native_to_full_mix_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "native_to_full_mix_logit_kl")
    average_full_mix_to_dir_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "full_mix_to_dir_only_logit_kl")
    average_full_mix_to_mag_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "full_mix_to_mag_only_logit_kl")

    logger.info(
        "[Summary] metric=%s | native=%.6f | dir_only=%.6f | mag_only=%.6f | full_mix=%.6f",
        metric_name,
        average_native_metric,
        average_dir_only_metric,
        average_mag_only_metric,
        average_full_mix_metric,
    )
    logger.info(
        "[Summary] delta_dir_only=%.6f | delta_mag_only=%.6f | delta_full_mix=%.6f",
        average_delta_dir_only,
        average_delta_mag_only,
        average_delta_full_mix,
    )
    logger.info(
        "[Summary] avg_kl(native||dir)=%.6f | avg_kl(native||mag)=%.6f | avg_kl(native||full)=%.6f | avg_kl(full||dir)=%.6f | avg_kl(full||mag)=%.6f",
        average_native_to_dir_only_logit_kl,
        average_native_to_mag_only_logit_kl,
        average_native_to_full_mix_logit_kl,
        average_full_mix_to_dir_only_logit_kl,
        average_full_mix_to_mag_only_logit_kl,
    )
    return {
        "benchmark_mode": config.benchmark_mode,
        "metric_name": metric_name,
        "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        dataset_results_key: dataset_results_by_name,
        "average_metric": average_full_mix_metric,
        "average_native_metric": average_native_metric,
        "average_dir_only_metric": average_dir_only_metric,
        "average_mag_only_metric": average_mag_only_metric,
        "average_full_mix_metric": average_full_mix_metric,
        "average_delta_dir_only": average_delta_dir_only,
        "average_delta_mag_only": average_delta_mag_only,
        "average_delta_full_mix": average_delta_full_mix,
        f"average_{metric_name}": average_full_mix_metric,
        f"average_native_{metric_name}": average_native_metric,
        "dataset_logit_kl": dataset_logit_kl_by_name,
        "average_native_to_dir_only_logit_kl": average_native_to_dir_only_logit_kl,
        "average_native_to_mag_only_logit_kl": average_native_to_mag_only_logit_kl,
        "average_native_to_full_mix_logit_kl": average_native_to_full_mix_logit_kl,
        "average_full_mix_to_dir_only_logit_kl": average_full_mix_to_dir_only_logit_kl,
        "average_full_mix_to_mag_only_logit_kl": average_full_mix_to_mag_only_logit_kl,
    }


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train and evaluate a translated layer window anchored at a chosen reference-target layer position by replaying target prefill below the window, injecting the translated window, and continuing above it. Source/target KV caches can be rotated into SVD principal coordinates before translation and inverse-rotated afterward. Task decomposition and logit-KL comparison are run automatically as part of the same execution."
    )
    parser.add_argument("--model-ids", default="gpt2,gpt2")
    parser.add_argument("--model-directions", default="A_to_B")
    parser.add_argument("--reference-direction", default=None)
    parser.add_argument("--position-layer-idx", type=int, default=None, help="Reference target layer index to use as the anchor layer for translation/injection sweeps.")
    parser.add_argument("--injection-window-size", type=int, default=1, help="Total number of consecutive layers to translate and inject, starting from the anchor layer selected by --position-layer-idx. For example, 1 injects only the anchor layer, and 3 injects the anchor layer plus the next two upper layers.")
    parser.add_argument("--print-target-num-layers", action="store_true")

    parser.add_argument("--output-root", default="outputs/layer_position")
    parser.add_argument("--study-id", default=None)

    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--total-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=50_000)

    parser.add_argument("--translator-dim", type=int, default=1024)
    parser.add_argument("--translator-heads", type=int, default=16)
    parser.add_argument("--translator-depth", type=int, default=2)
    parser.add_argument("--translator-mlp-ratio", type=int, default=4)
    parser.add_argument(
        "--disable-principal-rotation",
        action="store_true",
        help="Disable SVD principal-basis rotation before translation and inverse rotation after translation.",
    )
    parser.add_argument(
        "--principal-rotation-streams",
        choices=["k", "v", "kv"],
        default="k",
        help="Which cache streams to rotate into principal coordinates before translation.",
    )
    parser.add_argument(
        "--principal-rotation-calibration-steps",
        type=int,
        default=512,
        help="Number of prefix batches used to estimate per-direction principal bases.",
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-max-examples-per-dataset", type=int, default=100)
    parser.add_argument("--eval-shuffle-stream", action="store_true")
    parser.add_argument("--benchmark-mode", choices=["logit_qa", "gen_qa"], default="logit_qa")
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
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
        enable_principal_rotation=not args.disable_principal_rotation,
        principal_rotation_streams=args.principal_rotation_streams,
        principal_rotation_calibration_steps=args.principal_rotation_calibration_steps,
        device=args.device,
        dtype=args.dtype,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        eval_max_examples_per_dataset=args.eval_max_examples_per_dataset,
        eval_shuffle_stream=args.eval_shuffle_stream,
        benchmark_mode=args.benchmark_mode,
        generation_max_new_tokens=args.generation_max_new_tokens,
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
    analysis_metrics = extract_analysis_metrics(combined_metrics)

    summary_path, metrics_path, metric_controls_chart_path, logit_kl_chart_path = save_run_artifacts(
        config=config,
        run_dir=run_dir,
        layer_mappings=layer_mappings,
        eval_metrics=eval_metrics,
        combined_metrics=combined_metrics,
    )
    analysis_metrics_path = save_analysis_artifacts(run_dir=run_dir, metrics=analysis_metrics)

    print(f"Run directory: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Metric controls chart: {metric_controls_chart_path}")
    print(f"Control analysis metrics: {analysis_metrics_path}")
    print(f"Principal rotation metadata: {build_principal_rotation_metadata_path(run_dir)}")
    print(f"Logit KL chart: {logit_kl_chart_path}")


if __name__ == "__main__":
    main()
