#!/usr/bin/env python3
import argparse
import csv
import logging
import math
import numpy as np
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import *
from eval_util import *
import exp.layer_position as lp


@dataclass
class CorrectionConfig:
    model_ids: str
    model_directions: str
    reference_direction: Optional[str]
    position_layer_idx: Optional[int]
    injection_window_size: int

    output_root: str
    study_id: Optional[str]

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

    translator_dim: int
    translator_heads: int
    translator_depth: int
    translator_mlp_ratio: int

    enable_principal_rotation: bool
    principal_rotation_streams: str
    principal_rotation_calibration_steps: int

    device: str
    dtype: str

    eval_batch_size: int
    eval_num_workers: int
    eval_max_examples_per_dataset: int
    eval_shuffle_stream: bool
    benchmark_mode: str
    generation_max_new_tokens: int

    correction_max_analysis_tokens: int
    correction_include_random_control: bool

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
        if self.correction_max_analysis_tokens < 1:
            raise ValueError("correction_max_analysis_tokens must be >= 1")


@dataclass
class CorrectionSummaryRow:
    study_id: str
    benchmark_mode: str
    position_layer_idx: int
    translated_num_layers: int
    source_layer_idx: int
    source_layer_end_idx: int
    target_layer_idx: int
    target_layer_end_idx: int
    num_samples: int
    num_tokens: int
    average_initial_shift_norm: float
    average_final_shift_norm: float
    average_final_shrink_ratio: float
    median_final_shrink_ratio: float
    shrink_fraction: float
    average_final_alpha: float
    average_final_alpha_over_initial: float
    average_final_beta: float
    average_final_beta_over_initial: float
    average_final_correction_cosine: float
    average_final_attn_alpha_over_initial: float
    average_final_mlp_alpha_over_initial: float
    average_random_final_shrink_ratio: float
    average_random_shrink_fraction: float
    average_random_final_correction_cosine: float
    run_dir: str
    post_window_boundary_idx: int = 0
    num_upper_layers: int = 0
    remaining_correction_layers: int = 0


def build_study_dir(config: CorrectionConfig) -> Path:
    study_id = config.study_id or f"run_{sanitize_slug(config.model_directions)}"
    return Path(config.output_root) / study_id


def build_run_output_dir(config: CorrectionConfig) -> Path:
    study_dir = build_study_dir(config)
    position_label = f"layer_idx_{int(config.position_layer_idx):03d}"
    return study_dir / position_label


def build_train_log_path(run_dir: Path) -> Path:
    return run_dir / "correction_training.log"


def build_eval_log_path(run_dir: Path) -> Path:
    return run_dir / "correction_evaluation.log"


def build_config_path(run_dir: Path) -> Path:
    return run_dir / "correction_run_config.json"


def build_metrics_path(run_dir: Path) -> Path:
    return run_dir / "correction_metrics.json"


def build_summary_path(study_dir: Path) -> Path:
    return study_dir / "correction_summary.csv"


def build_norm_ratio_chart_path(run_dir: Path) -> Path:
    return run_dir / "correction_norm_ratio_trajectory.png"


def build_projection_chart_path(run_dir: Path) -> Path:
    return run_dir / "correction_projection_trajectory.png"


def build_summary_shrink_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_final_shrink_ratio.png"


def build_summary_ratio_distribution_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_shrink_ratio_distribution.png"


def build_summary_decomposition_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_correction_decomposition.png"


def build_summary_random_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_structural_advantage.png"


def build_summary_phase_chart_path(study_dir: Path) -> Path:
    return study_dir / "correction_phase_scatter.png"


def build_summary_shift_norm_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_shift_norms.png"


def format_layer_range(start_idx: int, end_idx: int) -> str:
    if start_idx == end_idx:
        return f"L{start_idx}"
    return f"L{start_idx}-L{end_idx}"


def _nanmean(values: List[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def _nanmedian(values: List[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(valid)) if valid else float("nan")


def _fraction(values: List[bool]) -> float:
    return float(sum(1 for v in values if v) / len(values)) if values else float("nan")


def build_input_hidden_states_with_past(model: PreTrainedModel, input_ids: torch.Tensor, past_length: int) -> torch.Tensor:
    transformer = lp.require_gpt2_transformer(model)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(past_length, past_length + seq_len, device=input_ids.device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
    drop = getattr(transformer, "drop", None)
    if drop is not None:
        hidden_states = drop(hidden_states)
    return hidden_states


def run_gpt2_block_with_past_and_trace(
    block: nn.Module,
    hidden_states: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    residual = hidden_states
    attn_input = block.ln_1(hidden_states)
    attn_outputs = block.attn(
        attn_input,
        layer_past=layer_past,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=True,
        output_attentions=False,
    )
    if not isinstance(attn_outputs, tuple) or len(attn_outputs) < 2:
        raise ValueError("Expected GPT-2 attention outputs to include present cache")
    attn_output = attn_outputs[0]
    present = attn_outputs[1]
    if not isinstance(present, tuple) or len(present) != 2:
        raise ValueError("Expected present cache to be a (key, value) tuple")
    hidden_after_attn = residual + attn_output
    mlp_input = block.ln_2(hidden_after_attn)
    mlp_output = block.mlp(mlp_input)
    hidden_after_block = hidden_after_attn + mlp_output
    return hidden_after_block, present, attn_output, mlp_output


@torch.inference_mode()
def trace_single_token_with_past(
    model: PreTrainedModel,
    past_key_values: PastKeyValues,
    input_ids: torch.Tensor,
) -> Dict[str, Any]:
    transformer = lp.require_gpt2_transformer(model)
    past_length = 0 if len(past_key_values) == 0 else int(past_key_values[0][0].shape[2])
    hidden_states = build_input_hidden_states_with_past(model, input_ids, past_length)
    token_hidden_states = [hidden_states[:, -1, :].detach()]
    token_attn_additions: List[torch.Tensor] = []
    token_mlp_additions: List[torch.Tensor] = []
    updated_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, block in enumerate(transformer.h):
        layer_past = None if layer_idx >= len(past_key_values) else past_key_values[layer_idx]
        hidden_states, present, attn_output, mlp_output = run_gpt2_block_with_past_and_trace(
            block=block,
            hidden_states=hidden_states,
            layer_past=layer_past,
        )
        updated_past.append((present[0].detach(), present[1].detach()))
        token_attn_additions.append(attn_output[:, -1, :].detach())
        token_mlp_additions.append(mlp_output[:, -1, :].detach())
        token_hidden_states.append(hidden_states[:, -1, :].detach())
    return {
        "past_key_values": tuple(updated_past),
        "hidden_states": torch.cat(token_hidden_states, dim=0),
        "attn_additions": torch.cat(token_attn_additions, dim=0),
        "mlp_additions": torch.cat(token_mlp_additions, dim=0),
    }


def build_random_matched_window(
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def _build(native_block: torch.Tensor, translated_block: torch.Tensor) -> torch.Tensor:
        delta = translated_block - native_block
        delta_norm = delta.norm(dim=-1, keepdim=True)
        noise = torch.randn_like(delta)
        noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        matched_noise = noise / noise_norm * delta_norm
        return native_block + matched_noise
    return _build(native_key_block, translated_key_block), _build(native_value_block, translated_value_block)


def build_teacher_forcing_answer_token_ids(
    spec: HFDatasetSpec,
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[torch.Tensor]:
    if spec.answer_mode in {"boolq", "pubmed_qa"}:
        answer_value = example.get("answer", None)
        if not isinstance(answer_value, str) or not answer_value.strip():
            return None
        choice_ids = build_logit_answer_candidates(tokenizer, spec)
        token_ids = choice_ids.get(answer_value.strip().lower())
        return None if token_ids is None else token_ids.clone()

    if spec.answer_mode in {"squad", "newsqa"}:
        answers = example.get("answers", None)
        if not isinstance(answers, list) or not answers:
            return None
        gold_answer = next((text.strip() for text in answers if isinstance(text, str) and text.strip()), None)
        if not gold_answer:
            return None
        token_ids = tokenizer(f" {gold_answer}", add_special_tokens=False).input_ids
        if len(token_ids) < 1:
            return None
        return torch.tensor(token_ids, dtype=torch.long)

    raise ValueError(f"Unsupported answer_mode for correction analysis: {spec.answer_mode}")


def build_prepared_inputs(
    spec: HFDatasetSpec,
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    models: Dict[str, PreTrainedModel],
    config: CorrectionConfig,
) -> Dict[str, Any]:
    if config.benchmark_mode == "logit_qa":
        return prepare_logit_task_inputs(
            spec=spec,
            tokenizer=tokenizer,
            context=example.get("context", None),
            question=example["question"],
            device=config.device,
        )
    if config.benchmark_mode == "gen_qa":
        context_budget = compute_benchmark_context_budget(
            tokenizer=tokenizer,
            spec=spec,
            question=example["question"],
            eval_config=SimpleNamespace(generation_max_new_tokens=config.generation_max_new_tokens),
            models=models,
        )
        return prepare_generation_task_inputs(
            spec=spec,
            tokenizer=tokenizer,
            context=example["context"],
            question=example["question"],
            device=config.device,
            max_input_tokens=context_budget,
        )
    raise ValueError(f"Unsupported benchmark_mode: {config.benchmark_mode}")


class TrajectoryAccumulator:
    def __init__(self, num_points: int) -> None:
        self.num_points = num_points
        self.rho_values = [[] for _ in range(num_points)]
        self.alpha_over_initial_values = [[] for _ in range(num_points)]
        self.beta_over_initial_values = [[] for _ in range(num_points)]
        self.correction_cosine_values = [[] for _ in range(num_points)]

    def update(self, rho: List[float], alpha_over_initial: List[float], beta_over_initial: List[float], correction_cosine: List[float]) -> None:
        for idx in range(self.num_points):
            self.rho_values[idx].append(float(rho[idx]))
            self.alpha_over_initial_values[idx].append(float(alpha_over_initial[idx]))
            self.beta_over_initial_values[idx].append(float(beta_over_initial[idx]))
            self.correction_cosine_values[idx].append(float(correction_cosine[idx]))

    def summarize(self) -> Dict[str, List[float]]:
        return {
            "rho_median": [_nanmedian(values) for values in self.rho_values],
            "rho_mean": [_nanmean(values) for values in self.rho_values],
            "alpha_over_initial_median": [_nanmedian(values) for values in self.alpha_over_initial_values],
            "beta_over_initial_median": [_nanmedian(values) for values in self.beta_over_initial_values],
            "correction_cosine_median": [_nanmedian(values) for values in self.correction_cosine_values],
        }


class MetricCollector:
    def __init__(self) -> None:
        self.initial_shift_norms: List[float] = []
        self.final_shift_norms: List[float] = []
        self.final_shrink_ratios: List[float] = []
        self.final_alphas: List[float] = []
        self.final_alpha_over_initial: List[float] = []
        self.final_betas: List[float] = []
        self.final_beta_over_initial: List[float] = []
        self.final_correction_cosines: List[float] = []
        self.final_attn_alpha_over_initial: List[float] = []
        self.final_mlp_alpha_over_initial: List[float] = []
        self.final_shrink_flags: List[bool] = []

    def update(self, *, initial_shift_norm: float, final_shift_norm: float, final_shrink_ratio: float, final_alpha: float, final_alpha_over_initial: float, final_beta: float, final_beta_over_initial: float, final_correction_cosine: float, final_attn_alpha_over_initial: float, final_mlp_alpha_over_initial: float) -> None:
        self.initial_shift_norms.append(float(initial_shift_norm))
        self.final_shift_norms.append(float(final_shift_norm))
        self.final_shrink_ratios.append(float(final_shrink_ratio))
        self.final_alphas.append(float(final_alpha))
        self.final_alpha_over_initial.append(float(final_alpha_over_initial))
        self.final_betas.append(float(final_beta))
        self.final_beta_over_initial.append(float(final_beta_over_initial))
        self.final_correction_cosines.append(float(final_correction_cosine))
        self.final_attn_alpha_over_initial.append(float(final_attn_alpha_over_initial))
        self.final_mlp_alpha_over_initial.append(float(final_mlp_alpha_over_initial))
        self.final_shrink_flags.append(bool(final_shrink_ratio < 1.0))

    def summary(self) -> Dict[str, float]:
        return {
            "num_tokens": len(self.initial_shift_norms),
            "average_initial_shift_norm": _nanmean(self.initial_shift_norms),
            "average_final_shift_norm": _nanmean(self.final_shift_norms),
            "average_final_shrink_ratio": _nanmean(self.final_shrink_ratios),
            "median_final_shrink_ratio": _nanmedian(self.final_shrink_ratios),
            "shrink_fraction": _fraction(self.final_shrink_flags),
            "average_final_alpha": _nanmean(self.final_alphas),
            "average_final_alpha_over_initial": _nanmean(self.final_alpha_over_initial),
            "average_final_beta": _nanmean(self.final_betas),
            "average_final_beta_over_initial": _nanmean(self.final_beta_over_initial),
            "average_final_correction_cosine": _nanmean(self.final_correction_cosines),
            "average_final_attn_alpha_over_initial": _nanmean(self.final_attn_alpha_over_initial),
            "average_final_mlp_alpha_over_initial": _nanmean(self.final_mlp_alpha_over_initial),
        }


def compute_correction_metrics_from_traces(
    native_trace: Dict[str, Any],
    mixed_trace: Dict[str, Any],
    mapping: lp.LayerMapping,
) -> Dict[str, Any]:
    hidden_delta = mixed_trace["hidden_states"] - native_trace["hidden_states"]
    attn_delta = mixed_trace["attn_additions"] - native_trace["attn_additions"]
    mlp_delta = mixed_trace["mlp_additions"] - native_trace["mlp_additions"]

    source_idx = int(mapping.dst_layer_end_idx) + 1
    initial_shift = hidden_delta[source_idx]
    initial_shift_norm = float(initial_shift.norm().item())
    if initial_shift_norm < 1e-12:
        return {
            "valid": False,
            "source_idx": source_idx,
        }

    u = initial_shift / initial_shift_norm
    num_points = hidden_delta.shape[0] - source_idx

    rho_values: List[float] = []
    alpha_values: List[float] = []
    alpha_over_initial_values: List[float] = []
    beta_values: List[float] = []
    beta_over_initial_values: List[float] = []
    correction_cosine_values: List[float] = []

    for point_offset, hidden_idx in enumerate(range(source_idx, hidden_delta.shape[0])):
        delta_h = hidden_delta[hidden_idx]
        delta_h_norm = float(delta_h.norm().item())
        rho_values.append(delta_h_norm / initial_shift_norm)
        correction = delta_h - initial_shift
        alpha = float((-torch.dot(correction, u)).item())
        orth = correction + alpha * u
        beta = float(orth.norm().item())
        correction_norm = float(correction.norm().item())
        correction_cosine = float(alpha / correction_norm) if correction_norm > 1e-12 else 0.0
        alpha_values.append(alpha)
        alpha_over_initial_values.append(alpha / initial_shift_norm)
        beta_values.append(beta)
        beta_over_initial_values.append(beta / initial_shift_norm)
        correction_cosine_values.append(correction_cosine)

    cumulative_upper_attn = attn_delta[source_idx:, :].sum(dim=0) if source_idx < attn_delta.shape[0] else torch.zeros_like(initial_shift)
    cumulative_upper_mlp = mlp_delta[source_idx:, :].sum(dim=0) if source_idx < mlp_delta.shape[0] else torch.zeros_like(initial_shift)
    final_attn_alpha_over_initial = float((-torch.dot(cumulative_upper_attn, u)).item() / initial_shift_norm)
    final_mlp_alpha_over_initial = float((-torch.dot(cumulative_upper_mlp, u)).item() / initial_shift_norm)

    return {
        "valid": True,
        "source_idx": source_idx,
        "initial_shift_norm": initial_shift_norm,
        "final_shift_norm": float(hidden_delta[-1].norm().item()),
        "final_shrink_ratio": rho_values[-1],
        "final_alpha": alpha_values[-1],
        "final_alpha_over_initial": alpha_over_initial_values[-1],
        "final_beta": beta_values[-1],
        "final_beta_over_initial": beta_over_initial_values[-1],
        "final_correction_cosine": correction_cosine_values[-1],
        "final_attn_alpha_over_initial": final_attn_alpha_over_initial,
        "final_mlp_alpha_over_initial": final_mlp_alpha_over_initial,
        "trajectory": {
            "rho": rho_values,
            "alpha_over_initial": alpha_over_initial_values,
            "beta_over_initial": beta_over_initial_values,
            "correction_cosine": correction_cosine_values,
        },
    }


def maybe_append_input_ids(model: PreTrainedModel, past_key_values: PastKeyValues, input_ids: Optional[torch.Tensor]) -> PastKeyValues:
    if input_ids is None:
        return past_key_values
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        return past_key_values
    return append_input_ids_to_past(model=model, past_key_values=past_key_values, input_ids=input_ids)


def evaluate_correction(
    config: CorrectionConfig,
    run_dir: Path,
    translator_pool: lp.LayerWindowTranslatorPool,
    model_specs: Dict[str, ModelSpec],
    layer_mappings: Dict[str, lp.LayerMapping],
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    nodes: List[Node],
    edges: List[Edge],
    active_directions: List[str],
) -> Dict[str, Any]:
    logger = setup_logger(f"correction_eval_{run_dir.name}", build_eval_log_path(run_dir))
    logger.info("Starting correction analysis")
    logger.info("experiment_config=%s", asdict(config))
    lp.log_layer_mappings(logger, nodes, model_specs, layer_mappings)

    translator_pool.eval()
    for model in models.values():
        model.eval()

    eval_config = SimpleNamespace(
        batch_size=config.eval_batch_size,
        num_workers=config.eval_num_workers,
        max_examples_per_dataset=config.eval_max_examples_per_dataset,
        seed=config.seed,
        shuffle_eval_stream=config.eval_shuffle_stream,
        shuffle_buffer=config.shuffle_buffer,
        generation_max_new_tokens=config.generation_max_new_tokens,
    )

    dataset_specs = get_eval_spec_group(config.benchmark_mode)
    dataloader_builder = build_eval_dataloader if config.benchmark_mode == "logit_qa" else build_generation_eval_dataloader
    edge_map = build_edge_map(edges)

    reference_mapping = layer_mappings[active_directions[0]]
    num_layers = model_specs[reference_mapping.reference_target_node_id].num_layers
    source_idx = int(reference_mapping.dst_layer_end_idx) + 1
    num_points = num_layers + 1 - source_idx
    fullmix_collector = MetricCollector()
    random_collector = MetricCollector()
    fullmix_trajectory_by_token = [TrajectoryAccumulator(num_points) for _ in range(config.correction_max_analysis_tokens)]
    random_trajectory_by_token = [TrajectoryAccumulator(num_points) for _ in range(config.correction_max_analysis_tokens)]

    processed_examples = 0
    for spec in dataset_specs:
        dataloader = dataloader_builder(spec=spec, eval_config=eval_config)
        for batch in dataloader:
            for example in batch:
                try:
                    prepared_inputs = build_prepared_inputs(spec=spec, example=example, tokenizer=tokenizer, models=models, config=config)
                except Exception as exc:
                    logger.warning("Skipping example due to input preparation error: %s", exc)
                    continue
                answer_token_ids = build_teacher_forcing_answer_token_ids(spec=spec, example=example, tokenizer=tokenizer)
                if answer_token_ids is None or int(answer_token_ids.shape[0]) < 1:
                    continue
                answer_token_ids = answer_token_ids[: config.correction_max_analysis_tokens].to(config.device)
                cache_input_ids = prepared_inputs["cache_input_ids"]
                question_cache_ids = prepared_inputs.get("question_cache_ids", None)
                seed_token = prepared_inputs["seed_token"]
                try:
                    past_by_node_id = {node.id: extract_past_key_values(models[node.id], cache_input_ids) for node in nodes}
                except Exception as exc:
                    logger.warning("Skipping example due to cache extraction error: %s", exc)
                    continue

                for direction in active_directions:
                    edge = edge_map[direction]
                    mapping = layer_mappings[direction]
                    translated_key, translated_value, _ = translator_pool.translate_layer_window(
                        past_key_values=past_by_node_id[edge.src_id],
                        src_name=edge.src_id,
                        dst_name=edge.dst_id,
                    )
                    native_target_past = past_by_node_id[edge.dst_id]
                    native_key_block, native_value_block = lp.extract_layer_window_blocks(
                        past_key_values=native_target_past,
                        start_layer_idx=mapping.dst_layer_idx,
                        num_layers=mapping.translated_num_layers,
                    )
                    full_mix_past = lp.replay_target_prefill_with_injected_window(
                        target_model=models[edge.dst_id],
                        prefix_input_ids=cache_input_ids,
                        target_start_layer_idx=mapping.dst_layer_idx,
                        injected_key_block=translated_key,
                        injected_value_block=translated_value,
                        dst_spec=model_specs[edge.dst_id],
                    )
                    random_past = None
                    if config.correction_include_random_control:
                        random_key_block, random_value_block = build_random_matched_window(
                            native_key_block=native_key_block,
                            native_value_block=native_value_block,
                            translated_key_block=translated_key,
                            translated_value_block=translated_value,
                        )
                        random_past = lp.replay_target_prefill_with_injected_window(
                            target_model=models[edge.dst_id],
                            prefix_input_ids=cache_input_ids,
                            target_start_layer_idx=mapping.dst_layer_idx,
                            injected_key_block=random_key_block,
                            injected_value_block=random_value_block,
                            dst_spec=model_specs[edge.dst_id],
                        )

                    target_model = models[edge.dst_id]
                    native_past = maybe_append_input_ids(target_model, native_target_past, question_cache_ids)
                    fullmix_past = maybe_append_input_ids(target_model, full_mix_past, question_cache_ids)
                    if random_past is not None:
                        random_past = maybe_append_input_ids(target_model, random_past, question_cache_ids)
                    native_past = maybe_append_input_ids(target_model, native_past, seed_token)
                    fullmix_past = maybe_append_input_ids(target_model, fullmix_past, seed_token)
                    if random_past is not None:
                        random_past = maybe_append_input_ids(target_model, random_past, seed_token)

                    for token_idx in range(int(answer_token_ids.shape[0])):
                        current_input_ids = answer_token_ids[token_idx : token_idx + 1].view(1, 1)
                        native_trace = trace_single_token_with_past(target_model, native_past, current_input_ids)
                        fullmix_trace = trace_single_token_with_past(target_model, fullmix_past, current_input_ids)
                        correction_metrics = compute_correction_metrics_from_traces(native_trace, fullmix_trace, mapping)
                        if correction_metrics.get("valid", False):
                            fullmix_collector.update(
                                initial_shift_norm=correction_metrics["initial_shift_norm"],
                                final_shift_norm=correction_metrics["final_shift_norm"],
                                final_shrink_ratio=correction_metrics["final_shrink_ratio"],
                                final_alpha=correction_metrics["final_alpha"],
                                final_alpha_over_initial=correction_metrics["final_alpha_over_initial"],
                                final_beta=correction_metrics["final_beta"],
                                final_beta_over_initial=correction_metrics["final_beta_over_initial"],
                                final_correction_cosine=correction_metrics["final_correction_cosine"],
                                final_attn_alpha_over_initial=correction_metrics["final_attn_alpha_over_initial"],
                                final_mlp_alpha_over_initial=correction_metrics["final_mlp_alpha_over_initial"],
                            )
                            fullmix_trajectory_by_token[token_idx].update(
                                rho=correction_metrics["trajectory"]["rho"],
                                alpha_over_initial=correction_metrics["trajectory"]["alpha_over_initial"],
                                beta_over_initial=correction_metrics["trajectory"]["beta_over_initial"],
                                correction_cosine=correction_metrics["trajectory"]["correction_cosine"],
                            )
                        if random_past is not None:
                            random_trace = trace_single_token_with_past(target_model, random_past, current_input_ids)
                            random_metrics = compute_correction_metrics_from_traces(native_trace, random_trace, mapping)
                            if random_metrics.get("valid", False):
                                random_collector.update(
                                    initial_shift_norm=random_metrics["initial_shift_norm"],
                                    final_shift_norm=random_metrics["final_shift_norm"],
                                    final_shrink_ratio=random_metrics["final_shrink_ratio"],
                                    final_alpha=random_metrics["final_alpha"],
                                    final_alpha_over_initial=random_metrics["final_alpha_over_initial"],
                                    final_beta=random_metrics["final_beta"],
                                    final_beta_over_initial=random_metrics["final_beta_over_initial"],
                                    final_correction_cosine=random_metrics["final_correction_cosine"],
                                    final_attn_alpha_over_initial=random_metrics["final_attn_alpha_over_initial"],
                                    final_mlp_alpha_over_initial=random_metrics["final_mlp_alpha_over_initial"],
                                )
                                random_trajectory_by_token[token_idx].update(
                                    rho=random_metrics["trajectory"]["rho"],
                                    alpha_over_initial=random_metrics["trajectory"]["alpha_over_initial"],
                                    beta_over_initial=random_metrics["trajectory"]["beta_over_initial"],
                                    correction_cosine=random_metrics["trajectory"]["correction_cosine"],
                                )
                        native_past = native_trace["past_key_values"]
                        fullmix_past = fullmix_trace["past_key_values"]
                        if random_past is not None:
                            random_past = random_trace["past_key_values"]

                processed_examples += 1
                if processed_examples % 10 == 0:
                    logger.info(
                        "processed_examples=%d | fullmix_tokens=%d | avg_final_shrink_ratio=%.4f",
                        processed_examples,
                        len(fullmix_collector.final_shrink_ratios),
                        _nanmean(fullmix_collector.final_shrink_ratios),
                    )

    fullmix_summary = fullmix_collector.summary()
    random_summary = random_collector.summary() if config.correction_include_random_control else {
        "average_final_shrink_ratio": float("nan"),
        "shrink_fraction": float("nan"),
        "average_final_correction_cosine": float("nan"),
    }

    trajectory_summary = {
        "token_trajectories": {
            f"token_{token_idx:02d}": {
                "full_mix": fullmix_trajectory_by_token[token_idx].summarize(),
                "random": random_trajectory_by_token[token_idx].summarize() if config.correction_include_random_control else {},
            }
            for token_idx in range(config.correction_max_analysis_tokens)
        },
        "source_idx": source_idx,
        "num_layers": num_layers,
    }

    logger.info(
        "[CorrectionSummary] final_shrink_ratio=%.6f | shrink_fraction=%.6f | alpha_over_initial=%.6f | beta_over_initial=%.6f | random_final_shrink_ratio=%.6f",
        fullmix_summary["average_final_shrink_ratio"],
        fullmix_summary["shrink_fraction"],
        fullmix_summary["average_final_alpha_over_initial"],
        fullmix_summary["average_final_beta_over_initial"],
        random_summary["average_final_shrink_ratio"],
    )

    return {
        "benchmark_mode": config.benchmark_mode,
        "metric_semantics": {
            "initial_shift": "first hidden-state difference immediately after the entire injected window",
            "final_shrink_ratio": "||final hidden-state difference|| divided by ||initial post-window hidden-state difference||",
            "source_idx": "post-window boundary index used as the first correction analysis point",
        },
        "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        "full_mix": fullmix_summary,
        "random_control": random_summary,
        "trajectory": trajectory_summary,
        "processed_examples": int(processed_examples),
    }


def read_summary_rows(summary_path: Path) -> List[CorrectionSummaryRow]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw_row in reader:
            post_window_boundary_idx = int(raw_row.get("post_window_boundary_idx", int(raw_row["target_layer_end_idx"]) + 1))
            num_upper_layers = int(raw_row.get("num_upper_layers", raw_row.get("remaining_correction_layers", 0)))
            rows.append(CorrectionSummaryRow(
                study_id=raw_row["study_id"],
                benchmark_mode=raw_row["benchmark_mode"],
                position_layer_idx=int(raw_row["position_layer_idx"]),
                translated_num_layers=int(raw_row["translated_num_layers"]),
                source_layer_idx=int(raw_row["source_layer_idx"]),
                source_layer_end_idx=int(raw_row["source_layer_end_idx"]),
                target_layer_idx=int(raw_row["target_layer_idx"]),
                target_layer_end_idx=int(raw_row["target_layer_end_idx"]),
                num_samples=int(raw_row["num_samples"]),
                num_tokens=int(raw_row["num_tokens"]),
                average_initial_shift_norm=float(raw_row["average_initial_shift_norm"]),
                average_final_shift_norm=float(raw_row["average_final_shift_norm"]),
                average_final_shrink_ratio=float(raw_row["average_final_shrink_ratio"]),
                median_final_shrink_ratio=float(raw_row["median_final_shrink_ratio"]),
                shrink_fraction=float(raw_row["shrink_fraction"]),
                average_final_alpha=float(raw_row["average_final_alpha"]),
                average_final_alpha_over_initial=float(raw_row["average_final_alpha_over_initial"]),
                average_final_beta=float(raw_row["average_final_beta"]),
                average_final_beta_over_initial=float(raw_row["average_final_beta_over_initial"]),
                average_final_correction_cosine=float(raw_row["average_final_correction_cosine"]),
                average_final_attn_alpha_over_initial=float(raw_row["average_final_attn_alpha_over_initial"]),
                average_final_mlp_alpha_over_initial=float(raw_row["average_final_mlp_alpha_over_initial"]),
                average_random_final_shrink_ratio=float(raw_row["average_random_final_shrink_ratio"]),
                average_random_shrink_fraction=float(raw_row["average_random_shrink_fraction"]),
                average_random_final_correction_cosine=float(raw_row["average_random_final_correction_cosine"]),
                run_dir=raw_row["run_dir"],
                post_window_boundary_idx=post_window_boundary_idx,
                num_upper_layers=num_upper_layers,
                remaining_correction_layers=num_upper_layers,
            ))
        return rows


def update_summary(config: CorrectionConfig, run_dir: Path, metrics: Dict[str, Any], layer_mappings: Dict[str, lp.LayerMapping], model_specs: Dict[str, ModelSpec]) -> Path:
    study_dir = run_dir.parent
    summary_path = build_summary_path(study_dir)
    rows = read_summary_rows(summary_path)
    mapping = next(iter(layer_mappings.values()))
    full_mix = metrics["full_mix"]
    random_control = metrics["random_control"]
    post_window_boundary_idx = int(mapping.dst_layer_end_idx) + 1
    num_upper_layers = max(0, int(model_specs[mapping.reference_target_node_id].num_layers) - post_window_boundary_idx)
    row = CorrectionSummaryRow(
        study_id=study_dir.name,
        benchmark_mode=config.benchmark_mode,
        position_layer_idx=int(config.position_layer_idx),
        translated_num_layers=int(mapping.translated_num_layers),
        source_layer_idx=int(mapping.src_layer_idx),
        source_layer_end_idx=int(mapping.src_layer_end_idx),
        target_layer_idx=int(mapping.dst_layer_idx),
        target_layer_end_idx=int(mapping.dst_layer_end_idx),
        num_samples=int(metrics.get("processed_examples", config.eval_max_examples_per_dataset)),
        num_tokens=int(full_mix["num_tokens"]),
        average_initial_shift_norm=float(full_mix["average_initial_shift_norm"]),
        average_final_shift_norm=float(full_mix["average_final_shift_norm"]),
        average_final_shrink_ratio=float(full_mix["average_final_shrink_ratio"]),
        median_final_shrink_ratio=float(full_mix["median_final_shrink_ratio"]),
        shrink_fraction=float(full_mix["shrink_fraction"]),
        average_final_alpha=float(full_mix["average_final_alpha"]),
        average_final_alpha_over_initial=float(full_mix["average_final_alpha_over_initial"]),
        average_final_beta=float(full_mix["average_final_beta"]),
        average_final_beta_over_initial=float(full_mix["average_final_beta_over_initial"]),
        average_final_correction_cosine=float(full_mix["average_final_correction_cosine"]),
        average_final_attn_alpha_over_initial=float(full_mix["average_final_attn_alpha_over_initial"]),
        average_final_mlp_alpha_over_initial=float(full_mix["average_final_mlp_alpha_over_initial"]),
        average_random_final_shrink_ratio=float(random_control.get("average_final_shrink_ratio", float("nan"))),
        average_random_shrink_fraction=float(random_control.get("shrink_fraction", float("nan"))),
        average_random_final_correction_cosine=float(random_control.get("average_final_correction_cosine", float("nan"))),
        run_dir=str(run_dir),
        post_window_boundary_idx=post_window_boundary_idx,
        num_upper_layers=num_upper_layers,
        remaining_correction_layers=num_upper_layers,
    )
    rows = [existing for existing in rows if existing.position_layer_idx != row.position_layer_idx]
    rows.append(row)
    rows.sort(key=lambda item: item.position_layer_idx)
    fieldnames = list(asdict(row).keys())
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary_row in rows:
            writer.writerow(asdict(summary_row))
    return summary_path


def plot_run_trajectories(run_dir: Path, metrics: Dict[str, Any]) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt

    source_idx = int(metrics["trajectory"]["source_idx"])
    mapping = next(iter(metrics.get("layer_mappings", {}).values()), {})
    injected_window_label = format_layer_range(int(mapping.get("dst_layer_idx", 0)), int(mapping.get("dst_layer_end_idx", 0)))
    token_00 = metrics["trajectory"]["token_trajectories"].get("token_00", {})
    full = token_00.get("full_mix", {})
    rand = token_00.get("random", {})
    if not full:
        raise ValueError("No token_00 trajectory found for plotting")
    x_values = list(range(source_idx, source_idx + len(full["rho_median"])))

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(x_values, full["rho_median"], marker="o", label="Full-Mix median ||Δh_k|| / ||s_t||")
    if rand and rand.get("rho_median"):
        ax.plot(x_values, rand["rho_median"], marker="s", label="Random control median ||Δh_k|| / ||s_t||")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Post-window layer boundary index k")
    ax.set_ylabel("Norm ratio relative to first post-window shift")
    ax.set_title(f"Correction trajectory after injected window {injected_window_label} (token 0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    norm_ratio_path = build_norm_ratio_chart_path(run_dir)
    fig.savefig(norm_ratio_path, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(x_values, full["alpha_over_initial_median"], marker="o", label="Full-Mix median α_k / ||s_t||")
    ax.plot(x_values, full["beta_over_initial_median"], marker="^", label="Full-Mix median β_k / ||s_t||")
    ax.plot(x_values, full["correction_cosine_median"], marker="d", label="Full-Mix median correction cosine")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Post-window layer boundary index k")
    ax.set_ylabel("Projected correction relative to first post-window shift")
    ax.set_title(f"Correction projection trajectory after injected window {injected_window_label} (token 0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    projection_path = build_projection_chart_path(run_dir)
    fig.savefig(projection_path, dpi=200)
    plt.close(fig)
    return norm_ratio_path, projection_path


def _annotate_ranges(ax, rows: List[CorrectionSummaryRow], y_values: List[float]) -> None:
    for row, y in zip(rows, y_values):
        ax.annotate(
            format_layer_range(row.target_layer_idx, row.target_layer_end_idx),
            (float(row.position_layer_idx), float(y)),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )


def plot_summary(summary_path: Path) -> Dict[str, Path]:
    import matplotlib.pyplot as plt
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No rows found in {summary_path}")
    rows.sort(key=lambda row: row.position_layer_idx)
    x_values = [row.position_layer_idx for row in rows]
    study_dir = summary_path.parent

    outputs: Dict[str, Path] = {}

    avg_shrink = [row.average_final_shrink_ratio for row in rows]
    median_shrink = [row.median_final_shrink_ratio for row in rows]
    random_shrink = [row.average_random_final_shrink_ratio for row in rows]
    alpha = [row.average_final_alpha_over_initial for row in rows]
    beta = [row.average_final_beta_over_initial for row in rows]
    attn_alpha = [row.average_final_attn_alpha_over_initial for row in rows]
    mlp_alpha = [row.average_final_mlp_alpha_over_initial for row in rows]
    cosine = [row.average_final_correction_cosine for row in rows]
    random_cosine = [row.average_random_final_correction_cosine for row in rows]
    initial_shift = [row.average_initial_shift_norm for row in rows]
    final_shift = [row.average_final_shift_norm for row in rows]
    structural_shrink_advantage = [rand - full for rand, full in zip(random_shrink, avg_shrink)]
    structural_cosine_advantage = [full - rand for full, rand in zip(cosine, random_cosine)]
    beta_minus_alpha = [b - a for a, b in zip(alpha, beta)]
    beta_over_alpha = [float("nan") if abs(a) < 1e-8 else b / a for a, b in zip(alpha, beta)]

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(x_values, avg_shrink, marker="o", label="Full-Mix avg final shrink ratio")
    ax.plot(x_values, median_shrink, marker="^", label="Full-Mix median final shrink ratio")
    ax.plot(x_values, random_shrink, marker="s", label="Random control avg final shrink ratio")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    _annotate_ranges(ax, rows, avg_shrink)
    ax.set_xlabel("Reference target layer start index")
    ax.set_ylabel("Final ||Δh_L|| / ||s_t||")
    ax.set_title("Final post-window shrink ratio vs injected-window start index")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outputs["shrink"] = build_summary_shrink_chart_path(study_dir)
    fig.savefig(outputs["shrink"], dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(9.5, 5.6))
    ax = fig.add_subplot(111)
    ax.plot(x_values, alpha, marker="o", label="Total α_L / ||s_t||")
    ax.plot(x_values, beta, marker="^", label="Total β_L / ||s_t||")
    ax.plot(x_values, attn_alpha, marker="s", label="Attention α contribution")
    ax.plot(x_values, mlp_alpha, marker="d", label="MLP α contribution")
    ax.plot(x_values, beta_minus_alpha, marker="x", label="β - α")
    _annotate_ranges(ax, rows, beta)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Reference target layer start index")
    ax.set_ylabel("Initial-shift-normalized magnitude")
    ax.set_title("Correction decomposition vs injected-window start index")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outputs["decomposition"] = build_summary_decomposition_chart_path(study_dir)
    fig.savefig(outputs["decomposition"], dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_subplot(111)
    ax.scatter(alpha, beta, s=[max(40.0, 12.0 * v) for v in avg_shrink], marker="o")
    finite_pairs = [(x, y) for x, y in zip(alpha, beta) if math.isfinite(x) and math.isfinite(y)]
    if finite_pairs:
        max_xy = max(max(x for x, _ in finite_pairs), max(y for _, y in finite_pairs), 1.0)
        ax.plot([0.0, max_xy], [0.0, max_xy], linestyle="--", linewidth=1, label="β = α")
    for row, x, y, ratio in zip(rows, alpha, beta, avg_shrink):
        ax.annotate(f"L{row.position_layer_idx} (ρ={ratio:.2f})", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Total α_L / ||s_t||")
    ax.set_ylabel("Total β_L / ||s_t||")
    ax.set_title("Correction phase scatter: anti-alignment vs orthogonal error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outputs["phase"] = build_summary_phase_chart_path(study_dir)
    fig.savefig(outputs["phase"], dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6.0))
    ax1 = fig.add_subplot(211)
    ax1.plot(x_values, structural_shrink_advantage, marker="o", label="Random - Full-Mix shrink ratio")
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    _annotate_ranges(ax1, rows, structural_shrink_advantage)
    ax1.set_ylabel("Positive is better")
    ax1.set_title("Structural advantage over random control (post-window correction)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.plot(x_values, structural_cosine_advantage, marker="s", label="Full-Mix - Random correction cosine")
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_xlabel("Reference target layer start index")
    ax2.set_ylabel("Positive is better")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    outputs["random"] = build_summary_random_chart_path(study_dir)
    fig.savefig(outputs["random"], dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6.0))
    ax1 = fig.add_subplot(211)
    ax1.plot(x_values, initial_shift, marker="o", label="Initial shift norm ||s_t||")
    ax1.plot(x_values, final_shift, marker="s", label="Final shift norm ||Δh_L||")
    _annotate_ranges(ax1, rows, final_shift)
    ax1.set_ylabel("Average norm")
    ax1.set_title("Initial post-window vs final hidden shift norms")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.plot(x_values, beta_over_alpha, marker="^", label="β / α")
    ax2.axhline(1.0, linestyle="--", linewidth=1)
    ax2.set_xlabel("Reference target layer start index")
    ax2.set_ylabel("Orthogonal-to-correction ratio")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    outputs["shift_norms"] = build_summary_shift_norm_chart_path(study_dir)
    fig.savefig(outputs["shift_norms"], dpi=200)
    plt.close(fig)

    return outputs


def build_layer_position_config(config: CorrectionConfig) -> lp.LayerPositionConfig:
    return lp.LayerPositionConfig(
        model_ids=config.model_ids,
        model_directions=config.model_directions,
        reference_direction=config.reference_direction,
        position_layer_idx=config.position_layer_idx,
        injection_window_size=config.injection_window_size,
        output_root=config.output_root,
        study_id=config.study_id,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
        total_tokens=config.total_tokens,
        prefix_tokens=config.prefix_tokens,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        grad_clip_norm=config.grad_clip_norm,
        log_every=config.log_every,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        translator_mlp_ratio=config.translator_mlp_ratio,
        enable_principal_rotation=config.enable_principal_rotation,
        principal_rotation_streams=config.principal_rotation_streams,
        principal_rotation_calibration_steps=config.principal_rotation_calibration_steps,
        device=config.device,
        dtype=config.dtype,
        eval_batch_size=config.eval_batch_size,
        eval_num_workers=config.eval_num_workers,
        eval_max_examples_per_dataset=config.eval_max_examples_per_dataset,
        eval_shuffle_stream=config.eval_shuffle_stream,
        benchmark_mode=config.benchmark_mode,
        generation_max_new_tokens=config.generation_max_new_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the layer-window translator as in exp/layer_position.py, then run a teacher-forced correction analysis on benchmark tasks. The correction experiment measures how much the final hidden-state shift contracts relative to the first hidden-state shift after the entire injected window, and whether upper-layer residual updates anti-align with that post-window initial shift."
    )
    parser.add_argument("--model-ids", default="gpt2,gpt2")
    parser.add_argument("--model-directions", default="A_to_B")
    parser.add_argument("--reference-direction", default=None)
    parser.add_argument("--position-layer-idx", type=int, default=None)
    parser.add_argument("--injection-window-size", type=int, default=1)
    parser.add_argument("--print-target-num-layers", action="store_true")

    parser.add_argument("--output-root", default="outputs/correction")
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
    parser.add_argument("--disable-principal-rotation", action="store_true")
    parser.add_argument("--principal-rotation-streams", choices=["k", "v", "kv"], default="kv")
    parser.add_argument("--principal-rotation-calibration-steps", type=int, default=512)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-max-examples-per-dataset", type=int, default=100)
    parser.add_argument("--eval-shuffle-stream", action="store_true")
    parser.add_argument("--benchmark-mode", choices=["logit_qa", "gen_qa"], default="logit_qa")
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--correction-max-analysis-tokens", type=int, default=4)
    parser.add_argument("--disable-random-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_target_num_layers:
        print(lp.resolve_target_num_layers(args.model_ids, args.model_directions, args.reference_direction))
        return
    config = CorrectionConfig(
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
        correction_max_analysis_tokens=args.correction_max_analysis_tokens,
        correction_include_random_control=not args.disable_random_control,
    )
    if config.position_layer_idx is None:
        raise SystemExit("--position-layer-idx is required unless --print-target-num-layers is used.")

    set_seed(config.seed)
    run_dir = build_run_output_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges, active_directions, reference_edge = lp.resolve_reference_direction_metadata(
        model_ids=config.model_ids,
        model_directions=config.model_directions,
        reference_direction=config.reference_direction,
    )
    layer_position_config = build_layer_position_config(config)
    models, tokenizer, _, _ = lp.build_models_for_experiment(layer_position_config)
    translator_pool, model_specs, layer_mappings = lp.run_train(
        config=layer_position_config,
        run_dir=run_dir,
        models=models,
        tokenizer=tokenizer,
        nodes=nodes,
        edges=edges,
        active_directions=active_directions,
        reference_edge=reference_edge,
    )
    metrics = evaluate_correction(
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

    write_json(str(build_config_path(run_dir)), asdict(config))
    write_json(str(build_metrics_path(run_dir)), metrics)
    summary_path = update_summary(config, run_dir, metrics, layer_mappings, model_specs)
    run_chart_paths = plot_run_trajectories(run_dir, metrics)
    summary_chart_paths = plot_summary(summary_path)

    print(f"Run directory: {run_dir}")
    print(f"Correction metrics: {build_metrics_path(run_dir)}")
    print(f"Summary CSV: {summary_path}")
    print(f"Run trajectory chart: {run_chart_paths[0]}")
    print(f"Run projection chart: {run_chart_paths[1]}")
    for label, path in summary_chart_paths.items():
        print(f"Summary {label} chart: {path}")


if __name__ == "__main__":
    main()
