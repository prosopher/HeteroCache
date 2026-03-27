#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

STUDY_ID=${STUDY_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/correction}
COMMON_ARGS=("$@")

INJECTION_WINDOW_SIZE=1
BENCHMARK_MODE=logit_qa
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "${arg}" == --injection-window-size ]]; then
    next_index=$((i + 1))
    if (( next_index > $# )); then
      echo "Missing value after --injection-window-size" >&2
      exit 1
    fi
    INJECTION_WINDOW_SIZE="${!next_index}"
  elif [[ "${arg}" == --injection-window-size=* ]]; then
    INJECTION_WINDOW_SIZE="${arg#*=}"
  elif [[ "${arg}" == --benchmark-mode ]]; then
    next_index=$((i + 1))
    if (( next_index > $# )); then
      echo "Missing value after --benchmark-mode" >&2
      exit 1
    fi
    BENCHMARK_MODE="${!next_index}"
  elif [[ "${arg}" == --benchmark-mode=* ]]; then
    BENCHMARK_MODE="${arg#*=}"
  elif [[ "${arg}" == --position-layer-idx || "${arg}" == --position-layer-idx=* ]]; then
    echo "correction.sh sweeps positions automatically; do not pass --position-layer-idx" >&2
    exit 1
  fi
done

if ! [[ "${INJECTION_WINDOW_SIZE}" =~ ^[0-9]+$ ]] || (( INJECTION_WINDOW_SIZE < 1 )); then
  echo "correction.sh requires --injection-window-size to be a positive integer" >&2
  exit 1
fi

case "${BENCHMARK_MODE}" in
  logit_qa|gen_qa) ;;
  *)
    echo "Unsupported --benchmark-mode: ${BENCHMARK_MODE}" >&2
    exit 1
    ;;
esac

NUM_LAYERS=$(python exp/correction.py \
  --print-target-num-layers \
  --output-root "${OUTPUT_ROOT}" \
  --study-id "${STUDY_ID}" \
  --injection-window-size "${INJECTION_WINDOW_SIZE}" \
  "${COMMON_ARGS[@]}")

if ! [[ "${NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
  echo "Failed to resolve target num layers: ${NUM_LAYERS}" >&2
  exit 1
fi

echo "[Correction] study_id=${STUDY_ID}"
echo "[Correction] output_root=${OUTPUT_ROOT}"
echo "[Correction] benchmark_mode=${BENCHMARK_MODE}"
echo "[Correction] injection_window_size=${INJECTION_WINDOW_SIZE}"
echo "[Correction] target_num_layers=${NUM_LAYERS}"

MAX_START_LAYER_IDX=$((NUM_LAYERS - INJECTION_WINDOW_SIZE))
echo "[Correction] sweeping reference target layer indices 0..${MAX_START_LAYER_IDX}"
for ((layer_idx=0; layer_idx<=MAX_START_LAYER_IDX; layer_idx++)); do
  echo "[Correction] ===== reference target layer idx ${layer_idx} ====="
  python exp/correction.py \
    --position-layer-idx "${layer_idx}" \
    --output-root "${OUTPUT_ROOT}" \
    --study-id "${STUDY_ID}" \
    --injection-window-size "${INJECTION_WINDOW_SIZE}" \
    "${COMMON_ARGS[@]}"
done

SUMMARY_ROOT="${OUTPUT_ROOT}/${STUDY_ID}"
echo "[Correction] done"
echo "[Correction] summary_csv=${SUMMARY_ROOT}/correction_summary.csv"
echo "[Correction] shrink_ratio_chart_png=${SUMMARY_ROOT}/layer_idx_vs_final_shrink_ratio.png"
echo "[Correction] decomposition_chart_png=${SUMMARY_ROOT}/layer_idx_vs_correction_decomposition.png"
echo "[Correction] structural_advantage_chart_png=${SUMMARY_ROOT}/layer_idx_vs_structural_advantage.png"
echo "[Correction] phase_scatter_chart_png=${SUMMARY_ROOT}/correction_phase_scatter.png"
echo "[Correction] shift_norms_chart_png=${SUMMARY_ROOT}/layer_idx_vs_shift_norms.png"
