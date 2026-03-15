#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

STUDY_ID=${STUDY_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/layer_position}
COMMON_ARGS=("$@")

INJECTION_WINDOW_SIZE=1
BENCHMARK_MODE=qa_accuracy
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
    echo "layer_position.sh sweeps positions automatically; do not pass --position-layer-idx" >&2
    exit 1
  fi
done

if ! [[ "${INJECTION_WINDOW_SIZE}" =~ ^[0-9]+$ ]] || (( INJECTION_WINDOW_SIZE < 1 )); then
  echo "--injection-window-size must be a positive integer: ${INJECTION_WINDOW_SIZE}" >&2
  exit 1
fi

case "${BENCHMARK_MODE}" in
  qa_accuracy) METRIC_NAME=accuracy ;;
  multinews_f1) METRIC_NAME=f1 ;;
  *)
    echo "Unsupported --benchmark-mode: ${BENCHMARK_MODE}" >&2
    exit 1
    ;;
esac

NUM_LAYERS=$(python exp/layer_position.py \
  --print-target-num-layers \
  --output-root "${OUTPUT_ROOT}" \
  --study-id "${STUDY_ID}" \
  --injection-window-size "${INJECTION_WINDOW_SIZE}" \
  "${COMMON_ARGS[@]}")

if ! [[ "${NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
  echo "Failed to resolve target num layers: ${NUM_LAYERS}" >&2
  exit 1
fi

echo "[LayerPosition] study_id=${STUDY_ID}"
echo "[LayerPosition] output_root=${OUTPUT_ROOT}"
echo "[LayerPosition] injection_window_size=${INJECTION_WINDOW_SIZE}"
echo "[LayerPosition] target_num_layers=${NUM_LAYERS}"

MAX_START_LAYER_IDX=$((NUM_LAYERS - INJECTION_WINDOW_SIZE))
if (( MAX_START_LAYER_IDX < 0 )); then
  echo "--injection-window-size (${INJECTION_WINDOW_SIZE}) exceeds target_num_layers (${NUM_LAYERS})" >&2
  exit 1
fi

echo "[LayerPosition] sweeping reference target layer indices 0..${MAX_START_LAYER_IDX}"
for ((layer_idx=0; layer_idx<=MAX_START_LAYER_IDX; layer_idx++)); do
  echo "[LayerPosition] ===== reference target layer idx ${layer_idx} ====="
  python exp/layer_position.py \
    --position-layer-idx "${layer_idx}" \
    --output-root "${OUTPUT_ROOT}" \
    --study-id "${STUDY_ID}" \
    --injection-window-size "${INJECTION_WINDOW_SIZE}" \
    "${COMMON_ARGS[@]}"
done

SUMMARY_ROOT="${OUTPUT_ROOT}/${STUDY_ID}"
SUMMARY_PATH="${SUMMARY_ROOT}/summary.csv"
CHART_PATH="${SUMMARY_ROOT}/layer_idx_vs_${METRIC_NAME}.png"
DRIFT_COSINE_CHART_PATH="${SUMMARY_ROOT}/drift_cosine.png"
DRIFT_L2_CHART_PATH="${SUMMARY_ROOT}/drift_l2.png"

echo "[LayerPosition] done"
echo "[LayerPosition] summary_csv=${SUMMARY_PATH}"
echo "[LayerPosition] chart_png=${CHART_PATH}"
echo "[LayerPosition] drift_cosine_chart_png=${DRIFT_COSINE_CHART_PATH}"
echo "[LayerPosition] drift_l2_chart_png=${DRIFT_L2_CHART_PATH}"
