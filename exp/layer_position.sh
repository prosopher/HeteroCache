#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

STUDY_ID=${STUDY_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/layer_position}
COMMON_ARGS=("$@")

NUM_UPPER_LAYERS=0
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "${arg}" == --num-upper-layers ]]; then
    next_index=$((i + 1))
    if (( next_index > $# )); then
      echo "Missing value after --num-upper-layers" >&2
      exit 1
    fi
    NUM_UPPER_LAYERS="${!next_index}"
  elif [[ "${arg}" == --num-upper-layers=* ]]; then
    NUM_UPPER_LAYERS="${arg#*=}"
  fi
done

if ! [[ "${NUM_UPPER_LAYERS}" =~ ^[0-9]+$ ]]; then
  echo "--num-upper-layers must be a non-negative integer: ${NUM_UPPER_LAYERS}" >&2
  exit 1
fi

NUM_LAYERS=$(python exp/layer_position.py \
  --print-target-num-layers \
  --output-root "${OUTPUT_ROOT}" \
  --study-id "${STUDY_ID}" \
  --num-upper-layers "${NUM_UPPER_LAYERS}" \
  "${COMMON_ARGS[@]}")

if ! [[ "${NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
  echo "Failed to resolve target num layers: ${NUM_LAYERS}" >&2
  exit 1
fi

UPPER_DIR=$(printf 'upper_%02d' "${NUM_UPPER_LAYERS}")

echo "[LayerPosition] study_id=${STUDY_ID}"
echo "[LayerPosition] output_root=${OUTPUT_ROOT}"
echo "[LayerPosition] translated_upper_layers=${NUM_UPPER_LAYERS}"
echo "[LayerPosition] target_num_layers=${NUM_LAYERS}"

POSITION_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for ratio in "${POSITION_RATIOS[@]}"; do
  echo "[LayerPosition] ===== position ratio ${ratio} ====="
  python exp/layer_position.py \
    --position-ratio "${ratio}" \
    --output-root "${OUTPUT_ROOT}" \
    --study-id "${STUDY_ID}" \
    --num-upper-layers "${NUM_UPPER_LAYERS}" \
    "${COMMON_ARGS[@]}"
done


SUMMARY_ROOT="${OUTPUT_ROOT}/${STUDY_ID}/${UPPER_DIR}"
SUMMARY_PATH="${SUMMARY_ROOT}/summary.csv"
DRIFT_SUMMARY_PATH="${SUMMARY_ROOT}/drift_summary.csv"
DRIFT_COSINE_CHART_PATH="${SUMMARY_ROOT}/layer_position_drift_cosine.png"
DRIFT_L2_CHART_PATH="${SUMMARY_ROOT}/layer_position_drift_l2.png"

echo "[LayerPosition] done"
echo "[LayerPosition] summary_csv=${SUMMARY_PATH}"
echo "[LayerPosition] chart_png=${SUMMARY_ROOT}/layer_position_metric.png"
echo "[LayerPosition] drift_summary_csv=${DRIFT_SUMMARY_PATH}"
echo "[LayerPosition] drift_cosine_chart_png=${DRIFT_COSINE_CHART_PATH}"
echo "[LayerPosition] drift_l2_chart_png=${DRIFT_L2_CHART_PATH}"
