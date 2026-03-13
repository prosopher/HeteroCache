#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

STUDY_ID=${STUDY_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/layer_position}
COMMON_ARGS=("$@")

NUM_LAYERS=$(python exp/layer_position.py \
  --print-target-num-layers \
  --output-root "${OUTPUT_ROOT}" \
  --study-id "${STUDY_ID}" \
  "${COMMON_ARGS[@]}")

if ! [[ "${NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
  echo "Failed to resolve target num layers: ${NUM_LAYERS}" >&2
  exit 1
fi

echo "[LayerPosition] study_id=${STUDY_ID}"
echo "[LayerPosition] output_root=${OUTPUT_ROOT}"
echo "[LayerPosition] target_num_layers=${NUM_LAYERS}"

for ((layer_idx=0; layer_idx<NUM_LAYERS; layer_idx++)); do
  echo "[LayerPosition] ===== target layer ${layer_idx}/${NUM_LAYERS} ====="
  python exp/layer_position.py \
    --layer-to-translate "${layer_idx}" \
    --output-root "${OUTPUT_ROOT}" \
    --study-id "${STUDY_ID}" \
    "${COMMON_ARGS[@]}"
done

CHART_PATH=$(python exp/layer_position.py \
  --plot-study-summary \
  --output-root "${OUTPUT_ROOT}" \
  --study-id "${STUDY_ID}" \
  "${COMMON_ARGS[@]}")

SUMMARY_PATH="${OUTPUT_ROOT}/${STUDY_ID}/summary.csv"

echo "[LayerPosition] done"
echo "[LayerPosition] summary_csv=${SUMMARY_PATH}"
echo "[LayerPosition] chart_png=${CHART_PATH}"
