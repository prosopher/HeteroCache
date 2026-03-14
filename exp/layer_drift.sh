#!/usr/bin/env bash
set -euo pipefail

study_id="${study_id:-${1:-}}"
if [[ -z "${study_id}" ]]; then
  echo "Usage: study_id=<existing_study_id> bash exp/layer_drift.sh [extra python args...]" >&2
  echo "   or: bash exp/layer_drift.sh <existing_study_id> [extra python args...]" >&2
  exit 1
fi

if [[ $# -ge 1 && "$1" == "$study_id" ]]; then
  shift
fi

python exp/layer_drift.py \
  --study-id "$study_id" \
  "$@"
