#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ASSET_ROOT="${1:?asset_root required}"
PLAN_JSON="${2:?plan_json required}"
OUT_DIR="${3:-validation_outputs/plan_export}"

mkdir -p "${OUT_DIR}"
cmd=(python tools/run_plan.py
  --asset_root "${ASSET_ROOT}" \
  --plan_json "${PLAN_JSON}" \
  --out "${OUT_DIR}" \
  --trajectory_npz "${OUT_DIR}/trajectory.npz" \
  --trajectory_jsonl "${OUT_DIR}/trajectory.jsonl" \
  --export_animated_glb \
  --use_glb_scene auto)

"${cmd[@]}"
