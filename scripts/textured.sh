#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_POSE_MODE="${INITIAL_POSE_MODE:-zeros}"  # Choices: zeros | prismatic_lower | lower
JOBS="${JOBS:-6}"
FRAMES_PER_JOINT="${FRAMES_PER_JOINT:-36}"
MAX_PREVIEW_JOINTS="${MAX_PREVIEW_JOINTS:-8}"
CANONICALIZE_URDF_NAMES="${CANONICALIZE_URDF_NAMES:-1}"
CANONICALIZE_JOINT_NAMES="${CANONICALIZE_JOINT_NAMES:-1}"
DEFAULT_PARTICULATE_PY="/home/chunyu/miniconda3/envs/casual_agent/bin/python"
if [[ -z "${PY_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PARTICULATE_PY}" ]]; then
    PY_BIN="${DEFAULT_PARTICULATE_PY}"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PY_BIN="${CONDA_PREFIX}/bin/python"
  else
    PY_BIN="python"
  fi
fi
DATA_ROOT="${ROOT}/data/causal_data"

usage() {
  cat <<'EOF'
Usage:
  scripts/textured.sh <asset_name>
  scripts/textured.sh --all
  scripts/textured.sh --root <dataset_root> <asset_name>
  scripts/textured.sh --root <dataset_root> --all

Env:
  PY_BIN=<python_executable>            Default: python
  INITIAL_POSE_MODE=zeros|prismatic_lower|lower   Default: zeros
  JOBS=<int>                           Default: 6
  FRAMES_PER_JOINT=<int>               Default: 36
  MAX_PREVIEW_JOINTS=<int>             Default: 8
  CANONICALIZE_URDF_NAMES=0|1          Default: 1
  CANONICALIZE_JOINT_NAMES=0|1         Default: 1
EOF
}

build_one_asset() {
  local dataset_root="$1"
  local asset_name="$2"
  local asset_root="${dataset_root}/${asset_name}"
  local out_glb="${asset_root}/animated_textured_${asset_name}.glb"
  if [[ ! -d "${asset_root}" ]]; then
    echo "[WARN] Skip ${asset_name}: asset directory not found (${asset_root})"
    return 0
  fi
  if ! find "${asset_root}" -type f -name "*.urdf" -print -quit | grep -q .; then
    echo "[WARN] Skip ${asset_name}: no URDF found"
    return 0
  fi
  if [[ "${CANONICALIZE_URDF_NAMES}" != "0" ]]; then
    local canonicalize_args=(--asset_root "${asset_root}")
    if [[ "${CANONICALIZE_JOINT_NAMES}" == "0" ]]; then
      canonicalize_args+=(--keep_joint_names)
    fi
    echo "[INFO] Canonicalizing URDF names for ${asset_name}"
    "${PY_BIN}" "${ROOT}/tools/canonicalize_urdf_names.py" "${canonicalize_args[@]}" >/dev/null
  fi
  echo "[INFO] Building ${asset_name} -> ${out_glb}"
  echo "[INFO] Using PY_BIN=${PY_BIN} INITIAL_POSE_MODE=${INITIAL_POSE_MODE} DATASET_ROOT=${dataset_root} CANONICALIZE_URDF_NAMES=${CANONICALIZE_URDF_NAMES}"
  "${PY_BIN}" "${ROOT}/tools/build_textured_animated_glb.py" \
    --asset_root "${asset_root}" \
    --out_glb "${out_glb}" \
    --build_mode urdf_preview \
    --initial_pose_mode "${INITIAL_POSE_MODE}" \
    --frames_per_joint "${FRAMES_PER_JOINT}" \
    --max_preview_joints "${MAX_PREVIEW_JOINTS}" \
    --make_symlink
}

build_all_assets() {
  local dataset_root="$1"
  if [[ ! -d "${dataset_root}" ]]; then
    echo "[WARN] Dataset root not found: ${dataset_root}"
    return 0
  fi
  echo "[INFO] Batch rebuilding canonical textured meshes under ${dataset_root}"
  local canonicalize_args=()
  if [[ "${CANONICALIZE_URDF_NAMES}" != "0" ]]; then
    canonicalize_args+=(--canonicalize_urdf_names)
    if [[ "${CANONICALIZE_JOINT_NAMES}" == "0" ]]; then
      canonicalize_args+=(--keep_joint_names)
    fi
  fi
  echo "[INFO] Using PY_BIN=${PY_BIN} INITIAL_POSE_MODE=${INITIAL_POSE_MODE} JOBS=${JOBS} FRAMES_PER_JOINT=${FRAMES_PER_JOINT} MAX_PREVIEW_JOINTS=${MAX_PREVIEW_JOINTS} CANONICALIZE_URDF_NAMES=${CANONICALIZE_URDF_NAMES}"
  "${PY_BIN}" "${ROOT}/tools/build_partnet_mobility_textured_glbs.py" \
    --root "${dataset_root}" \
    --overwrite \
    --jobs "${JOBS}" \
    --frames_per_joint "${FRAMES_PER_JOINT}" \
    --initial_pose_mode "${INITIAL_POSE_MODE}" \
    --max_preview_joints "${MAX_PREVIEW_JOINTS}" \
    "${canonicalize_args[@]}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

DATASET_ROOT="${DATA_ROOT}"
MODE=""
ASSET_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --root requires a path"
        usage
        exit 2
      fi
      DATASET_ROOT="$2"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ -n "${ASSET_NAME}" ]]; then
        echo "[ERROR] Unexpected extra argument: $1"
        usage
        exit 2
      fi
      ASSET_NAME="$1"
      shift
      ;;
  esac
done

if [[ "${MODE}" == "all" ]]; then
  build_all_assets "${DATASET_ROOT}"
  exit 0
fi

ASSET_NAME="${ASSET_NAME:-drawer1}"
build_one_asset "${DATASET_ROOT}" "${ASSET_NAME}"
