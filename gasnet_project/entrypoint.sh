#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=""
OUTPUT_DIR="/workspace/output"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: entrypoint.sh --data-root /path/to/data [--output-dir /path/to/output] [train.py args...]"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="${DATA_ROOT:-/data}"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi

VRU_DIR="$DATA_ROOT/VRU"
PIC_DIR="$VRU_DIR/Pic"
SPLIT_DIR="$VRU_DIR/train_test_split"

if [[ ! -d "$PIC_DIR" ]] || [[ ! -d "$SPLIT_DIR" ]]; then
  echo "Invalid VRU layout. Expecting VRU/Pic and VRU/train_test_split under $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export DATA_ROOT
export OUTPUT_DIR

python - <<'PY'
import os
import torch

print("=== Environment ===")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("cudnn:", torch.backends.cudnn.version())
print("data root:", os.environ.get("DATA_ROOT", ""))
print("output dir:", os.environ.get("OUTPUT_DIR", ""))
PY

python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ.get("DATA_ROOT", "/data"))
train_list = root / "VRU" / "train_test_split" / "train_list.txt"
small = root / "VRU" / "train_test_split" / "test_list_1200.txt"
medium = root / "VRU" / "train_test_split" / "test_list_2400.txt"
big = root / "VRU" / "train_test_split" / "test_list_8000.txt"

print("=== Dataset Check ===")
for p in (train_list, small, medium, big):
    if not p.exists():
        raise SystemExit(f"Missing split file: {p}")
    with p.open("r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"{p.name}: {count} lines")
PY

TORCH_HOME_DIR="${TORCH_HOME:-/opt/torch_cache}"
WEIGHT_FILE="$TORCH_HOME_DIR/hub/checkpoints/resnet50-11ad3fa6.pth"
HAS_NO_PRETRAINED=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--no-pretrained" ]]; then
    HAS_NO_PRETRAINED=1
    break
  fi
done

if [[ ! -f "$WEIGHT_FILE" && $HAS_NO_PRETRAINED -eq 0 ]]; then
  echo "ResNet-50 weights not found at $WEIGHT_FILE; falling back to --no-pretrained." >&2
  EXTRA_ARGS+=("--no-pretrained")
fi

HAS_NO_COMPILE=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--no-compile" ]]; then
    HAS_NO_COMPILE=1
    break
  fi
done

if [[ $HAS_NO_COMPILE -eq 0 ]]; then
  if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    echo "No compiler found; falling back to --no-compile." >&2
    EXTRA_ARGS+=("--no-compile")
  fi
fi

python /workspace/gasnet_project/train.py \
  --data-root "$DATA_ROOT" \
  --save-path "$OUTPUT_DIR/gasnet_vru.pth" \
  --run-eval \
  "${EXTRA_ARGS[@]}"
