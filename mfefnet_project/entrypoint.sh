#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/workspace/data"
OUTPUT_DIR="/workspace/output"
EVAL_SPLITS="val,test"
PREPARE_DATA=1
H100_PRESET=1
SKIP_EVAL=0
EVAL_BEST=1
AMP_DTYPE="auto"
CHANNELS_LAST=0
COMPILE=0
COMPILE_MODE="max-autotune"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --eval-splits)
      EVAL_SPLITS="$2"; shift 2 ;;
    --prepare-data)
      PREPARE_DATA=1; shift ;;
    --no-prepare-data)
      PREPARE_DATA=0; shift ;;
    --h100-preset)
      H100_PRESET=1; shift ;;
    --no-h100-preset)
      H100_PRESET=0; shift ;;
    --skip-eval)
      SKIP_EVAL=1; shift ;;
    --eval-best)
      EVAL_BEST=1; shift ;;
    --no-eval-best)
      EVAL_BEST=0; shift ;;
    --amp-dtype)
      AMP_DTYPE="$2"; shift 2 ;;
    --channels-last)
      CHANNELS_LAST=1; shift ;;
    --compile)
      COMPILE=1; shift ;;
    --compile-mode)
      COMPILE_MODE="$2"; shift 2 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ARGS=(
  "--data-root" "$DATA_ROOT"
  "--output-dir" "$OUTPUT_DIR"
  "--eval-splits" "$EVAL_SPLITS"
  "--amp-dtype" "$AMP_DTYPE"
  "--compile-mode" "$COMPILE_MODE"
)

if [[ $PREPARE_DATA -eq 1 ]]; then
  ARGS+=("--prepare-data")
fi

if [[ $H100_PRESET -eq 1 ]]; then
  ARGS+=("--h100-preset")
fi

if [[ $SKIP_EVAL -eq 1 ]]; then
  ARGS+=("--skip-eval")
fi

if [[ $EVAL_BEST -eq 0 ]]; then
  ARGS+=("--no-eval-best")
fi

if [[ $CHANNELS_LAST -eq 1 ]]; then
  ARGS+=("--channels-last")
fi

if [[ $COMPILE -eq 1 ]]; then
  ARGS+=("--compile")
fi

exec python train.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
