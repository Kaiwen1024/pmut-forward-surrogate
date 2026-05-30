#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/raw_mat}"
CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/pmut_forward_unet_fullrange_10k.pt}"
OUT_DIR="${OUT_DIR:-results/fullrange_10k_retrain/eval_test}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

cd "${REPO_DIR}"

"${PYTHON}" -u -m pmut_forward.eval_2d_unet \
  --checkpoint_path "${CHECKPOINT}" \
  --data_dir "${DATA_DIR}" \
  --cache_path data/fullrange_cache \
  --min_active_count 0 \
  --eval_split test \
  --save_predictions \
  --device cuda \
  --single_gpu \
  --amp \
  --amp_dtype bfloat16 \
  --channels_last \
  --batch_size 16 \
  --num_workers 0 \
  --pin_memory \
  --out_dir "${OUT_DIR}"
