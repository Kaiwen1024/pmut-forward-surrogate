#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/fullrange_10k_retrain}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/artifacts/checkpoints}"
RELEASED_CHECKPOINT="${CHECKPOINT_DIR}/pmut_forward_unet_fullrange_10k.pt"
TRAINED_CHECKPOINT="${RESULTS_DIR}/pmut_forward_unet_fullrange_10k.pt"
PYTHON="${PYTHON:-python}"

mkdir -p "${CHECKPOINT_DIR}"

if [[ -f "${TRAINED_CHECKPOINT}" ]]; then
  cp "${TRAINED_CHECKPOINT}" "${RELEASED_CHECKPOINT}"
  echo "[Artifacts] copied trained checkpoint to ${RELEASED_CHECKPOINT}"
elif [[ -f "${RELEASED_CHECKPOINT}" ]]; then
  echo "[Artifacts] using existing released checkpoint: ${RELEASED_CHECKPOINT}"
else
  echo "[Artifacts] missing checkpoint." >&2
  echo "[Artifacts] Expected either:" >&2
  echo "  - ${TRAINED_CHECKPOINT} after full retraining, or" >&2
  echo "  - ${RELEASED_CHECKPOINT} from the released artifact." >&2
  exit 1
fi

cd "${REPO_DIR}"
"${PYTHON}" scripts/export_artifact_manifest.py
