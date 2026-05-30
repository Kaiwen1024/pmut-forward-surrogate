# PMUT Forward Surrogate

This repository contains the forward surrogate artifact for fast acoustic-field
prediction of binary PMUT array layouts.

```text
binary PMUT layout M -> multi-frequency acoustic magnitude field |P_tot|
```

The repository is intentionally scoped to the forward prediction problem. It
does not include reverse-design or CEM optimization code in the main workflow.

## Artifact Release

This README describes artifact release `paper-v1`. Reviewers should use the
tagged release `paper-v1` rather than the moving main branch.

## Overview

| Item | Value |
|---|---|
| Task | Binary PMUT layout to 23-channel acoustic field magnitude |
| Model | Fixed-structure 2D U-Net |
| Dataset size | 10000 full-range samples |
| Split | 7000 / 1500 / 1500 train / validation / test |
| Input tensor | `(N, 1, 50, 50)` |
| Target tensor | `(N, 23, 50, 50)` |
| Released checkpoint | `artifacts/checkpoints/pmut_forward_unet_fullrange_10k.pt` |
| Main evaluation record | `results/fullrange_10k_retrain/eval_test/summary.json` |
| Manuscript figures | `figures/paper/` |

For an initial repository sanity check, run:

```bash
conda env create -f environment.yml
conda activate pmut-forward
git lfs install
git lfs pull
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/smoke_test.py --cache_path data/fullrange_cache
```

## Main Results

The reported model predicts 23 frequency-channel field maps from one `50 x 50`
binary layout.

### Pointwise Test Metrics

| Metric | Value |
|---|---:|
| Test R2 | 0.9989006 |
| Test RMSE (Pa) | 660.7211 |
| Test MAE (Pa) | 178.3647 |
| Standardized RMSE | 0.02356 |
| Single-mask inference latency, mean | 3.122 ms |
| Batch-100 inference latency, mean | 53.285 ms |
| Batch-100 per-sample latency, mean | 0.533 ms |

The complete evaluation record is stored in:

```text
results/fullrange_10k_retrain/eval_test/summary.json
```

### Application-Level Metrics

| Metric | Mean | Median | 95th percentile |
|---|---:|---:|---:|
| ROI curve RMSE (Pa) | 575 | 443 | 1292 |
| Peak \|P\| error (Pa) | 1357 | 946 | 3871 |
| Peak frequency error (MHz) | 0.066 | 0.000 | 0.450 |
| 6-dB center frequency error (MHz) | 0.020 | 0.000 | 0.200 |
| 6-dB bandwidth error (MHz) | 0.042 | 0.000 | 0.400 |
| High-response RMSE (Pa) | 1371 | 1175 | 2847 |

These metrics are stored in:

```text
results/fullrange_10k_retrain/application_metrics.json
```

### Figures

Repository summary figures for rapid inspection are stored in:

```text
figures/artifact/test_metrics_summary.png
figures/artifact/representative_field_prediction.png
```

These repository summary figures are generated from the released evaluation
records for rapid inspection. The manuscript figures are stored separately
under `figures/paper/`.

The manuscript data-display figures associated with the current artifact
release are stored under:

```text
figures/paper/
```

This directory contains manuscript Fig. 2--Fig. 5, with PDF/PNG/SVG exports,
plotting scripts, and compact derived data or metadata. See
`figures/paper/README.md` for the figure-to-file mapping and rerun notes.

If a reviewer reruns evaluation or retraining and then reruns the plotting
scripts, the regenerated figures will reflect that local run. Re-evaluating the
released checkpoint is expected to reproduce the reported numbers up to small
numerical differences; full retraining may produce larger run-to-run
differences.

## Data and Model

### Data Protocol

```text
number of samples: 10000
input tensor:      (N, 1, 50, 50)
target tensor:     (N, 23, 50, 50)
split:             train / validation / test = 7000 / 1500 / 1500
split seed:        42
```

Each input is a `50 x 50` binary PMUT layout. The active-element count ranges
from 0 to 2500 in the released full-range dataset. Each target is a
`23 x 50 x 50` Pa-valued acoustic magnitude field `|P_tot|`. For
application-level analysis, the region of interest (ROI) is defined as the
central `10 x 10` spatial window.

In the released data loader, the raw MATLAB keys are `mask_rand` for the binary
layout and `Ptot` for the acoustic field.

The training split is used for optimization and output standardization. The
validation split is used for early stopping and model selection. The test split
was not used for training, output standardization, early stopping,
hyperparameter tuning, or model selection.

### Model Configuration

```text
base_channels = 56
depth         = 6
kernel_size   = 5
pool_type     = avg
```

The model definition is in `src/pmut_forward/model_2d_unet.py`. Training and
data utilities are in `src/pmut_forward/train_2d_unet.py`.

## Repository Layout

```text
src/pmut_forward/              Core package: data loading, model, training, evaluation
scripts/                       Reproducible command-line entry points
configs/                       Fixed model and training configuration
data/fullrange_cache/          Split full-range cache, managed by Git LFS
artifacts/checkpoints/         Released checkpoint, managed by Git LFS
results/fullrange_10k_retrain/ Training and evaluation outputs
figures/artifact/              Repository summary figures for rapid inspection
figures/paper/                 Manuscript Fig. 2--Fig. 5 exports and plotting records
```

The raw `.mat` dataset is not required if `data/fullrange_cache/` is available.
If the cache is missing or must be rebuilt, place the raw `.mat` files under
`data/raw_mat/` or pass another path with `--data_dir`.

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate pmut-forward
```

For GPU evaluation or training, use a CUDA-capable PyTorch installation. The
original run used an A100 GPU with AMP bfloat16 and channels-last tensors.

### Large Files

Large arrays and checkpoints are managed by Git LFS:

```bash
git lfs install
git lfs pull
```

The split cache files are:

```text
data/fullrange_cache/meta.json
data/fullrange_cache/x.npy
data/fullrange_cache/y_part_000.npy
data/fullrange_cache/y_part_001.npy
```

## Reproduction Workflow

### 1. Smoke Test

Run this check first after cloning:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/smoke_test.py --cache_path data/fullrange_cache
```

Expected output includes:

```text
n_samples = 10000
input_shape = [10000, 1, 50, 50]
target_shape = [10000, 23, 50, 50]
split = 7000 / 1500 / 1500
split_seed = 42
```

If the cache is unavailable and you have the raw `.mat` files, rebuild or read
from them with:

```bash
python scripts/smoke_test.py \
  --data_dir /path/to/pmut_mat_dataset \
  --cache_path data/fullrange_cache \
  --rebuild_cache
```

### 2. Evaluate the Released Checkpoint

The default evaluation script uses the repository cache and checkpoint:

```bash
bash scripts/evaluate_fullrange_test.sh
```

Useful environment-variable overrides are:

```bash
PYTHON=python
DATA_DIR=/path/to/pmut_mat_dataset
CHECKPOINT=artifacts/checkpoints/pmut_forward_unet_fullrange_10k.pt
OUT_DIR=results/fullrange_10k_retrain/eval_test
CUDA_VISIBLE_DEVICES=0
```

For example:

```bash
DATA_DIR=/path/to/pmut_mat_dataset bash scripts/evaluate_fullrange_test.sh
```

When `data/fullrange_cache/` exists, `DATA_DIR` is only a fallback for cache
rebuilding.

### 3. Optional Full Retraining

Full retraining is optional and GPU-intensive:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_fixed_fullrange_10k.py \
  --data_dir /path/to/pmut_mat_dataset \
  --cache_path data/fullrange_cache \
  --structure_json results/fullrange_10k_retrain/model_structure.json \
  --pin_memory --amp --amp_dtype bfloat16 --channels_last --device cuda \
  2>&1 | tee results/fullrange_10k_retrain/train_fixed.log
```

The fixed training configuration is also summarized in:

```text
configs/fullrange_10k_fixed_unet.json
```

### 4. Regenerate Figures and Artifact Metadata

After training or evaluation, regenerate the repository summary figures and
checksum metadata:

```bash
python scripts/make_figures.py
bash scripts/collect_final_artifacts.sh
```

`collect_final_artifacts.sh` copies a newly trained checkpoint from
`results/fullrange_10k_retrain/pmut_forward_unet_fullrange_10k.pt` when that
file exists. If no newly trained checkpoint is present, it keeps using the
released checkpoint in `artifacts/checkpoints/pmut_forward_unet_fullrange_10k.pt`
and only refreshes the artifact manifest and checksum file. This allows a
reviewer to run the metadata collection step after cloning the released
artifact without first performing full retraining.

Important outputs include:

```text
results/fullrange_10k_retrain/metrics.json
results/fullrange_10k_retrain/eval_test/summary.json
figures/artifact/test_metrics_summary.png
figures/artifact/representative_field_prediction.png
figures/paper/
artifacts/checkpoints/pmut_forward_unet_fullrange_10k.pt
artifact_manifest.json
sha256sums.txt
```

The full test prediction arrays generated during evaluation
(`results/fullrange_10k_retrain/eval_test/y_test_*.npy`) are ignored by
default. They are needed only for rerunning some manuscript plotting scripts
from raw prediction arrays. Rerunning those plotting scripts after generating
new evaluation outputs will overwrite the corresponding local figure exports
with figures from that local run.

## Runtime Notes

The smoke test only loads the cached arrays and verifies tensor shapes, finite
values, and the deterministic train/validation/test split. Full test-set
evaluation is more expensive because it runs inference over the held-out test
split and may write prediction arrays to `results/fullrange_10k_retrain/eval_test/`.
Full retraining is GPU-intensive and is mainly intended for complete artifact
verification or follow-up experiments.

Exact runtime depends on GPU type, CPU memory, and storage bandwidth. Treat the
released `summary.json`, `metrics.json`, training curves, and checksum files as
the primary record of the reported run, and use the scripts to reproduce or
audit these artifacts when sufficient hardware is available.

## File Integrity

Use the checksum file to verify released artifacts:

```bash
sha256sum -c sha256sums.txt
```

This checks that the large cache files, checkpoint, metrics, and figures match
the released artifact manifest. Large regenerated prediction arrays under
`results/fullrange_10k_retrain/eval_test/` are ignored by default and are not
listed in the released checksum file.
