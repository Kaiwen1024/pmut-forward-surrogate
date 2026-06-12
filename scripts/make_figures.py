#!/usr/bin/env python
"""Generate lightweight forward-prediction figures for artifact inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_loss_curves(results_dir: Path, figures_dir: Path) -> None:
    metrics = load_metrics(results_dir / "metrics.json")
    values = metrics.get("test_metrics_denorm", {})
    labels = ["R2", "RMSE", "MAE"]
    nums = [
        float(values.get("r2", 0.0)),
        float(values.get("rmse", 0.0)),
        float(values.get("mae", 0.0)),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(labels, nums, color=["#2f6f9f", "#d1495b", "#edae49"])
    ax.set_title("Test Metrics")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "test_metrics_summary.png", dpi=220)
    plt.close(fig)


def plot_prediction_example(eval_dir: Path, figures_dir: Path, sample_index: int = 0, channel: int = 11) -> None:
    true_path = eval_dir / "y_test_true.npy"
    pred_path = eval_dir / "y_test_pred.npy"
    if not true_path.exists() or not pred_path.exists():
        missing = [str(path) for path in (true_path, pred_path) if not path.exists()]
        print(
            "[Figures] skipping representative_field_prediction.png; "
            f"missing prediction file(s): {', '.join(missing)}"
        )
        return
    y_true = np.load(true_path, mmap_mode="r")
    y_pred = np.load(pred_path, mmap_mode="r")
    idx = min(sample_index, y_true.shape[0] - 1)
    ch = min(channel, y_true.shape[1] - 1)
    err = np.abs(y_pred[idx, ch] - y_true[idx, ch])
    vmax = max(float(np.max(y_true[idx, ch])), float(np.max(y_pred[idx, ch])), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), constrained_layout=True)
    for ax, arr, title in zip(axes, [y_true[idx, ch], y_pred[idx, ch], err], ["Ground Truth", "Prediction", "Absolute Error"]):
        im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=None if title == "Absolute Error" else vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(figures_dir / "representative_field_prediction.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate PMUT forward-prediction figures.")
    p.add_argument("--results_dir", type=Path, default=Path("results/fullrange_10k_retrain"))
    p.add_argument("--eval_dir", type=Path, default=Path("results/fullrange_10k_retrain/eval_test"))
    p.add_argument("--figures_dir", type=Path, default=Path("figures") / "artifact")
    args = p.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(args.results_dir, args.figures_dir)
    plot_prediction_example(args.eval_dir, args.figures_dir)


if __name__ == "__main__":
    main()
