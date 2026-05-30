#!/usr/bin/env python3
"""Draw the training-curve figure from the latest saved epoch history."""

from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "export"

RESULT_DIR = Path("./results/fullrange_10k_retrain")
HISTORY_CSV = RESULT_DIR / "training_history.csv"
HISTORY_JSON = RESULT_DIR / "training_history.json"
METRICS_JSON = RESULT_DIR / "metrics.json"
RUN_METADATA_JSON = RESULT_DIR / "run_metadata.json"

FIGURE_CSV = DATA_DIR / "training_curves.csv"
SOURCE_NOTE = DATA_DIR / "DATA_SOURCE.md"


def load_history() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = list(csv.DictReader(HISTORY_CSV.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {HISTORY_CSV}")
    epochs = np.asarray([int(r["epoch"]) for r in rows], dtype=np.int64)
    train_loss = np.asarray([float(r["train_loss"]) for r in rows], dtype=np.float64)
    val_loss = np.asarray([float(r["val_loss"]) for r in rows], dtype=np.float64)
    val_rmse = np.asarray([float(r["val_rmse"]) for r in rows], dtype=np.float64)
    return epochs, train_loss, val_loss, val_rmse


def write_figure_csv(epochs: np.ndarray, train_loss: np.ndarray, val_loss: np.ndarray, val_rmse: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FIGURE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_rmse"])
        for row in zip(epochs, train_loss, val_loss, val_rmse):
            writer.writerow([int(row[0]), f"{row[1]:.10g}", f"{row[2]:.10g}", f"{row[3]:.10g}"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        clip_on=False,
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="major", length=2.4, width=0.55, pad=1.6)
    if grid_axis in {"both", "x"}:
        ax.grid(True, axis="x", color="#DEE5EE", linewidth=0.42, alpha=0.78)
    if grid_axis in {"both", "y"}:
        ax.grid(True, axis="y", color="#E8ECF2", linewidth=0.42, alpha=0.78)
    ax.set_axisbelow(True)


def draw() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    run_metadata = json.loads(RUN_METADATA_JSON.read_text(encoding="utf-8")) if RUN_METADATA_JSON.exists() else {}
    epochs, train_loss, val_loss, val_rmse = load_history()
    write_figure_csv(epochs, train_loss, val_loss, val_rmse)

    best_val_mse = float(metrics["best_val_mse_norm_final"])
    best_epoch = int(metrics.get("best_val_epoch_final", epochs[np.argmin(val_loss)]))
    best_val_rmse = math.sqrt(best_val_mse)

    if best_epoch not in set(int(v) for v in epochs):
        best_epoch = int(epochs[np.argmin(val_loss)])

    SOURCE_NOTE.write_text(
        "\n".join(
            [
                "# Data Source",
                "",
                "This figure is redrawn from the latest retraining run with saved epoch-wise history.",
                "",
                f"- Source history CSV: `{HISTORY_CSV}`",
                f"- Source history JSON: `{HISTORY_JSON}`",
                f"- Source metrics file: `{METRICS_JSON}`",
                f"- Source run metadata: `{RUN_METADATA_JSON}`",
                f"- Figure CSV: `{FIGURE_CSV}`",
                f"- Best normalized validation MSE: {best_val_mse:.10g}",
                f"- Best validation epoch: {best_epoch}",
                f"- Corresponding normalized validation RMSE: {best_val_rmse:.10g}",
                f"- Epochs recorded: {int(epochs.min())}--{int(epochs.max())} ({len(epochs)} epochs)",
                f"- Run created UTC: {run_metadata.get('created_utc', 'unknown')}",
                "- Note: no curve values were recovered from raster images for this figure.",
            ]
        ),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Tinos",
            "mathtext.it": "Tinos:italic",
            "mathtext.bf": "Tinos:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.8,
            "axes.labelsize": 6.9,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.6,
            "legend.frameon": False,
        }
    )

    blue = "#5D91BF"
    teal = "#7CB19F"
    orange = "#D58B5C"
    dark = "#2F3D49"

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.28), dpi=240)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.25, wspace=0.27)

    ax = axes[0]
    ax.plot(epochs, train_loss, color=blue, linewidth=1.05, label="Training loss")
    ax.plot(epochs, val_loss, color=teal, linewidth=1.05, label="Validation loss")
    ax.scatter([best_epoch], [best_val_mse], s=12, color=orange, zorder=4)
    ax.annotate(
        f"Best val. MSE = {best_val_mse:.6f}",
        xy=(best_epoch, best_val_mse),
        xytext=(-74, 21),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.45, "color": "#5A6470"},
        fontsize=5.2,
        color="#38434D",
    )
    ax.set_title("Training and validation loss", pad=2.5)
    panel_label(ax, "a")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.set_yscale("log")
    ax.set_xlim(0, max(130, int(epochs.max()) + 3))
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_ylim(3.5e-4, max(float(np.max(train_loss[:5])), float(np.max(val_loss[:5]))) * 1.55)
    ax.legend(fontsize=5.7, loc="upper right", handlelength=1.6)
    clean_axis(ax, "y")

    ax = axes[1]
    ax.plot(epochs, val_rmse, color=dark, linewidth=1.1, label="Validation RMSE")
    ax.fill_between(epochs, val_rmse, 0.0215, color="#AFC6D9", alpha=0.28, linewidth=0)
    ax.scatter([best_epoch], [best_val_rmse], s=13, color=orange, zorder=4)
    ax.annotate(
        f"Best val. RMSE = {best_val_rmse:.5f}",
        xy=(best_epoch, best_val_rmse),
        xytext=(-76, 26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.45, "color": "#5A6470"},
        fontsize=5.2,
        color="#38434D",
    )
    ax.set_title("Validation RMSE", pad=2.5)
    panel_label(ax, "b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized RMSE")
    ax.set_xlim(0, max(130, int(epochs.max()) + 3))
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_ylim(0.0215, max(float(np.max(val_rmse[:5])) * 1.08, 0.112))
    ax.legend(fontsize=5.7, loc="upper right", handlelength=1.6)
    clean_axis(ax, "y")

    for ext in ["pdf", "svg", "png"]:
        path = EXPORT_DIR / f"fig2_training_curves_wide.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(path, **kwargs)
    plt.close(fig)

def draw_single_column() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    epochs, train_loss, val_loss, val_rmse = load_history()

    best_val_mse = float(metrics["best_val_mse_norm_final"])
    best_epoch = int(metrics.get("best_val_epoch_final", epochs[np.argmin(val_loss)]))
    best_val_rmse = math.sqrt(best_val_mse)

    if best_epoch not in set(int(v) for v in epochs):
        best_epoch = int(epochs[np.argmin(val_loss)])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Tinos",
            "mathtext.it": "Tinos:italic",
            "mathtext.bf": "Tinos:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.8,
            "axes.labelsize": 6.9,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.6,
            "legend.frameon": False,
        }
    )

    blue = "#5D91BF"
    teal = "#7CB19F"
    orange = "#D58B5C"
    dark = "#2F3D49"

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.05), dpi=240)
    fig.subplots_adjust(left=0.18, right=0.975, top=0.94, bottom=0.12, hspace=0.52)

    ax = axes[0]
    ax.plot(epochs, train_loss, color=blue, linewidth=1.05, label="Training loss")
    ax.plot(epochs, val_loss, color=teal, linewidth=1.05, label="Validation loss")
    ax.scatter([best_epoch], [best_val_mse], s=12, color=orange, zorder=4)
    ax.annotate(
        f"Best val. MSE = {best_val_mse:.6f}",
        xy=(best_epoch, best_val_mse),
        xytext=(-78, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.45, "color": "#5A6470"},
        fontsize=5.1,
        color="#38434D",
    )
    ax.set_title("Training and validation loss", pad=2.5)
    panel_label(ax, "a")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.set_yscale("log")
    ax.set_xlim(0, max(130, int(epochs.max()) + 3))
    ax.set_xticks([0, 40, 80, 120])
    ax.set_ylim(3.5e-4, max(float(np.max(train_loss[:5])), float(np.max(val_loss[:5]))) * 1.55)
    ax.legend(fontsize=5.5, loc="upper right", handlelength=1.45)
    clean_axis(ax, "y")

    ax = axes[1]
    ax.plot(epochs, val_rmse, color=dark, linewidth=1.1, label="Validation RMSE")
    ax.fill_between(epochs, val_rmse, 0.0215, color="#AFC6D9", alpha=0.28, linewidth=0)
    ax.scatter([best_epoch], [best_val_rmse], s=13, color=orange, zorder=4)
    ax.annotate(
        f"Best val. RMSE = {best_val_rmse:.5f}",
        xy=(best_epoch, best_val_rmse),
        xytext=(-78, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.45, "color": "#5A6470"},
        fontsize=5.1,
        color="#38434D",
    )
    ax.set_title("Validation RMSE", pad=2.5)
    panel_label(ax, "b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized RMSE")
    ax.set_xlim(0, max(130, int(epochs.max()) + 3))
    ax.set_xticks([0, 40, 80, 120])
    ax.set_ylim(0.0215, max(float(np.max(val_rmse[:5])) * 1.08, 0.112))
    ax.legend(fontsize=5.5, loc="upper right", handlelength=1.45)
    clean_axis(ax, "y")

    for ext in ["pdf", "svg", "png"]:
        path = EXPORT_DIR / f"fig2_training_curves.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 500
        fig.savefig(path, **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true", help="Write the wider exploratory export.")
    parser.add_argument("--singlecol-only", action="store_true", help="Alias for the manuscript export.")
    args = parser.parse_args()

    missing = [path for path in [HISTORY_CSV, METRICS_JSON] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source output(s): " + ", ".join(str(path) for path in missing))
    if args.wide:
        draw()
    else:
        draw_single_column()


if __name__ == "__main__":
    main()
