#!/usr/bin/env python
"""Train the fixed-structure PMUT forward 2D U-Net on the 10k full-range set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pmut_forward import train_2d_unet as base
from pmut_forward.model_2d_unet import PMUTUNet2D


def history_to_records(history: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    """Return one flat record per epoch for CSV/JSON provenance."""
    n_epochs = len(history.get("train_loss", []))
    records: List[Dict[str, Any]] = []
    for idx in range(n_epochs):
        record: Dict[str, Any] = {"epoch": idx + 1}
        for key, values in history.items():
            record[key] = float(values[idx]) if idx < len(values) else None
        records.append(record)
    return records


def save_training_history(history: Dict[str, List[float]], out_dir: Path) -> List[Dict[str, Any]]:
    records = history_to_records(history)
    payload = {
        "schema": "pmut-forward-training-history-v1",
        "n_epochs_recorded": len(records),
        "series": {key: [float(v) for v in values] for key, values in history.items()},
        "records": records,
    }
    (out_dir / "training_history.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = ["epoch", *history.keys()]
    with (out_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records


def jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def save_run_metadata(
    args: argparse.Namespace,
    out_dir: Path,
    data_meta: Dict[str, Any],
    fixed_params: Dict[str, Any],
    model_structure: Dict[str, Any],
    training_protocol: Dict[str, Any],
) -> None:
    metadata = {
        "schema": "pmut-forward-training-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": jsonable_args(args),
        "data_meta": data_meta,
        "fixed_params": fixed_params,
        "model_structure": model_structure,
        "training_protocol": training_protocol,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_fixed_split_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Dict[str, Any]:
    y_std = base.ChannelStandardizer.fit(y[train_idx])
    return {
        "train_loader": base.make_loader(
            x[train_idx],
            y_std.transform(y[train_idx]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val_loader": base.make_loader(
            x[val_idx],
            y_std.transform(y[val_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test_loader": base.make_loader(
            x[test_idx],
            y_std.transform(y[test_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "y_std": y_std,
        "y_test": y[test_idx],
        "split_final": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train fixed PMUT forward 2D U-Net.")
    p.add_argument("--data_dir", type=Path, default=base.DEFAULT_DATA_DIR)
    p.add_argument("--input_key", type=str, default="mask_rand")
    p.add_argument("--target_key", type=str, default="Ptot")
    p.add_argument("--complex_mode", type=str, default="magnitude", choices=["magnitude", "real", "imag"])
    p.add_argument("--cache_path", type=Path, default=base.DEFAULT_CACHE_DIR)
    p.add_argument("--min_active_count", type=int, default=0)
    p.add_argument("--max_active_count", type=int, default=None)
    p.add_argument("--structure_json", type=Path, default=Path("results/fullrange_10k_retrain/model_structure.json"))
    p.add_argument("--out_dir", type=Path, default=Path("results/fullrange_10k_retrain"))
    p.add_argument("--checkpoint_name", type=str, default="pmut_forward_unet_fullrange_10k.pt")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--final_seed", type=int, default=2026)
    p.add_argument("--dropout", type=float, default=0.00010482826590949159)
    p.add_argument("--lr", type=float, default=0.0003314168878377181)
    p.add_argument("--weight_decay", type=float, default=8.268838161067728e-06)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--final_epochs", type=int, default=180)
    p.add_argument("--final_patience", type=int, default=18)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(args.final_seed, deterministic=False)

    x, y, data_meta = base.load_dataset(
        data_dir=args.data_dir,
        input_key=args.input_key,
        target_key=args.target_key,
        complex_mode=args.complex_mode,
        cache_path=args.cache_path,
        rebuild_cache=False,
        max_samples=None,
        min_active_count=args.min_active_count,
        max_active_count=args.max_active_count,
    )
    train_idx, val_idx, test_idx = base.split_indices(
        n=x.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    final_pack = build_fixed_split_dataloaders(
        x=x,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    structure = json.loads(args.structure_json.read_text(encoding="utf-8"))
    model_structure = {
        "base_channels": int(structure["base_channels"]),
        "depth": int(structure["depth"]),
        "kernel_size": int(structure["kernel_size"]),
        "pool_type": str(structure["pool_type"]),
    }

    model = PMUTUNet2D(
        in_channels=int(x.shape[1]),
        out_channels=int(y.shape[1]),
        dropout=float(args.dropout),
        **model_structure,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    model, history, best_val = base.train_model(
        model=model,
        train_loader=final_pack["train_loader"],
        val_loader=final_pack["val_loader"],
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.final_epochs,
        patience=args.final_patience,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        channels_last=args.channels_last,
        use_fused_optimizer=False,
    )
    history_records = save_training_history(history, args.out_dir)
    best_epoch = int(np.argmin(history["val_loss"]) + 1) if history["val_loss"] else None

    criterion = nn.MSELoss()
    test_stats_norm = base.evaluate(model, final_pack["test_loader"], device, criterion, channels_last=args.channels_last)
    y_pred_norm = base.predict_numpy(model, final_pack["test_loader"], device, channels_last=args.channels_last)
    y_pred = final_pack["y_std"].inverse_transform(y_pred_norm)
    y_test = final_pack["y_test"]

    mse = float(np.mean((y_pred - y_test) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    r2 = base.r2_score_np(y_test.reshape(-1), y_pred.reshape(-1))
    mape = base.mape_np(y_test.reshape(-1), y_pred.reshape(-1))
    source_indices = np.asarray(data_meta.get("source_indices", list(range(x.shape[0]))), dtype=np.int64)

    np.save(args.out_dir / "filtered_global_indices.npy", source_indices)
    np.save(args.out_dir / "train_global_indices.npy", source_indices[train_idx])
    np.save(args.out_dir / "val_global_indices.npy", source_indices[val_idx])
    np.save(args.out_dir / "test_global_indices.npy", source_indices[test_idx])

    fixed_params = {
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
    }
    training_protocol = {
        "dataset": "10000 full-range PMUT samples",
        "split": "train/val/test = 7000/1500/1500",
        "split_seed": int(args.split_seed),
        "standardizer_fit_split": "train",
        "model_selection_split": "validation",
        "test_usage": "final reporting only",
    }
    save_run_metadata(
        args=args,
        out_dir=args.out_dir,
        data_meta=data_meta,
        fixed_params=fixed_params,
        model_structure=model_structure,
        training_protocol=training_protocol,
    )
    checkpoint = {
        "state_dict": base.unwrap_model(model).state_dict(),
        "model_config": {
            "in_channels": int(x.shape[1]),
            "out_channels": int(y.shape[1]),
            "dropout": float(args.dropout),
            **model_structure,
        },
        "data_meta": data_meta,
        "standardizer": {
            "mean": final_pack["y_std"].mean,
            "std": final_pack["y_std"].std,
        },
        "args": vars(args),
        "fixed_params": fixed_params,
        "model_structure": model_structure,
        "training_protocol": training_protocol,
        "training_history_files": {
            "csv": "training_history.csv",
            "json": "training_history.json",
        },
        "run_metadata_file": "run_metadata.json",
    }
    torch.save(checkpoint, args.out_dir / args.checkpoint_name)

    metrics = {
        "n_samples": int(x.shape[0]),
        "active_count_filter": {
            "min_active_count": int(args.min_active_count),
            "max_active_count": args.max_active_count,
        },
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_seed": int(args.split_seed),
        "final_training_split": final_pack["split_final"],
        "final_training_mode": "fixed_split",
        "final_seed": int(args.final_seed),
        "fixed_params": fixed_params,
        "model_structure": model_structure,
        "training_protocol": training_protocol,
        "best_val_mse_norm_final": float(best_val),
        "best_val_epoch_final": best_epoch,
        "epochs_recorded": int(len(history_records)),
        "training_history_files": {
            "csv": "training_history.csv",
            "json": "training_history.json",
        },
        "run_metadata_file": "run_metadata.json",
        "test_metrics_denorm": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        },
        "test_metrics_norm": test_stats_norm,
        "data_meta": data_meta,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    base.plot_curves(history, args.out_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Done] saved final training outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
