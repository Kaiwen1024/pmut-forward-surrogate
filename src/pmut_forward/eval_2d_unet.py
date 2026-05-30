#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained 2D U-Net checkpoint on the PMUT dataset.

Main purpose:
    1. Report strict test-split metrics using the same split seed/ratios saved
       in the checkpoint.
    2. Measure inference latency for deployment-oriented reporting:
       - single-mask latency
       - 100-mask batch latency
       - average per-sample latency in batch mode
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from . import train_2d_unet as base
    from .model_2d_unet import PMUTUNet2D
except ImportError:
    import train_2d_unet as base
    from model_2d_unet import PMUTUNet2D


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(
    ckpt: Dict[str, Any],
    device: torch.device,
    multi_gpu: bool,
    device_ids: Optional[List[int]],
    compile_enabled: bool,
    compile_mode: str,
    compile_dynamic: bool,
    channels_last: bool,
) -> nn.Module:
    cfg = ckpt["model_config"]

    if "kernel_size" in cfg or "pool_type" in cfg:
        model = PMUTUNet2D(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]),
            base_channels=int(cfg["base_channels"]),
            depth=int(cfg["depth"]),
            kernel_size=int(cfg.get("kernel_size", 3)),
            pool_type=str(cfg.get("pool_type", "max")),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
    else:
        model = base.UNet2D(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]),
            base_channels=int(cfg["base_channels"]),
            depth=int(cfg["depth"]),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    model.load_state_dict(ckpt["state_dict"])
    model = base.maybe_parallel(model, use_multi_gpu=multi_gpu, device_ids=device_ids)
    model = base.maybe_compile_model(
        model,
        enable_compile=compile_enabled,
        compile_mode=compile_mode,
        dynamic=compile_dynamic,
    )
    model.eval()
    return model


def resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def infer_runtime_bool(cli_value: Optional[bool], train_args: Dict[str, Any], key: str, default: bool = False) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return bool(train_args.get(key, default))


def infer_runtime_value(cli_value: Any, train_args: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return train_args.get(key, default)


def make_y_standardizer(ckpt: Dict[str, Any], y_train: np.ndarray) -> base.ChannelStandardizer:
    std_meta = ckpt.get("standardizer")
    if std_meta is not None:
        mean = np.asarray(std_meta["mean"], dtype=np.float32)
        std = np.asarray(std_meta["std"], dtype=np.float32)
        return base.ChannelStandardizer(mean=mean, std=std)
    return base.ChannelStandardizer.fit(y_train)


@torch.inference_mode()
def evaluate_and_predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    channels_last: bool,
) -> tuple[Dict[str, float], np.ndarray, float]:
    criterion = nn.MSELoss()
    outputs: List[np.ndarray] = []
    total_loss = 0.0
    n_samples = 0
    use_amp = amp and device.type == "cuda"
    autocast_dtype = base.resolve_amp_dtype(amp_dtype)

    sync_if_cuda(device)
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        xb = base.maybe_channels_last(xb, channels_last)
        yb = yb.to(device, non_blocking=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
        with autocast_ctx:
            pred = model(xb)
        loss = criterion(pred.float(), yb.float())
        bs = int(xb.size(0))
        total_loss += float(loss.item()) * bs
        n_samples += bs
        outputs.append(pred.float().cpu().numpy())
    sync_if_cuda(device)
    elapsed = time.perf_counter() - t0

    mse = total_loss / max(n_samples, 1)
    stats = {"loss": mse, "mse": mse, "rmse": math.sqrt(max(mse, 0.0))}
    return stats, np.concatenate(outputs, axis=0), elapsed


@torch.inference_mode()
def predict_batch(
    model: nn.Module,
    x_batch: np.ndarray,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    channels_last: bool,
) -> np.ndarray:
    xb = torch.from_numpy(x_batch).to(device)
    xb = base.maybe_channels_last(xb, channels_last)
    use_amp = amp and device.type == "cuda"
    autocast_dtype = base.resolve_amp_dtype(amp_dtype)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
    with autocast_ctx:
        pred = model(xb)
    return pred.float().cpu().numpy()


def benchmark_speed(
    model: nn.Module,
    x_ref: np.ndarray,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    channels_last: bool,
    warmup: int,
    single_repeats: int,
    batch_repeats: int,
    batch_n: int,
) -> Dict[str, float]:
    if x_ref.shape[0] < batch_n:
        reps = int(math.ceil(batch_n / x_ref.shape[0]))
        x_batch = np.tile(x_ref, (reps, 1, 1, 1))[:batch_n]
    else:
        x_batch = x_ref[:batch_n]
    x_single = x_batch[:1]

    # Warm up both batch shapes. With torch.compile enabled, batch size changes
    # can trigger separate graph compilation, which would otherwise pollute the
    # measured single-mask mean latency.
    for _ in range(warmup):
        predict_batch(model, x_batch, device, amp, amp_dtype, channels_last)
    for _ in range(warmup):
        predict_batch(model, x_single, device, amp, amp_dtype, channels_last)
    sync_if_cuda(device)

    single_times: List[float] = []
    for _ in range(single_repeats):
        t0 = time.perf_counter()
        predict_batch(model, x_single, device, amp, amp_dtype, channels_last)
        sync_if_cuda(device)
        single_times.append(time.perf_counter() - t0)

    batch_times: List[float] = []
    for _ in range(batch_repeats):
        t0 = time.perf_counter()
        predict_batch(model, x_batch, device, amp, amp_dtype, channels_last)
        sync_if_cuda(device)
        batch_times.append(time.perf_counter() - t0)

    single = np.asarray(single_times, dtype=np.float64)
    batch = np.asarray(batch_times, dtype=np.float64)
    return {
        "single_mask_ms_mean": float(single.mean() * 1000.0),
        "single_mask_ms_median": float(np.median(single) * 1000.0),
        "batch100_ms_mean": float(batch.mean() * 1000.0),
        "batch100_ms_median": float(np.median(batch) * 1000.0),
        "batch100_per_sample_ms_mean": float(batch.mean() * 1000.0 / batch_n),
        "batch_size_for_batch_benchmark": float(batch_n),
        "single_repeats": float(single_repeats),
        "batch_repeats": float(batch_repeats),
        "warmup_repeats": float(warmup),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate 2D U-Net checkpoint and benchmark inference speed.")
    p.add_argument("--checkpoint_path", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--cache_path", type=Path, default=None)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--min_active_count", type=int, default=None)
    p.add_argument("--max_active_count", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--multi_gpu", dest="multi_gpu", action="store_true")
    p.add_argument("--single_gpu", dest="multi_gpu", action="store_false")
    p.set_defaults(multi_gpu=None)
    p.add_argument("--device_ids", type=str, default=None)
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=None)
    p.add_argument("--amp_dtype", type=str, default=None, choices=["float16", "bfloat16"])
    p.add_argument("--channels_last", dest="channels_last", action="store_true")
    p.add_argument("--no_channels_last", dest="channels_last", action="store_false")
    p.set_defaults(channels_last=None)
    p.add_argument("--compile", dest="compile", action="store_true")
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.set_defaults(compile=None)
    p.add_argument("--compile_mode", type=str, default=None, choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile_dynamic", action="store_true")
    p.add_argument("--eval_split", type=str, default="test", choices=["test", "all"])
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--speed_warmup", type=int, default=5)
    p.add_argument("--speed_single_repeats", type=int, default=30)
    p.add_argument("--speed_batch_repeats", type=int, default=10)
    p.add_argument("--speed_batch_n", type=int, default=100)
    p.add_argument("--out_dir", type=Path, default=Path("results") / "fullrange_10k_retrain" / "eval_test")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint_path, device)
    train_args = dict(ckpt.get("args", {}))

    data_dir_raw = infer_runtime_value(args.data_dir, train_args, "data_dir", None)
    if data_dir_raw is None:
        raise ValueError("data_dir is required either from CLI or checkpoint args.")
    data_dir = Path(data_dir_raw)

    cache_path_raw = infer_runtime_value(args.cache_path, train_args, "cache_path", None)
    cache_path = Path(cache_path_raw) if cache_path_raw else None
    input_key = str(infer_runtime_value(None, train_args, "input_key", "mask_rand"))
    target_key = str(infer_runtime_value(None, train_args, "target_key", "Ptot"))
    complex_mode = str(infer_runtime_value(None, train_args, "complex_mode", "magnitude"))
    max_samples = infer_runtime_value(args.max_samples, train_args, "max_samples", None)
    min_active_count = int(infer_runtime_value(args.min_active_count, train_args, "min_active_count", 0))
    max_active_count = infer_runtime_value(args.max_active_count, train_args, "max_active_count", None)
    batch_size = int(infer_runtime_value(args.batch_size, train_args, "batch_size", 16))
    num_workers = int(infer_runtime_value(args.num_workers, train_args, "num_workers", 0))
    pin_memory = infer_runtime_bool(args.pin_memory, train_args, "pin_memory", False)
    amp = infer_runtime_bool(args.amp, train_args, "amp", False)
    amp_dtype = str(infer_runtime_value(args.amp_dtype, train_args, "amp_dtype", "bfloat16"))
    channels_last = infer_runtime_bool(args.channels_last, train_args, "channels_last", False)
    compile_enabled = infer_runtime_bool(args.compile, train_args, "compile", False)
    compile_mode = str(infer_runtime_value(args.compile_mode, train_args, "compile_mode", "default"))
    multi_gpu = infer_runtime_bool(args.multi_gpu, train_args, "multi_gpu", False)
    device_ids = base.parse_device_ids(args.device_ids) if args.device_ids is not None else base.parse_device_ids(train_args.get("device_ids"))

    if infer_runtime_bool(None, train_args, "fast_math", False):
        base.configure_torch(fast_math=True, cudnn_benchmark=False)

    x, y, data_meta = base.load_dataset(
        data_dir=data_dir,
        input_key=input_key,
        target_key=target_key,
        complex_mode=complex_mode,
        cache_path=cache_path,
        rebuild_cache=args.rebuild_cache,
        max_samples=max_samples,
        min_active_count=min_active_count,
        max_active_count=max_active_count,
    )

    split_seed = int(infer_runtime_value(None, train_args, "split_seed", infer_runtime_value(None, train_args, "seed", 42)))
    train_idx, val_idx, test_idx = base.split_indices(
        n=x.shape[0],
        train_ratio=float(infer_runtime_value(None, train_args, "train_ratio", 0.7)),
        val_ratio=float(infer_runtime_value(None, train_args, "val_ratio", 0.15)),
        seed=split_seed,
    )

    x_train = x[train_idx]
    y_train = y[train_idx]
    y_std = make_y_standardizer(ckpt, y_train)
    source_indices = np.asarray(data_meta.get("source_indices", list(range(x.shape[0]))), dtype=np.int64)

    if args.eval_split == "test":
        eval_idx = test_idx
    else:
        eval_idx = np.arange(x.shape[0])

    x_eval = x[eval_idx]
    y_eval = y[eval_idx]
    y_eval_norm = y_std.transform(y_eval)

    loader = base.make_loader(
        x_eval,
        y_eval_norm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        ckpt=ckpt,
        device=device,
        multi_gpu=multi_gpu,
        device_ids=device_ids,
        compile_enabled=compile_enabled,
        compile_mode=compile_mode,
        compile_dynamic=args.compile_dynamic,
        channels_last=channels_last,
    )

    norm_stats, y_pred_norm, total_eval_sec = evaluate_and_predict(
        model=model,
        loader=loader,
        device=device,
        amp=amp,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )
    y_pred = y_std.inverse_transform(y_pred_norm)

    mse = float(np.mean((y_pred - y_eval) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_pred - y_eval)))
    r2 = base.r2_score_np(y_eval.reshape(-1), y_pred.reshape(-1))
    mape = base.mape_np(y_eval.reshape(-1), y_pred.reshape(-1))

    speed = benchmark_speed(
        model=model,
        x_ref=x_eval,
        device=device,
        amp=amp,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        warmup=args.speed_warmup,
        single_repeats=args.speed_single_repeats,
        batch_repeats=args.speed_batch_repeats,
        batch_n=args.speed_batch_n,
    )

    if args.save_predictions:
        np.save(args.out_dir / f"y_{args.eval_split}_true.npy", y_eval)
        np.save(args.out_dir / f"y_{args.eval_split}_pred.npy", y_pred)
    eval_global_indices = source_indices[eval_idx]
    np.save(args.out_dir / f"{args.eval_split}_global_indices.npy", eval_global_indices)

    gpu_count = len(getattr(model, "device_ids", [])) if isinstance(model, nn.DataParallel) else (1 if device.type == "cuda" else 0)
    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "data_dir": str(data_dir),
        "cache_path": str(cache_path) if cache_path is not None else None,
        "eval_split": args.eval_split,
        "n_samples_total": int(x.shape[0]),
        "evaluated_samples": int(x_eval.shape[0]),
        "active_count_filter": {
            "min_active_count": int(min_active_count),
            "max_active_count": None if max_active_count is None else int(max_active_count),
        },
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_seed": split_seed,
        "input_shape": list(x.shape[1:]),
        "target_shape": list(y.shape[1:]),
        "device": str(device),
        "gpu_count": int(gpu_count),
        "multi_gpu": bool(multi_gpu),
        "device_ids": device_ids,
        "amp": bool(amp),
        "amp_dtype": amp_dtype,
        "channels_last": bool(channels_last),
        "compile": bool(compile_enabled),
        "compile_mode": compile_mode if compile_enabled else None,
        "pointwise_metrics_denorm": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        },
        "pointwise_metrics_standardized": norm_stats,
        "speed_metrics": speed,
        "total_eval_sec_including_data_loading": float(total_eval_sec),
        "throughput_samples_per_sec_including_data_loading": float(x_eval.shape[0] / max(total_eval_sec, 1e-12)),
        "data_meta": data_meta,
        "train_args": json_safe(train_args),
        "note": "Use eval_split=test for strict held-out generalization reporting.",
    }

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Done] saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
