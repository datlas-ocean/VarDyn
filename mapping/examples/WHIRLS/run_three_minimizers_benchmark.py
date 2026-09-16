#!/usr/bin/env python3
"""Run the SciPy, decoupled Optax, and full-GPU Optax WHIRLS benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np


OPTIMIZERS = ("scipy", "optax-decoupled", "optax-full-gpu")
RESULT_PREFIX = "VARDYN_MINIMIZER_BENCHMARK_JSON="


def parse_args():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=directory / "config_VarDyn-QG.py",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--window-days", type=float, default=14.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=directory / "benchmark_three_minimizers_full_gpu_results.json",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=OPTIMIZERS,
        default=OPTIMIZERS,
    )
    parser.add_argument(
        "--device-resident-state", choices=("on", "off"), default="on"
    )
    parser.add_argument(
        "--jit-cost-and-grad", choices=("on", "off"), default="on"
    )
    parser.add_argument(
        "--cost-and-grad-schedule",
        choices=("scan", "python"),
        default="scan",
    )
    return parser.parse_args()


def run_case(args, optimizer, work_dir):
    worker = Path(__file__).resolve().parent / "benchmark_4dvar_minimizers.py"
    trajectory = work_dir / f"{optimizer}-trajectory.npz"
    command = [
        sys.executable,
        str(worker),
        "--config",
        str(args.config.resolve()),
        "--label",
        f"qg-14d-{optimizer}",
        "--optimizer",
        optimizer,
        "--iterations",
        str(args.iterations),
        "--window-days",
        str(args.window_days),
        "--device-resident-state",
        args.device_resident_state,
        "--jit-cost-and-grad",
        args.jit_cost_and_grad,
        "--cost-and-grad-schedule",
        args.cost_and_grad_schedule,
        "--relative-gradient-tolerance",
        "0",
        "--output-npz",
        str(trajectory),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(work_dir / "matplotlib"))
    completed = subprocess.run(
        command,
        cwd=args.config.resolve().parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = work_dir / f"{optimizer}.log"
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{optimizer} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    result_lines = [
        line[len(RESULT_PREFIX):]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if not result_lines:
        raise RuntimeError(f"No benchmark JSON found in {log_path}")
    result = json.loads(result_lines[-1])
    result["log_path"] = str(log_path)
    return result, trajectory


def finite_pair(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    return np.asarray(left[mask], dtype=np.float64), np.asarray(
        right[mask], dtype=np.float64
    )


def compare_trajectories(paths, optimizers=None):
    optimizers = tuple(optimizers or paths)
    loaded = {name: np.load(path) for name, path in paths.items()}
    comparisons = {}
    try:
        for left_name, right_name in combinations(optimizers, 2):
            left = loaded[left_name]
            right = loaded[right_name]
            variables = sorted(
                (set(left.files) & set(right.files)) - {"time_unix"}
            )
            variable_metrics = {}
            sum_difference_sq = 0.0
            sum_reference_sq = 0.0
            sum_relative = 0.0
            valid_count = 0
            for variable in variables:
                left_values, right_values = finite_pair(
                    left[variable], right[variable]
                )
                difference = left_values - right_values
                scale = 0.5 * (
                    np.linalg.norm(left_values)
                    + np.linalg.norm(right_values)
                )
                relative_l2 = np.linalg.norm(difference) / max(scale, 1e-30)
                point_scale = 0.5 * (
                    np.abs(left_values) + np.abs(right_values)
                )
                floor = max(
                    float(np.nanmax(point_scale)) * 1e-8,
                    1e-30,
                )
                point_relative = np.abs(difference) / np.maximum(
                    point_scale, floor
                )
                variable_metrics[variable] = {
                    "relative_l2": float(relative_l2),
                    "mean_relative_absolute": float(np.mean(point_relative)),
                    "root_mean_square_difference": float(
                        np.sqrt(np.mean(difference**2))
                    ),
                    "valid_values": int(difference.size),
                }
                sum_difference_sq += float(np.vdot(difference, difference))
                sum_reference_sq += float(scale**2)
                sum_relative += float(np.sum(point_relative))
                valid_count += int(difference.size)
            comparisons[f"{left_name}__vs__{right_name}"] = {
                "variables": variable_metrics,
                "aggregate_relative_l2": float(
                    np.sqrt(sum_difference_sq)
                    / max(np.sqrt(sum_reference_sq), 1e-30)
                ),
                "aggregate_mean_relative_absolute": float(
                    sum_relative / max(valid_count, 1)
                ),
                "valid_values": valid_count,
            }
    finally:
        for archive in loaded.values():
            archive.close()
    return comparisons


def main():
    args = parse_args()
    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vardyn-minimizers-"))
    else:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    trajectories = {}
    for optimizer in args.optimizers:
        print(f"Running {optimizer}...", flush=True)
        runs[optimizer], trajectories[optimizer] = run_case(
            args, optimizer, work_dir
        )
        print(
            f"  {runs[optimizer]['mean_iteration_seconds']:.3f} s/iteration",
            flush=True,
        )

    report = {
        "config": str(args.config.resolve()),
        "iterations": args.iterations,
        "window_days": args.window_days,
        "gpu": str(args.gpu),
        "optimizers": list(args.optimizers),
        "device_resident_state": args.device_resident_state,
        "jit_cost_and_grad": args.jit_cost_and_grad,
        "cost_and_grad_schedule": args.cost_and_grad_schedule,
        "work_dir": str(work_dir),
        "runs": runs,
        "output_comparisons": compare_trajectories(
            trajectories, args.optimizers
        ),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Results written to {args.result_json.resolve()}")


if __name__ == "__main__":
    main()
