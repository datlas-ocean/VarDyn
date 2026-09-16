#!/usr/bin/env python3
"""Benchmark multiple identical 4DVar jobs sharing one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

from run_three_minimizers_benchmark import (
    OPTIMIZERS,
    RESULT_PREFIX,
    compare_trajectories,
)


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
        "--concurrency",
        type=int,
        nargs="+",
        default=(2, 4),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=directory / "benchmark_concurrent_minimizers_results.json",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=directory / "benchmark_three_minimizers_full_gpu_results.json",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
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


def process_tree_rss(pid):
    try:
        process = psutil.Process(pid)
        processes = [process] + process.children(recursive=True)
        return sum(
            child.memory_info().rss
            for child in processes
            if child.is_running()
        )
    except (psutil.Error, ProcessLookupError):
        return 0


def nvidia_gpu_sample(gpu):
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    fields = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    return {
        "memory_used_mib": float(fields[0]),
        "utilization_percent": float(fields[1]),
        "power_watts": float(fields[2]),
    }


def take_sample(processes, gpu, phase, start_monotonic):
    gpu_sample = nvidia_gpu_sample(gpu)
    return {
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "wall_time": time.time(),
        "phase": phase,
        "aggregate_cpu_rss_mib": sum(
            process_tree_rss(process.pid) for process in processes
        ) / (1024**2),
        "alive_workers": sum(process.poll() is None for process in processes),
        **(gpu_sample or {}),
    }


def wait_with_sampling(
    predicate,
    processes,
    gpu,
    phase,
    samples,
    start_monotonic,
    interval,
    timeout=None,
):
    wait_start = time.monotonic()
    while not predicate():
        failed = [process.returncode for process in processes if process.poll()]
        if failed:
            raise RuntimeError(f"worker failed before {phase}: {failed}")
        if timeout is not None and time.monotonic() - wait_start > timeout:
            raise TimeoutError(f"timeout while waiting for {phase}")
        samples.append(take_sample(processes, gpu, phase, start_monotonic))
        time.sleep(interval)
    samples.append(take_sample(processes, gpu, phase, start_monotonic))


def parse_worker_result(log_path):
    result_lines = [
        line[len(RESULT_PREFIX):]
        for line in log_path.read_text().splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if not result_lines:
        raise RuntimeError(f"No benchmark JSON found in {log_path}")
    return json.loads(result_lines[-1])


def summarize_samples(samples, baseline_gpu_memory):
    gpu_memory = [
        sample["memory_used_mib"]
        for sample in samples
        if "memory_used_mib" in sample
    ]
    utilization = [
        sample["utilization_percent"]
        for sample in samples
        if "utilization_percent" in sample
    ]
    power = [
        sample["power_watts"]
        for sample in samples
        if "power_watts" in sample
    ]
    cpu_rss = [sample["aggregate_cpu_rss_mib"] for sample in samples]
    return {
        "sample_count": len(samples),
        "peak_aggregate_cpu_rss_mib": max(cpu_rss, default=0.0),
        "peak_gpu_memory_used_mib": max(gpu_memory, default=0.0),
        "peak_incremental_gpu_memory_mib": max(
            (value - baseline_gpu_memory for value in gpu_memory),
            default=0.0,
        ),
        "mean_gpu_utilization_percent": statistics.fmean(utilization)
        if utilization
        else None,
        "p95_gpu_utilization_percent": float(np.percentile(utilization, 95))
        if utilization
        else None,
        "mean_gpu_power_watts": statistics.fmean(power) if power else None,
        "peak_gpu_power_watts": max(power, default=None),
    }


def trajectory_replica_spread(paths):
    archives = [np.load(path) for path in paths]
    try:
        variables = sorted(set.intersection(*(set(item.files) for item in archives)) - {"time_unix"})
        metrics = {}
        for variable in variables:
            reference = np.asarray(archives[0][variable], dtype=np.float64)
            relative_l2 = []
            for archive in archives[1:]:
                candidate = np.asarray(archive[variable], dtype=np.float64)
                valid = np.isfinite(reference) & np.isfinite(candidate)
                difference = reference[valid] - candidate[valid]
                scale = 0.5 * (
                    np.linalg.norm(reference[valid])
                    + np.linalg.norm(candidate[valid])
                )
                relative_l2.append(
                    float(np.linalg.norm(difference) / max(scale, 1e-30))
                )
            metrics[variable] = {
                "maximum_relative_l2_vs_replica_0": max(relative_l2, default=0.0)
            }
        return metrics
    finally:
        for archive in archives:
            archive.close()


def run_batch(args, optimizer, concurrency, work_dir, serial_baseline):
    batch_dir = work_dir / f"{optimizer}-n{concurrency}"
    barrier_dir = batch_dir / "barrier"
    barrier_dir.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).resolve().parent / "benchmark_4dvar_minimizers.py"
    baseline_gpu = nvidia_gpu_sample(args.gpu) or {"memory_used_mib": 0.0}
    processes = []
    log_streams = []
    trajectories = []
    for participant in range(concurrency):
        participant_id = str(participant)
        log_path = batch_dir / f"worker-{participant}.log"
        trajectory = batch_dir / f"worker-{participant}-trajectory.npz"
        run_directory = batch_dir / f"worker-{participant}-files"
        log_stream = log_path.open("w")
        command = [
            sys.executable,
            str(worker_script),
            "--config",
            str(args.config.resolve()),
            "--label",
            f"qg-14d-{optimizer}-n{concurrency}-r{participant}",
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
            "--barrier-dir",
            str(barrier_dir),
            "--participant-id",
            participant_id,
            "--run-directory",
            str(run_directory),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        environment.setdefault("MPLBACKEND", "Agg")
        environment.setdefault("MPLCONFIGDIR", str(run_directory / "matplotlib"))
        processes.append(
            subprocess.Popen(
                command,
                cwd=args.config.resolve().parent,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
        )
        log_streams.append(log_stream)
        trajectories.append(trajectory)

    samples = []
    start_monotonic = time.monotonic()
    try:
        wait_with_sampling(
            lambda: len(list(barrier_dir.glob("ready-*"))) == concurrency,
            processes,
            args.gpu,
            "preparation",
            samples,
            start_monotonic,
            args.sample_seconds,
            timeout=args.ready_timeout,
        )
        go_time = time.time()
        (barrier_dir / "go").write_text(str(go_time))
        wait_with_sampling(
            lambda: len(list(barrier_dir.glob("done-*"))) == concurrency,
            processes,
            args.gpu,
            "minimization",
            samples,
            start_monotonic,
            args.sample_seconds,
        )
        done_times = [
            float(path.read_text()) for path in barrier_dir.glob("done-*")
        ]
        release_time = time.time()
        (barrier_dir / "release-outputs").write_text(str(release_time))
        wait_with_sampling(
            lambda: all(process.poll() is not None for process in processes),
            processes,
            args.gpu,
            "outputs",
            samples,
            start_monotonic,
            args.sample_seconds,
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for stream in log_streams:
            stream.close()

    failures = [process.returncode for process in processes if process.returncode]
    if failures:
        raise RuntimeError(f"{optimizer} n={concurrency} failed: {failures}")
    workers = [
        parse_worker_result(batch_dir / f"worker-{index}.log")
        for index in range(concurrency)
    ]
    minimization_seconds = [item["minimization_seconds"] for item in workers]
    mean_iteration_seconds = [item["mean_iteration_seconds"] for item in workers]
    makespan = max(done_times) - go_time
    total_iterations = sum(item["iterations_completed"] for item in workers)
    serial_seconds = serial_baseline["minimization_seconds"]
    summary = {
        "optimizer": optimizer,
        "concurrency": concurrency,
        "minimization_makespan_seconds": makespan,
        "aggregate_iterations": total_iterations,
        "aggregate_iterations_per_second": total_iterations / makespan,
        "experiments_per_hour": concurrency * 3600 / makespan,
        "mean_worker_minimization_seconds": statistics.fmean(minimization_seconds),
        "min_worker_minimization_seconds": min(minimization_seconds),
        "max_worker_minimization_seconds": max(minimization_seconds),
        "mean_iteration_seconds": statistics.fmean(mean_iteration_seconds),
        "slowdown_vs_serial": statistics.fmean(minimization_seconds) / serial_seconds,
        "throughput_scaling_vs_serial": (
            total_iterations / makespan
        ) / (serial_baseline["iterations_completed"] / serial_seconds),
        "worker_time_coefficient_of_variation": (
            statistics.pstdev(minimization_seconds)
            / statistics.fmean(minimization_seconds)
        ),
        "final_cost_min": min(item["cost_history"][-1] for item in workers),
        "final_cost_max": max(item["cost_history"][-1] for item in workers),
        "replica_output_spread": trajectory_replica_spread(trajectories),
        "resources_all_phases": summarize_samples(
            samples, baseline_gpu["memory_used_mib"]
        ),
        "resources_minimization": summarize_samples(
            [item for item in samples if item["phase"] == "minimization"],
            baseline_gpu["memory_used_mib"],
        ),
        "baseline_gpu_memory_used_mib": baseline_gpu["memory_used_mib"],
        "workers": workers,
        "trajectory_paths": [str(path) for path in trajectories],
        "sample_path": str(batch_dir / "resource_samples.json"),
    }
    (batch_dir / "resource_samples.json").write_text(
        json.dumps(samples, indent=2)
    )
    return summary


def main():
    args = parse_args()
    if any(value < 2 for value in args.concurrency):
        raise ValueError("concurrency levels must be >= 2")
    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vardyn-concurrent-"))
    else:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    baseline_report = json.loads(args.baseline_json.read_text())
    serial_runs = baseline_report["runs"]

    batches = {}
    representative_trajectories = {}
    for concurrency in args.concurrency:
        for optimizer in args.optimizers:
            key = f"{optimizer}-n{concurrency}"
            print(f"Running {key}...", flush=True)
            batches[key] = run_batch(
                args,
                optimizer,
                concurrency,
                work_dir,
                serial_runs[optimizer],
            )
            representative_trajectories.setdefault(concurrency, {})[optimizer] = Path(
                batches[key]["trajectory_paths"][0]
            )
            print(
                f"  {batches[key]['mean_iteration_seconds']:.3f} s/iteration, "
                f"throughput scaling {batches[key]['throughput_scaling_vs_serial']:.2f}x",
                flush=True,
            )

    cross_optimizer_outputs = {
        f"n{concurrency}": compare_trajectories(paths, args.optimizers)
        for concurrency, paths in representative_trajectories.items()
    }
    report = {
        "config": str(args.config.resolve()),
        "iterations": args.iterations,
        "window_days": args.window_days,
        "gpu": str(args.gpu),
        "concurrency_levels": args.concurrency,
        "optimizers": list(args.optimizers),
        "device_resident_state": args.device_resident_state,
        "jit_cost_and_grad": args.jit_cost_and_grad,
        "cost_and_grad_schedule": args.cost_and_grad_schedule,
        "work_dir": str(work_dir),
        "serial_baseline_path": str(args.baseline_json.resolve()),
        "serial_baseline": serial_runs,
        "batches": batches,
        "cross_optimizer_output_comparisons": cross_optimizer_outputs,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Results written to {args.result_json.resolve()}")


if __name__ == "__main__":
    main()
