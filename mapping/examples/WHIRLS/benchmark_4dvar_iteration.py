#!/usr/bin/env python3
"""Benchmark one VarDyn 4DVar cost-and-gradient evaluation.

This helper is intentionally executed in a fresh Python process for each
variant. That isolates JAX compilation caches and prevents modules from one
Git revision leaking into another.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--device-resident-state",
        choices=("config", "on", "off"),
        default="config",
        help="Override INV.device_resident_state for this isolated run.",
    )
    parser.add_argument(
        "--jit-cost-and-grad",
        choices=("config", "on", "off"),
        default="config",
        help="Override INV.jit_cost_and_grad for this isolated run.",
    )
    parser.add_argument(
        "--cost-and-grad-schedule",
        choices=("config", "python", "scan"),
        default="config",
        help="Select the historical Python loops or rolled lax.scan loops.",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=2.0,
        help="Assimilation-window length. Use 0 for the complete config window.",
    )
    return parser.parse_args()


def block_until_ready(tree, jax):
    return jax.tree_util.tree_map(
        lambda value: (
            value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value
        ),
        tree,
    )


def device_memory_snapshot(device):
    """Return allocator counters in bytes and MiB."""
    stats = device.memory_stats() or {}
    byte_keys = (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_reserved",
        "peak_bytes_reserved",
        "bytes_limit",
        "largest_alloc_size",
        "largest_free_block_bytes",
        "pool_bytes",
        "peak_pool_bytes",
    )
    snapshot = {
        key: int(stats.get(key, 0) or 0)
        for key in byte_keys
    }
    snapshot["num_allocs"] = int(stats.get("num_allocs", 0) or 0)
    for key in byte_keys:
        snapshot[f"{key}_mib"] = snapshot[key] / (1024**2)
    limit = snapshot["bytes_limit"]
    snapshot["in_use_percent"] = (
        100 * snapshot["bytes_in_use"] / limit if limit else None
    )
    snapshot["peak_in_use_percent"] = (
        100 * snapshot["peak_bytes_in_use"] / limit if limit else None
    )
    return snapshot


def build_checkpoints(config, model, obs_operator, np):
    nstep_check = int(
        config.INV.timestep_checkpoint.total_seconds() // model.dt
    )
    checkpoints = [0]
    time_checkpoints = [np.datetime64(model.timestamps[0])]
    t_checkpoints = [model.T[0]]
    check = 0

    for index, timestamp in enumerate(model.timestamps[:-1]):
        if index > 0 and (
            obs_operator.is_obs(timestamp) or check == nstep_check
        ):
            checkpoints.append(index)
            time_checkpoints.append(np.datetime64(timestamp))
            t_checkpoints.append(model.T[index])
            if check == nstep_check:
                check = 0
        check += 1

    checkpoints.append(len(model.timestamps) - 1)
    time_checkpoints.append(np.datetime64(model.timestamps[-1]))
    t_checkpoints.append(model.T[-1])
    return (
        np.asarray(checkpoints),
        np.asarray(time_checkpoints),
        np.asarray(t_checkpoints),
        nstep_check,
    )


def main():
    args = parse_args()
    config_path = args.config.resolve()
    os.chdir(config_path.parent)

    # These must be set before importing JAX or matplotlib through VarDyn.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax
    import numpy as np

    from src import basis, exp, inv, mod, obs, obsop, state

    device = jax.devices()[0]
    memory_initial = device_memory_snapshot(device)
    setup_start = time.perf_counter()
    config = exp.Exp(str(config_path))

    if args.device_resident_state != "config":
        config.INV.device_resident_state = (
            args.device_resident_state == "on"
        )
    if args.jit_cost_and_grad != "config":
        config.INV.jit_cost_and_grad = args.jit_cost_and_grad == "on"
    if args.cost_and_grad_schedule != "config":
        config.INV.cost_and_grad_schedule = args.cost_and_grad_schedule
    if getattr(config.INV, "cost_and_grad_scan_unroll", None) is None:
        # Older revisions may materialize unknown configuration keys as None.
        config.INV.cost_and_grad_scan_unroll = 1

    if args.window_days > 0:
        requested_end = config.EXP.init_date + timedelta(
            days=args.window_days
        )
        config.EXP.final_date = min(config.EXP.final_date, requested_end)

    # Benchmark only the numerical iteration. Reuse observation caches and
    # disable diagnostic side effects.
    config.EXP.compute_obs = False
    config.EXP.flag_plot = 0
    config.EXP.saveoutputs = False
    config.INV.compute_test = False
    config.INV.print_time = False
    config.INV.save_minimization = False
    config.INV.transfer_guard = None
    config.INV.profile_dir = None

    # The example config deliberately leaves the optional per-satellite error
    # directory empty. For a newly shortened window there is no observation
    # cache yet, so fall back to each product's configured sigma_noise instead
    # of trying to open paths such as /noise_alg.nc.
    for observation_config in config.OBS.values():
        error_path = getattr(observation_config, "path_err", None)
        if error_path and not Path(error_path).expanduser().exists():
            observation_config.path_err = None

    model_state = state.State(config, verbose=False)
    model = mod.Model(config, model_state, verbose=False)
    observations = obs.Obs(config, model_state)
    obs_operator = obsop.Obsop(
        config,
        model_state,
        observations,
        model,
        verbose=False,
    )
    reduced_basis = basis.Basis(config, model_state, verbose=False)
    obs_operator.process_obs()

    (
        checkpoints,
        time_checkpoints,
        t_checkpoints,
        nstep_check,
    ) = build_checkpoints(config, model, obs_operator, np)
    model.set_bc(time_checkpoints, t_bc=t_checkpoints)
    model.init(model_state)

    time_basis = np.arange(
        0,
        model.T[-1] + nstep_check * model.dt,
        nstep_check * model.dt,
    ) / (24 * 3600)
    background, basis_sigma = reduced_basis.set_basis(
        time_basis,
        return_q=True,
        State=model_state,
    )

    device_resident = bool(
        getattr(config.INV, "device_resident_state", False)
        and hasattr(model_state, "to_device")
    )
    if device_resident:
        model_state.to_device()

    if config.INV.sigma_B is not None:
        background_covariance = inv.Cov(config.INV.sigma_B)
    else:
        background_covariance = inv.Cov(basis_sigma)
    observation_covariance = inv.Cov(config.INV.sigma_R)

    variational = inv.Variational(
        config=config,
        M=model,
        H=obs_operator,
        State=model_state,
        B=background_covariance,
        R=observation_covariance,
        Basis=reduced_basis,
        Xb=background,
        checkpoints=checkpoints,
        freq_it_plot=config.INV.freq_it_plot,
        print_time=False,
    )
    control_host = np.zeros(background.size, dtype=np.float64)
    block_until_ready(
        (
            model_state.var,
            model_state.params,
            background,
            basis_sigma,
        ),
        jax,
    )
    setup_seconds = time.perf_counter() - setup_start
    memory_after_setup = device_memory_snapshot(device)

    def evaluate_once():
        # Include the deliberate SciPy adapter transfers in the optimized
        # measurement. The legacy revision receives its original host vector.
        control = (
            jax.device_put(control_host)
            if device_resident
            else control_host
        )
        start = time.perf_counter()
        cost, gradient = variational.cost_and_grad(control)
        block_until_ready((cost, gradient), jax)
        cost_host, gradient_host = jax.device_get((cost, gradient))
        elapsed = time.perf_counter() - start
        return (
            elapsed,
            float(np.asarray(cost_host)),
            np.asarray(gradient_host, dtype=np.float64),
        )

    compile_seconds, warmup_cost, warmup_gradient = evaluate_once()
    memory_after_warmup = device_memory_snapshot(device)
    samples = []
    costs = []
    gradient_norms = []
    gradient_sums = []
    gradient_max_abs = []
    for _ in range(args.repeats):
        elapsed, cost, gradient = evaluate_once()
        samples.append(elapsed)
        costs.append(cost)
        gradient_norms.append(float(np.linalg.norm(gradient)))
        gradient_sums.append(float(np.sum(gradient, dtype=np.float64)))
        gradient_max_abs.append(float(np.max(np.abs(gradient))))
    memory_after_iterations = device_memory_snapshot(device)

    evaluation_peak_increment = max(
        0,
        memory_after_warmup["peak_bytes_in_use"]
        - memory_after_setup["peak_bytes_in_use"],
    )
    evaluation_resident_increment = (
        memory_after_warmup["bytes_in_use"]
        - memory_after_setup["bytes_in_use"]
    )

    result = {
        "label": args.label,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_resident_state": device_resident,
        "jit_cost_and_grad": bool(
            getattr(config.INV, "jit_cost_and_grad", False)
        ),
        "cost_and_grad_schedule": getattr(
            config.INV,
            "cost_and_grad_schedule",
            "python",
        ),
        "cost_and_grad_scan_unroll": int(
            getattr(config.INV, "cost_and_grad_scan_unroll", 1) or 1
        ),
        "window_start": config.EXP.init_date.isoformat(),
        "window_end": config.EXP.final_date.isoformat(),
        "window_days": (
            config.EXP.final_date - config.EXP.init_date
        ).total_seconds()
        / 86400,
        "grid_shape": list(np.shape(model_state.mask)),
        "control_size": int(background.size),
        "checkpoint_count": int(checkpoints.size),
        "setup_seconds": setup_seconds,
        "compile_and_first_evaluation_seconds": compile_seconds,
        "iteration_seconds": samples,
        "iteration_median_seconds": statistics.median(samples),
        "iteration_mean_seconds": statistics.mean(samples),
        "iteration_min_seconds": min(samples),
        "cost": statistics.mean(costs),
        "gradient_norm": statistics.mean(gradient_norms),
        "gradient_sum": statistics.mean(gradient_sums),
        "gradient_max_abs": statistics.mean(gradient_max_abs),
        "warmup_cost": warmup_cost,
        "warmup_gradient_norm": float(np.linalg.norm(warmup_gradient)),
        "memory_initial": memory_initial,
        "memory_after_setup": memory_after_setup,
        "memory_after_warmup": memory_after_warmup,
        "memory_after_iterations": memory_after_iterations,
        "evaluation_peak_increment_bytes": evaluation_peak_increment,
        "evaluation_peak_increment_mib": (
            evaluation_peak_increment / (1024**2)
        ),
        "evaluation_resident_increment_bytes": (
            evaluation_resident_increment
        ),
        "evaluation_resident_increment_mib": (
            evaluation_resident_increment / (1024**2)
        ),
    }
    print("VARDYN_BENCHMARK_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
