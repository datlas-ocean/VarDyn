#!/usr/bin/env python3
"""Benchmark VarDyn minimizers with an identical 4DVar problem.

Each variant runs in a fresh process. The historical revision uses SciPy
L-BFGS-B with a host control vector; current variants use the complete rolled
cost-and-gradient evaluation on GPU. The decoupled Optax variant keeps the
L-BFGS algebra and control on GPU but drives its scalar line search from Python,
so the 4DVar executable is not inlined in a second, optimizer-sized executable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from datetime import timedelta
from pathlib import Path

MAPPING_ROOT = Path(__file__).resolve().parents[2]
if str(MAPPING_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPPING_ROOT))

from benchmark_4dvar_iteration import (
    block_until_ready,
    build_checkpoints,
    device_memory_snapshot,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--optimizer",
        choices=("scipy", "optax-decoupled", "optax-full-gpu"),
        required=True,
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument(
        "--relative-gradient-tolerance",
        type=float,
        default=0.0,
        help="Stop when ||g_k|| / ||g_0|| stays below this value; 0 disables.",
    )
    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=1,
        help="Required consecutive accepted iterations below the threshold.",
    )
    parser.add_argument(
        "--minimum-iterations",
        type=int,
        default=1,
        help="Do not apply the benchmark stopping criterion before this count.",
    )
    parser.add_argument("--window-days", type=float, default=2.0)
    parser.add_argument(
        "--device-resident-state",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--jit-cost-and-grad",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--cost-and-grad-schedule",
        choices=("python", "scan"),
        required=True,
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        help="Write the final prognostic trajectory after minimization.",
    )
    parser.add_argument("--barrier-dir", type=Path)
    parser.add_argument("--participant-id")
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="Isolate generated observation/operator caches for this worker.",
    )
    return parser.parse_args()


def cpu_memory_snapshot():
    """Return current RSS and the process peak RSS in MiB."""
    import psutil

    rss = psutil.Process().memory_info().rss
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "rss_bytes": int(rss),
        "rss_mib": rss / (1024**2),
        "peak_rss_bytes": int(peak_kib * 1024),
        "peak_rss_mib": peak_kib / 1024,
    }


def relative_gradient_converged(gradient_norm_history, args):
    """Apply one library-independent stopping criterion at accepted steps."""
    tolerance = args.relative_gradient_tolerance
    if tolerance <= 0.0:
        return False
    iterations_completed = len(gradient_norm_history) - 1
    if iterations_completed < args.minimum_iterations:
        return False
    initial_norm = max(float(gradient_norm_history[0]), 1e-30)
    patience = args.convergence_patience
    if len(gradient_norm_history) - 1 < patience:
        return False
    recent = gradient_norm_history[-patience:]
    return all(
        float(norm) / initial_norm <= tolerance
        for norm in recent
    )


def prepare_problem(args):
    config_path = args.config.resolve()
    os.chdir(config_path.parent)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from src import basis, exp, inv, mod, obs, obsop, state

    setup_start = time.perf_counter()
    config = exp.Exp(str(config_path))
    if args.run_directory is not None:
        run_directory = args.run_directory.resolve()
        run_directory.mkdir(parents=True, exist_ok=True)
        private_directories = {
            name: run_directory / name
            for name in ("obs", "scratch", "outputs", "controls", "operators")
        }
        for directory in private_directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        config.EXP.path_obs = str(private_directories["obs"])
        config.EXP.tmp_DA_path = str(private_directories["scratch"])
        config.EXP.path_save = str(private_directories["outputs"])
        config.INV.path_save_control_vectors = str(private_directories["controls"])
        for operator_config in config.OBSOP.values():
            if hasattr(operator_config, "path_save"):
                operator_config.path_save = str(private_directories["operators"])
    input_substitutions = []
    sst_boundary = config.MOD.bc_files.get("SST")
    if sst_boundary is not None and not glob.glob(sst_boundary["file"]):
        remss_pattern = "/data1/flo/SST/L4/REMSS/mw/20250[56]*.nc"
        if glob.glob(remss_pattern):
            input_substitutions.append({
                "field": "MOD.bc_files.SST.file",
                "missing": sst_boundary["file"],
                "replacement": remss_pattern,
                "reason": "configured ODYSSEA L4 directory is empty",
            })
            sst_boundary["file"] = remss_pattern
            sst_boundary["name_var"] = "analysed_sst"
            sst_boundary["name_lon"] = "lon"
            sst_boundary["name_lat"] = "lat"
            sst_boundary["name_time"] = "time"
    sst_observation = config.OBS.get("SST_MW")
    if sst_observation is not None and not glob.glob(sst_observation.path):
        remss_pattern = "/data1/flo/SST/L4/REMSS/mw/20250[56]*.nc"
        if glob.glob(remss_pattern):
            input_substitutions.append({
                "field": "OBS.SST_MW.path",
                "missing": sst_observation.path,
                "replacement": remss_pattern,
                "reason": "configured ODYSSEA L4 directory is empty",
            })
            sst_observation.path = remss_pattern
            sst_observation.name_var = {"SST": "analysed_sst"}
            sst_observation.name_lon = "lon"
            sst_observation.name_lat = "lat"
            sst_observation.name_time = "time"
    config.INV.device_resident_state = args.device_resident_state == "on"
    config.INV.jit_cost_and_grad = args.jit_cost_and_grad == "on"
    config.INV.cost_and_grad_schedule = args.cost_and_grad_schedule
    if getattr(config.INV, "cost_and_grad_scan_unroll", None) is None:
        config.INV.cost_and_grad_scan_unroll = 1

    if args.window_days > 0:
        requested_end = config.EXP.init_date + timedelta(days=args.window_days)
        config.EXP.final_date = min(config.EXP.final_date, requested_end)

    # Rebuild in every isolated process so stale caches created from missing
    # input paths cannot silently change the benchmark problem.
    config.EXP.compute_obs = True
    config.EXP.flag_plot = 0
    config.EXP.saveoutputs = False
    config.INV.compute_test = False
    config.INV.print_time = False
    config.INV.save_minimization = False
    config.INV.transfer_guard = None
    config.INV.profile_dir = None

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

    if config.INV.device_resident_state and hasattr(model_state, "to_device"):
        model_state.to_device()

    background_covariance = inv.Cov(
        config.INV.sigma_B
        if config.INV.sigma_B is not None
        else basis_sigma
    )
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
    block_until_ready(
        (model_state.var, model_state.params, background, basis_sigma),
        jax,
    )

    control_dtype = getattr(variational, "cost_dtype", jnp.float64)
    control_device = jnp.zeros(background.size, dtype=control_dtype)
    control_host = np.zeros(background.size, dtype=np.float64)
    return {
        "config": config,
        "variational": variational,
        "control_device": control_device,
        "control_host": control_host,
        "checkpoints": checkpoints,
        "setup_seconds": time.perf_counter() - setup_start,
        "grid_shape": list(np.shape(model_state.mask)),
        "control_size": int(background.size),
        "model_state": model_state,
        "model": model,
        "basis": reduced_basis,
        "background_covariance": background_covariance,
        "input_substitutions": input_substitutions,
    }


def host_value_and_grad(variational, control, jax, np):
    value, gradient = variational.cost_and_grad(control)
    block_until_ready((value, gradient), jax)
    value, gradient = jax.device_get((value, gradient))
    return float(np.asarray(value)), np.asarray(gradient, dtype=np.float64)


def warmup(problem, optimizer_name, jax, np):
    variational = problem["variational"]
    control = (
        problem["control_host"]
        if optimizer_name == "scipy"
        else problem["control_device"]
    )
    start = time.perf_counter()
    value, gradient = variational.cost_and_grad(control)
    block_until_ready((value, gradient), jax)
    elapsed = time.perf_counter() - start
    value_host, gradient_host = jax.device_get((value, gradient))
    return (
        elapsed,
        float(np.asarray(value_host)),
        np.asarray(gradient_host, dtype=np.float64),
        value,
        gradient,
    )


def run_scipy(
    problem,
    initial_cost,
    initial_gradient,
    args,
    jax,
    np,
    start_callback=None,
):
    import scipy.optimize as scipy_optimize

    variational = problem["variational"]
    initial_control = problem["control_host"]
    cached_control = np.array(initial_control, copy=True)
    cached_cost = initial_cost
    cached_gradient = np.array(initial_gradient, copy=True)
    evaluations = 1
    accepted_cost = initial_cost
    cost_history = [initial_cost]
    gradient_norm_history = [float(np.linalg.norm(initial_gradient))]
    iteration_seconds = []
    cumulative_seconds = [0.0]
    evaluation_count_history = [evaluations]
    last_iteration_time = None
    stopped_by_criterion = False

    def objective(control):
        nonlocal cached_control, cached_cost, cached_gradient, evaluations
        nonlocal accepted_cost
        if not np.array_equal(control, cached_control):
            cached_cost, cached_gradient = host_value_and_grad(
                variational,
                control,
                jax,
                np,
            )
            cached_control = np.array(control, copy=True)
            evaluations += 1
        accepted_cost = cached_cost
        return cached_cost, cached_gradient

    if start_callback is not None:
        start_callback()
    last_iteration_time = time.perf_counter()
    minimization_start = time.perf_counter()

    def callback(control):
        nonlocal last_iteration_time, stopped_by_criterion
        now = time.perf_counter()
        if not np.array_equal(control, cached_control):
            objective(control)
        iteration_seconds.append(now - last_iteration_time)
        last_iteration_time = now
        cost_history.append(float(accepted_cost))
        gradient_norm_history.append(float(np.linalg.norm(cached_gradient)))
        cumulative_seconds.append(now - minimization_start)
        evaluation_count_history.append(evaluations)
        if relative_gradient_converged(gradient_norm_history, args):
            stopped_by_criterion = True
            raise StopIteration

    result = scipy_optimize.minimize(
        objective,
        initial_control,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": args.iterations,
            "maxfun": max(1000, 50 * args.iterations),
            "maxls": 20,
            "ftol": 0.0,
            "gtol": 0.0,
            "maxcor": args.history_size,
            "disp": False,
        },
    )
    block_until_ready(result.x, jax)
    return {
        "cost_history": cost_history,
        "gradient_norm_history": gradient_norm_history,
        "iteration_seconds": iteration_seconds,
        "cumulative_seconds": cumulative_seconds,
        "evaluation_count_history": evaluation_count_history,
        "optimizer_init_seconds": 0.0,
        "minimization_seconds": time.perf_counter() - minimization_start,
        "iterations_completed": len(iteration_seconds),
        "function_evaluations": evaluations,
        "converged": stopped_by_criterion,
        "status": (
            "relative gradient criterion reached"
            if stopped_by_criterion
            else str(result.message)
        ),
        "final_control_norm": float(np.linalg.norm(result.x)),
        "_final_control": result.x,
    }



def run_optax_decoupled(
    problem,
    initial_cost,
    initial_value,
    initial_grad,
    args,
    jax,
    np,
    start_callback=None,
):
    from src.inv import minimize_optax_decoupled

    result = minimize_optax_decoupled(
        problem["variational"].cost_and_grad,
        problem["control_device"],
        maxiter=args.iterations,
        history_size=args.history_size,
        gtol=args.relative_gradient_tolerance or None,
        convergence_patience=args.convergence_patience,
        minimum_iterations=args.minimum_iterations,
        initial_value=initial_value,
        initial_gradient=initial_grad,
        start_callback=start_callback,
    )
    return {
        "cost_history": result.cost_history,
        "gradient_norm_history": result.gradient_norm_history,
        "iteration_seconds": result.iteration_seconds,
        "cumulative_seconds": result.cumulative_seconds,
        "evaluation_count_history": result.evaluation_count_history,
        "line_search_evaluations": result.line_search_evaluations,
        "step_size_history": result.step_size_history,
        "line_search": "python_armijo_backtracking",
        "optimizer_init_seconds": result.optimizer_init_seconds,
        "minimization_seconds": result.minimization_seconds,
        "iterations_completed": result.iterations_completed,
        "function_evaluations": result.function_evaluations,
        "converged": result.converged,
        "status": result.status,
        "final_control_norm": float(
            np.linalg.norm(np.asarray(jax.device_get(result.control)))
        ),
        "_final_control": result.control,
    }


def run_optax_full_gpu(
    problem,
    initial_value,
    initial_grad,
    args,
    jax,
    np,
    start_callback=None,
):
    from src.inv import minimize_optax_full_gpu

    result = minimize_optax_full_gpu(
        problem["variational"].cost_and_grad,
        problem["control_device"],
        maxiter=args.iterations,
        history_size=args.history_size,
        gtol=args.relative_gradient_tolerance or None,
        convergence_patience=args.convergence_patience,
        minimum_iterations=args.minimum_iterations,
        initial_value=initial_value,
        initial_gradient=initial_grad,
        start_callback=start_callback,
    )
    return {
        "cost_history": result.cost_history,
        "gradient_norm_history": result.gradient_norm_history,
        "iteration_seconds": result.iteration_seconds,
        "cumulative_seconds": result.cumulative_seconds,
        "evaluation_count_history": result.evaluation_count_history,
        "line_search_evaluations": result.line_search_evaluations,
        "step_size_history": result.step_size_history,
        "line_search": "compiled_armijo_backtracking",
        "optimizer_init_seconds": result.optimizer_init_seconds,
        "minimization_seconds": result.minimization_seconds,
        "iterations_completed": result.iterations_completed,
        "function_evaluations": result.function_evaluations,
        "converged": result.converged,
        "status": result.status,
        "final_control_norm": float(
            np.linalg.norm(np.asarray(jax.device_get(result.control)))
        ),
        "_final_control": result.control,
    }


def save_analysis_trajectory(problem, control, output_path, jax, np):
    """Propagate the analysis and save comparable prognostic outputs."""
    import jax.numpy as jnp

    config = problem["config"]
    variational = problem["variational"]
    model = problem["model"]
    basis = problem["basis"]
    covariance = problem["background_covariance"]
    with variational.cost_precision():
        analysis = variational.Xb + covariance.sqr(
            jnp.asarray(control, dtype=variational.cost_dtype)
        )
    analysis = jnp.asarray(analysis, dtype=variational.model_control_dtype)

    state = problem["model_state"].copy()
    model.init(state)
    names = tuple(model.var_to_save)
    trajectory = {name: [state.var[name]] for name in names}
    dates = [config.EXP.init_date.timestamp()]
    date = config.EXP.init_date
    checkpoint_seconds = int(config.INV.timestep_checkpoint.total_seconds())
    save_seconds = int(config.EXP.saveoutput_time_step.total_seconds())
    nstep = min(
        int(checkpoint_seconds // model.dt),
        int(save_seconds // model.dt),
    )
    while date < config.EXP.final_date:
        seconds = int((date - config.EXP.init_date).total_seconds())
        if seconds % checkpoint_seconds == 0:
            basis.operg(seconds / 86400, analysis, State=state.params)
        model.step(t=seconds, State=state, nstep=nstep)
        date += timedelta(seconds=nstep * model.dt)
        elapsed = int((date - config.EXP.init_date).total_seconds())
        if elapsed % save_seconds == 0 and date <= config.EXP.final_date:
            for name in names:
                trajectory[name].append(state.var[name])
            dates.append(date.timestamp())

    stacked = {
        name: np.asarray(jax.device_get(jnp.stack(values)))
        for name, values in trajectory.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time_unix=np.asarray(dates, dtype=np.float64),
        **stacked,
    )
    return {
        "path": str(output_path.resolve()),
        "variables": list(names),
        "time_count": len(dates),
    }


def concurrent_start_barrier(args):
    """Wait until the parent releases all minimizers at the same instant."""
    if args.barrier_dir is None:
        return None
    if not args.participant_id:
        raise ValueError("--participant-id is required with --barrier-dir")
    barrier_dir = args.barrier_dir.resolve()
    barrier_dir.mkdir(parents=True, exist_ok=True)
    ready = barrier_dir / f"ready-{args.participant_id}"
    ready.write_text(str(time.time()))
    go = barrier_dir / "go"
    while not go.exists():
        time.sleep(0.05)
    return {
        "ready_wall_time": float(ready.read_text()),
        "go_wall_time": float(go.read_text()),
        "start_wall_time": time.time(),
    }


def concurrent_finish_barrier(args):
    """Keep postprocessing off the GPU until every minimizer has stopped."""
    if args.barrier_dir is None:
        return None
    barrier_dir = args.barrier_dir.resolve()
    done = barrier_dir / f"done-{args.participant_id}"
    done_time = time.time()
    done.write_text(str(done_time))
    release = barrier_dir / "release-outputs"
    while not release.exists():
        time.sleep(0.05)
    return {
        "done_wall_time": done_time,
        "outputs_release_wall_time": float(release.read_text()),
    }

def main():
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    if args.convergence_patience < 1:
        raise ValueError("--convergence-patience must be >= 1")
    if args.minimum_iterations < 1:
        raise ValueError("--minimum-iterations must be >= 1")
    if not 0.0 <= args.relative_gradient_tolerance < 1.0:
        raise ValueError(
            "--relative-gradient-tolerance must be in [0, 1)"
        )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax
    import numpy as np

    device = jax.devices()[0]
    cpu_memory_initial = cpu_memory_snapshot()
    memory_initial = device_memory_snapshot(device)
    problem = prepare_problem(args)
    cpu_memory_after_setup = cpu_memory_snapshot()
    memory_after_setup = device_memory_snapshot(device)

    (
        compile_seconds,
        initial_cost,
        initial_gradient_host,
        initial_value,
        initial_gradient,
    ) = warmup(problem, args.optimizer, jax, np)
    memory_after_warmup = device_memory_snapshot(device)
    cpu_memory_after_warmup = cpu_memory_snapshot()
    barrier_timing = None

    def start_callback():
        nonlocal barrier_timing
        barrier_timing = concurrent_start_barrier(args)

    if args.optimizer == "scipy":
        optimizer_result = run_scipy(
            problem,
            initial_cost,
            initial_gradient_host,
            args,
            jax,
            np,
            start_callback,
        )
    elif args.optimizer == "optax-decoupled":
        optimizer_result = run_optax_decoupled(
            problem,
            initial_cost,
            initial_value,
            initial_gradient,
            args,
            jax,
            np,
            start_callback,
        )
    else:
        optimizer_result = run_optax_full_gpu(
            problem,
            initial_value,
            initial_gradient,
            args,
            jax,
            np,
            start_callback,
        )

    memory_after_minimization = device_memory_snapshot(device)
    cpu_memory_after_minimization = cpu_memory_snapshot()
    finish_timing = concurrent_finish_barrier(args)
    final_control = optimizer_result.pop("_final_control")
    output_trajectory = None
    if args.output_npz is not None:
        output_trajectory = save_analysis_trajectory(
            problem,
            final_control,
            args.output_npz,
            jax,
            np,
        )
    config = problem["config"]
    result = {
        "label": args.label,
        "optimizer": args.optimizer,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(device),
        "window_start": config.EXP.init_date.isoformat(),
        "window_end": config.EXP.final_date.isoformat(),
        "window_days": (
            config.EXP.final_date - config.EXP.init_date
        ).total_seconds() / 86400,
        "grid_shape": problem["grid_shape"],
        "control_size": problem["control_size"],
        "input_substitutions": problem["input_substitutions"],
        "checkpoint_count": int(problem["checkpoints"].size),
        "device_resident_state": bool(config.INV.device_resident_state),
        "jit_cost_and_grad": bool(config.INV.jit_cost_and_grad),
        "cost_and_grad_schedule": config.INV.cost_and_grad_schedule,
        "iterations_requested": args.iterations,
        "history_size": args.history_size,
        "relative_gradient_tolerance": args.relative_gradient_tolerance,
        "convergence_patience": args.convergence_patience,
        "minimum_iterations": args.minimum_iterations,
        "setup_seconds": problem["setup_seconds"],
        "cost_compile_and_first_evaluation_seconds": compile_seconds,
        "initial_cost": initial_cost,
        "initial_gradient_norm": float(np.linalg.norm(initial_gradient_host)),
        "memory_initial": memory_initial,
        "memory_after_setup": memory_after_setup,
        "memory_after_warmup": memory_after_warmup,
        "memory_after_minimization": memory_after_minimization,
        "cpu_memory_initial": cpu_memory_initial,
        "cpu_memory_after_setup": cpu_memory_after_setup,
        "cpu_memory_after_warmup": cpu_memory_after_warmup,
        "cpu_memory_after_minimization": cpu_memory_after_minimization,
        "output_trajectory": output_trajectory,
        "barrier_timing": barrier_timing,
        "finish_timing": finish_timing,
        **optimizer_result,
    }
    result["final_relative_gradient_norm"] = (
        result["gradient_norm_history"][-1]
        / max(result["gradient_norm_history"][0], 1e-30)
    )
    result["total_numerical_seconds"] = (
        result["cost_compile_and_first_evaluation_seconds"]
        + result["optimizer_init_seconds"]
        + result["minimization_seconds"]
    )
    result["mean_iteration_seconds"] = (
        result["minimization_seconds"]
        / max(result["iterations_completed"], 1)
    )
    print("VARDYN_MINIMIZER_BENCHMARK_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
