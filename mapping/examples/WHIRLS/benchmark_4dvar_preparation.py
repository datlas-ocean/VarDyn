#!/usr/bin/env python3
"""Measure the monthly 4DVar preparation phases in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from benchmark_4dvar_iteration import (
    block_until_ready,
    build_checkpoints,
    device_memory_snapshot,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main():
    args = _arguments()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")
    config_path = args.config.resolve()
    os.chdir(config_path.parent)

    import jax
    import numpy as np

    from src import basis, exp, inv, mod, obs, obsop, state

    timings = {}

    def measure(name, function):
        start = time.perf_counter()
        value = function()
        timings[name] = time.perf_counter() - start
        return value

    total_start = time.perf_counter()
    config = measure("config_seconds", lambda: exp.Exp(str(config_path)))
    config.EXP.compute_obs = False
    for operator_config in config.OBSOP.values():
        operator_config.compute_op = False
    config.EXP.flag_plot = 0
    config.EXP.saveoutputs = False
    config.INV.compute_test = False
    config.INV.print_time = False
    config.INV.save_minimization = False

    for observation_config in config.OBS.values():
        error_path = getattr(observation_config, "path_err", None)
        if error_path and not Path(error_path).expanduser().exists():
            observation_config.path_err = None

    model_state = measure(
        "state_seconds",
        lambda: state.State(config, verbose=False),
    )
    model = measure(
        "model_and_bc_open_seconds",
        lambda: mod.Model(config, model_state, verbose=False),
    )
    observations = measure(
        "observation_catalog_seconds",
        lambda: obs.Obs(config, model_state),
    )
    obs_operator = measure(
        "observation_operator_init_seconds",
        lambda: obsop.Obsop(
            config,
            model_state,
            observations,
            model,
            verbose=False,
        ),
    )
    reduced_basis = measure(
        "basis_init_seconds",
        lambda: basis.Basis(config, model_state, verbose=False),
    )
    measure("observation_processing_seconds", obs_operator.process_obs)

    checkpoint_start = time.perf_counter()
    (
        checkpoints,
        time_checkpoints,
        t_checkpoints,
        nstep_check,
    ) = build_checkpoints(config, model, obs_operator, np)
    timings["checkpoint_build_seconds"] = time.perf_counter() - checkpoint_start

    measure(
        "boundary_interpolation_seconds",
        lambda: model.set_bc(time_checkpoints, t_bc=t_checkpoints),
    )
    measure("model_init_seconds", lambda: model.init(model_state))

    time_basis = np.arange(
        0,
        model.T[-1] + nstep_check * model.dt,
        nstep_check * model.dt,
    ) / (24 * 3600)
    background, basis_sigma = measure(
        "basis_build_seconds",
        lambda: reduced_basis.set_basis(
            time_basis,
            return_q=True,
            State=model_state,
        ),
    )

    final_start = time.perf_counter()
    if config.INV.jit_cost_and_grad and hasattr(model_state, "to_device"):
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
    timings["variational_and_device_seconds"] = (
        time.perf_counter() - final_start
    )

    result = {
        "label": args.label,
        "config": config_path.name,
        "window_days": (
            config.EXP.final_date - config.EXP.init_date
        ).total_seconds() / 86400,
        "grid_shape": list(np.shape(model_state.mask)),
        "control_size": int(background.size),
        "checkpoint_count": int(checkpoints.size),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "total_preparation_seconds": time.perf_counter() - total_start,
        "memory_after_preparation": device_memory_snapshot(jax.devices()[0]),
        **timings,
    }
    del variational
    print("VARDYN_PREPARATION_BENCHMARK_JSON=" + json.dumps(result))


if __name__ == "__main__":
    main()
