#!/usr/bin/env python3
"""Replay optimized VarDyn controls and write a complete NetCDF product.

This rebuilds every spatial-tile trajectory from its existing ``Xres.nc``
without running a new minimization, merges each temporal subwindow, then
merges all temporal subwindows into the final daily NetCDF files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "mapping"))

from src import exp, inv, state  # noqa: E402
from src.run_assimilation import (  # noqa: E402
    generate_dates,
    merge_time_windows_outputs,
    prepare_process,
    run_assimilation_time_window,
)


EXPERIMENT = "VarDyn-QG_UpperDyn_nadirs-minus-al_2024"
INIT_DATE = datetime(2024, 1, 1)
FINAL_DATE = datetime(2025, 1, 1)
NAME_VAR_SAVE = ["SSH_tot", "ug", "vg", "ssh", "sla"]


def _tile_netcdf_complete(tile_state, dates) -> bool:
    for date in dates:
        filename = (
            f"{tile_state.name_exp_save}_y{date.year}m{date.month:02d}"
            f"d{date.day:02d}h{date.hour:02d}m{date.minute:02d}.nc"
        )
        if not (Path(tile_state.path_save) / filename).is_file():
            return False
    return True


def _configure_outputs(config, output_root: Path, control_root: Path,
                       source_scratch: Path) -> None:
    config.EXP.name_experiment = EXPERIMENT
    config.EXP.name_exp_save = EXPERIMENT
    config.EXP.path_save = str(output_root)
    config.EXP.saveoutputs_zarr = False
    # Keep the original scratch tree: it contains the observation/operator
    # caches used by the optimized run.
    config.EXP.tmp_DA_path = str(source_scratch)
    config.EXP.compute_obs = False
    if config.OBSOP is not None:
        if "super" in config.OBSOP:
            config.OBSOP.compute_op = False
        else:
            for name in config.OBSOP:
                config.OBSOP[name].compute_op = False
    config.INV.path_save_control_vectors = str(control_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-eq", required=True)
    parser.add_argument("--source-controls", required=True)
    parser.add_argument("--source-scratch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-controls", required=True)
    parser.add_argument("--timings", required=True)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--lat-min", type=float)
    args = parser.parse_args()

    source_controls = Path(args.source_controls).resolve()
    source_scratch = Path(args.source_scratch).resolve()
    output_root = Path(args.output).resolve()
    replay_controls = Path(args.replay_controls).resolve()
    timings_path = Path(args.timings).resolve()
    gpu_devices = [item.strip() for item in args.gpu_devices.split(",")]

    output_root.mkdir(parents=True, exist_ok=True)
    replay_controls.mkdir(parents=True, exist_ok=True)
    timings_path.parent.mkdir(parents=True, exist_ok=True)

    started_total = time.perf_counter()
    config = exp.Exp(args.config)
    config_eq = exp.Exp(args.config_eq)
    _configure_outputs(config, output_root, replay_controls, source_scratch)
    _configure_outputs(config_eq, output_root, replay_controls, source_scratch)
    if args.lat_min is not None:
        config.GRID.lat_min = args.lat_min
    # Spatial bounds must match the archived experiment configuration.
    for key in ("lon_min", "lon_max", "lat_min", "lat_max"):
        config_eq.GRID[key] = config.GRID[key]

    global_state = state.State(config)
    prepared_started = time.perf_counter()
    (ignored_processes, list_config, list_state,
     list_date_start, list_date_end, list_date_middle, ignored_lonlat,
     weights_space, weights_space_sum, interpolators) = prepare_process(
        config, config_eq, global_state,
        INIT_DATE, FINAL_DATE,
        grid_type="GRID_CAR",
        nx_proc=512, ny_proc=256, dx=10, dy=10,
        time_window_size_proc=50,
        space_window_size_proc_x=50,
        space_window_size_proc_y=25,
        time_overlap=10,
        space_overlap_x=2.5,
        space_overlap_y=2.5,
        flag_init_from_previous=True,
        flag_init=False,
        flag_background=False,
        flag_assim=True,
        flag_assim_restart=False,
        gpu_devices=gpu_devices,
    )
    del ignored_processes, ignored_lonlat

    results = {
        "experiment": EXPERIMENT,
        "mode": "forward replay from optimized Xres; no minimization",
        "output_format": "netcdf",
        "source_controls": str(source_controls),
        "output": str(output_root),
        "preparation_seconds": time.perf_counter() - prepared_started,
        "windows": [],
    }
    timings_path.write_text(json.dumps(results, indent=2) + "\n")

    for index, (date_start, date_middle, date_end) in enumerate(zip(
            list_date_start, list_date_middle, list_date_end)):
        dates = generate_dates(
            date_start, date_end, config.EXP.saveoutput_time_step)
        processes = []
        reused_tiles = 0
        for tile_config, tile_state in zip(
                list_config[index], list_state[index]):
            if _tile_netcdf_complete(tile_state, dates):
                reused_tiles += 1
                continue

            replay_config = tile_config.copy()
            replay_config.EXP = tile_config.EXP.copy()
            replay_config.INV = tile_config.INV.copy()
            relative_control_path = Path(
                tile_config.INV.path_save_control_vectors).relative_to(
                    replay_controls)
            xres_path = source_controls / relative_control_path / "Xres.nc"
            if not xres_path.is_file():
                raise FileNotFoundError(
                    f"Missing optimized control for tile replay: {xres_path}")

            replay_config.INV.path_init_4Dvar = str(xres_path)
            replay_config.INV.restart_4Dvar = False
            replay_config.INV.maxiter = 0
            replay_config.INV.save_minimization = False
            processes.append(partial(
                inv.Inv_4Dvar,
                config=replay_config,
                State=tile_state,
                verbose=0,
            ))

        print(
            f"[NetCDF replay] window {index + 1}/{len(list_date_start)}: "
            f"{len(processes)} tiles to replay, {reused_tiles} complete tiles reused",
            flush=True,
        )
        window_started = time.perf_counter()
        merged = run_assimilation_time_window(
            config, date_start, date_middle, date_end,
            list_state[index], processes,
            name_var_save=NAME_VAR_SAVE,
            weights_space=weights_space,
            weights_space_sum=weights_space_sum,
            interpolators=interpolators,
            flag_assim=True,
            flag_merge_outputs=True,
            flag_diag=False,
            flag_overwrite_outputs=True,
            nprocs=args.nprocs,
            gpu_devices=gpu_devices,
            cleanup_tile_zarr=False,
        )
        if not merged:
            raise RuntimeError(
                f"NetCDF merge failed for temporal window {date_middle:%Y-%m-%d}")
        results["windows"].append({
            "middle": date_middle.isoformat(),
            "tiles_replayed": len(processes),
            "tiles_reused": reused_tiles,
            "seconds": time.perf_counter() - window_started,
        })
        timings_path.write_text(json.dumps(results, indent=2) + "\n")

    final_merge_started = time.perf_counter()
    merge_time_windows_outputs(
        config,
        list_date_start,
        list_date_middle,
        list_date_end,
        time_overlap=10,
        zarr_output=False,
    )
    results["final_merge_seconds"] = time.perf_counter() - final_merge_started
    results["total_seconds"] = time.perf_counter() - started_total
    results["status"] = "complete"
    timings_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
