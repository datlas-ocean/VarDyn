#!/usr/bin/env python3
import os
import shutil
import sys
import pickle
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

import xarray as xr
from pathlib import Path

# Add MASSH mapping path — override with the MASSH_PATH environment variable
# Default: resolve relative to this script's location (slurm/ → mapping/)
_MASSH_PATH = os.environ.get('MASSH_PATH', str(Path(__file__).parent.parent / 'mapping'))
sys.path.append(_MASSH_PATH)
from src import exp, state
from src.run_assimilation import (
    merge_output_date,
    generate_dates,
    parallel_merge,
    merge_time_windows_outputs,
)

# -------------------- Logging helpers --------------------
def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

def _convert_subwindow_outputs_to_zarr(config, list_date_start, list_date_middle, list_date_end, output_dtype):
    """Convert spatially merged subwindow NetCDF files and remove originals."""
    for middle, start, end in zip(list_date_middle, list_date_start, list_date_end):
        date = start
        root = Path(config.EXP.path_save) / f"subwindow_{str(middle)[:10]}"
        while date <= end:
            stem = (f"{config.EXP.name_exp_save}_y{date.year}m{date.month:02d}d{date.day:02d}"
                    f"h{date.hour:02d}m{date.minute:02d}")
            nc_path = root / f"{stem}.nc"
            zarr_path = root / f"{stem}.zarr"
            if nc_path.exists():
                with xr.open_dataset(nc_path) as source:
                    dataset = source.load()
                for name in dataset.data_vars:
                    if np.issubdtype(dataset[name].dtype, np.floating):
                        dataset[name] = dataset[name].astype(output_dtype)
                temporary = root / f".{stem}.zarr.tmp-{os.getpid()}"
                if temporary.exists():
                    shutil.rmtree(temporary)
                dataset.to_zarr(temporary, mode="w")
                dataset.close()
                if zarr_path.exists():
                    shutil.rmtree(zarr_path)
                os.replace(temporary, zarr_path)
                nc_path.unlink()
            date += config.EXP.saveoutput_time_step

def _validate_final_outputs(config, list_date_start, list_date_end):
    """Verify that every expected global output exists and is readable."""
    expected = set()
    step = config.EXP.saveoutput_time_step
    for start, end in zip(list_date_start, list_date_end):
        date = start
        while date <= end:
            expected.add(date)
            date += step

    missing = []
    unreadable = []
    for date in sorted(expected):
        path = os.path.join(
            config.EXP.path_save,
            f'{config.EXP.name_exp_save}'
            f'_y{date.year}m{date.month:02d}d{date.day:02d}'
            f'h{date.hour:02d}m{date.minute:02d}.nc',
        )
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            with xr.open_dataset(path) as dataset:
                # Opening the HDF5 container catches truncated/corrupt files.
                _ = dataset.sizes
        except Exception as exc:
            unreadable.append(f'{path}: {exc}')

    if missing or unreadable:
        details = []
        if missing:
            details.append(f'missing={len(missing)} (first: {missing[0]})')
        if unreadable:
            details.append(f'unreadable={len(unreadable)} (first: {unreadable[0]})')
        raise RuntimeError('Final output validation failed: ' + '; '.join(details))

    log(f'Validated {len(expected)} final global output files')

def _validate_dated_outputs(config, dates, root, zarr_output=False):
    """Fail when any expected per-date product is missing or unreadable."""
    missing = []
    unreadable = []
    suffix = '.zarr' if zarr_output else '.nc'
    for date in dates:
        filename = (
            f'{config.EXP.name_exp_save}'
            f'_y{date.year}m{date.month:02d}d{date.day:02d}'
            f'h{date.hour:02d}m{date.minute:02d}{suffix}'
        )
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            opener = xr.open_zarr if zarr_output else xr.open_dataset
            with opener(path) as dataset:
                _ = dataset.sizes
        except Exception as exc:
            unreadable.append(f'{path}: {exc}')

    if missing or unreadable:
        details = []
        if missing:
            details.append(f'missing={len(missing)} (first: {missing[0]})')
        if unreadable:
            details.append(
                f'unreadable={len(unreadable)} (first: {unreadable[0]})')
        raise RuntimeError('Output validation failed: ' + '; '.join(details))

    log(f'Validated {len(dates)} dated outputs in {root}')


# -------------------- Main merge workflow --------------------
def merge_outputs(
    path_config,
    dir_save_pickle=None,
    name_var_save=['sla'],
    num_workers=4,
    plot=False,
    iw_start=0,
    iw_end=None,
    merge_time_windows=True,
    skip_spatial_merge=False,
    force=False,
    rank=0,
    world=1,
    zarr_output=False,
    output_float64=False,
):
    log(f"Loading configuration from {path_config}")
    with open(path_config, "rb") as f:
        config = pickle.load(f)

    if dir_save_pickle is None:
        raise ValueError(
            "dir_save_pickle must be provided (root directory where prepare_VarDyn.py wrote its pickles)"
        )
    path_save_pickle = os.path.join(dir_save_pickle, config.EXP.name_experiment)

    # Load precomputed pickle files
    with open(f'{path_save_pickle}/weights.pkl', "rb") as f:
        data = pickle.load(f)
    weights_space_sum = data['weights_space_sum']
    list_tile_paths = data.get('list_tile_paths')
    # Per-tile weights/interpolators are loaded on-the-fly by merge_output_date

    with open(f'{path_save_pickle}/dates.pkl', "rb") as f:
        list_date_start, list_date_middle, list_date_end = pickle.load(f)
    with open(f'{path_save_pickle}/list_State.pkl', "rb") as f:
        list_State_all = pickle.load(f)

    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)
    output_dtype = np.float64 if output_float64 else np.float32
    # output_dtype is resolved above so every spatial merge uses the same precision.

    # Merge all spatial subwindows per time window
    n_windows = len(list_date_start)

    if iw_end is None or iw_end > n_windows:
        iw_end = n_windows
    
    if iw_start < 0 or iw_start >= iw_end:
        raise ValueError(f"Invalid time window range: [{iw_start}, {iw_end})")
    
    if not skip_spatial_merge:
        log(f"Spatial merge for time windows {iw_start} → {iw_end}")
        
        for iw in range(iw_start, iw_end):
            date_start  = list_date_start[iw]
            date_middle = list_date_middle[iw]
            date_end    = list_date_end[iw]
            State_window = list_State_all[iw]
            
            log(f"Processing time window {iw}: {date_start} → {date_end}")
            
            config0 = config.copy()
            config0.EXP = config0.EXP.copy()
            config0.EXP.tmp_DA_path += f'/subwindow_{str(date_middle)[:10]}'
            config0.EXP.path_save += f'/subwindow_{str(date_middle)[:10]}'
            # Tile archives may use one Zarr store per tile, but the spatial
            # merge must remain one product per date: the time-window merge
            # consumes those date-addressable intermediates (NetCDF first,
            # optionally converted to Zarr below).
            config0.EXP.saveoutputs_zarr = False
            State0 = state.State(config0)
            dates_window = generate_dates(date_start, date_end, config0.EXP.saveoutput_time_step)

            # Shard dates across array tasks
            if world > 1:
                dates_window = dates_window[rank::world]
                log(f"  rank {rank}/{world}: {len(dates_window)} dates assigned")

            if not force:
                expected_outputs = [
                    os.path.join(
                        config0.EXP.path_save,
                        f'{config0.EXP.name_exp_save}'
                        f'_y{date.year}'
                        f'm{str(date.month).zfill(2)}'
                        f'd{str(date.day).zfill(2)}'
                        f'h{str(date.hour).zfill(2)}'
                        f'm{str(date.minute).zfill(2)}.nc'
                    )
                    for date in dates_window
                ]
                if expected_outputs and all(os.path.exists(path) for path in expected_outputs):
                    log(f"Skipping time window {iw}: merged outputs already exist")
                    _validate_dated_outputs(
                        config0, dates_window, config0.EXP.path_save)
                    continue

            parallel_merge(dates_window, State0, State_window, name_var_save, kernel, None, weights_space_sum, None, list_tile_paths=list_tile_paths, num_workers=num_workers, output_dtype=output_dtype)
            _validate_dated_outputs(
                config0, dates_window, config0.EXP.path_save)
    else:
        log("Skipping spatial merge (already done)")

    # Convert spatially merged products before the time-window merge.
    # output_dtype is resolved above so every spatial merge uses the same precision.
    if zarr_output and merge_time_windows:
        log("Converting subwindow NetCDF outputs to Zarr")
        _convert_subwindow_outputs_to_zarr(config, list_date_start, list_date_middle, list_date_end, output_dtype)

    # Merge time windows
    time_overlap = (list_date_end[0] - list_date_start[1]).days if len(list_date_start) > 1 else 0
    if merge_time_windows:
        log("Merging overlapping time windows")
        merge_time_windows_outputs(
            config,
            list_date_start,
            list_date_middle,
            list_date_end,
            time_overlap,
            zarr_output=zarr_output,
            output_dtype=output_dtype,
        )
        final_dates = sorted({
            date
            for start, end in zip(list_date_start, list_date_end)
            for date in generate_dates(
                start, end, config.EXP.saveoutput_time_step)
        })
        _validate_dated_outputs(
            config, final_dates, config.EXP.path_save,
            zarr_output=zarr_output)
    else:
        log("Skipping final time-window merge")

# -------------------- Script entry point --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge DA outputs for an experiment.")
    parser.add_argument("path_config", type=str, help="Path to experiment config pickle")
    parser.add_argument("--dir_save_pickle", type=str, required=True,
                        help="Root directory where prepare_VarDyn.py wrote its pickles (DIR_SAVE_PICKLE)")
    parser.add_argument(
        "--name_var_save",
        type=str,
        default='sla',
        help="Comma-separated list of variables to merge, e.g. sla,ssh,ugos,vgos"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers per time window")

    parser.add_argument(
        "--iw_start",
        type=int,
        default=0,
        help="Start index of time windows to merge (inclusive)"
    )
    parser.add_argument(
        "--iw_end",
        type=int,
        default=None,
        help="End index of time windows to merge (exclusive)"
    )

    parser.add_argument(
        "--merge_time_windows",
        action="store_true",
        help="Merge overlapping time windows (final step)"
    )
    parser.add_argument(
        "--skip_spatial_merge",
        action="store_true",
        help="Skip spatial merge (use when spatial merges are already done)"
    )
    parser.add_argument("--force", action="store_true", help="Force recomputing merged outputs even if they already exist")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this task (for date sharding across SLURM array tasks)")
    parser.add_argument("--world", type=int, default=1, help="Total number of tasks sharing the merge")
    parser.add_argument("--zarr_output", action="store_true", help="Use Zarr for merged outputs and remove intermediate NetCDF files")
    parser.add_argument("--output_float64", action="store_true", help="Save merged floating-point data as float64 (default: float32)")

    args = parser.parse_args()

    name_var_save = [v.strip() for v in args.name_var_save.split(",")]

    log(f"Starting merge with {args.num_workers} workers")
    merge_outputs(
        args.path_config,
        dir_save_pickle=args.dir_save_pickle,
        name_var_save=name_var_save,
        num_workers=args.num_workers,
        iw_start=args.iw_start,
        iw_end=args.iw_end,
        merge_time_windows=args.merge_time_windows,
        skip_spatial_merge=args.skip_spatial_merge,
        force=args.force,
        rank=args.rank,
        world=args.world,
        zarr_output=args.zarr_output,
        output_float64=args.output_float64,
    )
    log("Merge finished successfully")
