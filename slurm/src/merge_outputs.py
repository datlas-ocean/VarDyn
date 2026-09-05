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
import pandas as pd
import zarr
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

def _zarr_archive_path(config, root):
    return Path(root) / f"{config.EXP.name_exp_save}.zarr"


def _zarr_parts_root(config, root, world):
    name = config.EXP.name_exp_save
    return Path(root) / f'.{name}.spatial-parts-{world}'


def _zarr_part_root(config, root, rank, world):
    return _zarr_parts_root(config, root, world) / f'rank-{rank:04d}'



def _consolidate_zarr_metadata(archive_path):
    """Publish an up-to-date consolidated view after all appends."""
    zarr.consolidate_metadata(str(archive_path))


def _remove_legacy_dated_outputs(config, dates, root):
    """Remove former one-file-per-date products after archive validation."""
    root = Path(root)
    for date in dates:
        stem = (
            f"{config.EXP.name_exp_save}_y{date.year}m{date.month:02d}"
            f"d{date.day:02d}h{date.hour:02d}m{date.minute:02d}")
        nc_path = root / f"{stem}.nc"
        dated_zarr_path = root / f"{stem}.zarr"
        if dated_zarr_path.exists():
            shutil.rmtree(dated_zarr_path)
        if nc_path.exists():
            nc_path.unlink()


def _remove_subwindow_merged_outputs(
        config, list_date_start, list_date_middle, list_date_end):
    """Remove validated spatial-merge products without touching tile output.

    A subwindow directory also contains one directory per tile.  Those tile
    trajectories are deliberately retained as restart checkpoints, so cleanup
    is limited to the archive and dated files at the subwindow root.
    """
    archives_removed = 0
    for start, middle, end in zip(
            list_date_start, list_date_middle, list_date_end):
        root = Path(config.EXP.path_save) / f'subwindow_{str(middle)[:10]}'
        dates = generate_dates(start, end, config.EXP.saveoutput_time_step)
        _remove_legacy_dated_outputs(config, dates, root)
        archive = _zarr_archive_path(config, root)
        if archive.exists():
            shutil.rmtree(archive)
            archives_removed += 1
        lock = Path(f'{archive}.lock')
        if lock.exists():
            lock.unlink()
    log(
        f'Removed merged outputs from {len(list_date_middle)} subwindows '
        f'({archives_removed} Zarr archives); tile checkpoints retained')


def _compact_tile_zarr_trajectories(
        config, window_middle, checkpoint_date, list_tile_paths):
    """Replace each tile trajectory with its one-record restart checkpoint.

    Spatial merging needs the complete trajectory, while the following time
    window only needs its start date. Compaction therefore happens only after
    the spatial window has been validated and its successor has completed.
    """
    if not list_tile_paths:
        log('No tile paths available; skipping tile trajectory compaction')
        return

    window_root = (
        Path(config.EXP.path_save) /
        f'subwindow_{str(window_middle)[:10]}')
    checkpoint_time = pd.Timestamp(checkpoint_date)
    compacted = 0
    already_compact = 0
    missing = 0

    for tile_pickle_path in list_tile_paths:
        tile_name = Path(tile_pickle_path).name
        tile_root = window_root / tile_name
        archive = tile_root / f'{config.EXP.name_exp_save}.zarr'
        if not archive.exists():
            # All-land tiles legitimately produce no trajectory.
            missing += 1
            continue

        temporary = tile_root / (
            f'.{config.EXP.name_exp_save}.checkpoint-{os.getpid()}.zarr')
        temporary_lock = Path(f'{temporary}.lock')
        backup = tile_root / (
            f'.{config.EXP.name_exp_save}.trajectory-{os.getpid()}.zarr')
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists():
            shutil.rmtree(backup)

        with xr.open_zarr(archive, consolidated=False) as dataset:
            times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
            indexes = np.flatnonzero(times == checkpoint_time)
            if indexes.size == 0:
                raise RuntimeError(
                    f'Tile restart checkpoint {checkpoint_time} is missing '
                    f'from {archive}')
            if dataset.sizes.get('time', 0) == 1:
                already_compact += 1
                continue
            record = dataset.isel(
                time=slice(int(indexes[-1]), int(indexes[-1]) + 1)).load()

        try:
            state.State._save_zarr_record(
                record, str(temporary), checkpoint_time,
                window_start=checkpoint_time, window_end=checkpoint_time)
            with xr.open_zarr(temporary, consolidated=False) as checkpoint:
                checkpoint_times = pd.DatetimeIndex(
                    pd.to_datetime(checkpoint.time.values))
                if (checkpoint.sizes.get('time', 0) != 1
                        or checkpoint_times[0] != checkpoint_time):
                    raise RuntimeError(
                        f'Invalid compacted tile checkpoint: {temporary}')
            _consolidate_zarr_metadata(temporary)

            os.replace(archive, backup)
            try:
                os.replace(temporary, archive)
            except Exception:
                if not archive.exists():
                    os.replace(backup, archive)
                raise
            else:
                shutil.rmtree(backup)
                compacted += 1
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            if temporary_lock.exists():
                temporary_lock.unlink()
            if backup.exists() and archive.exists():
                shutil.rmtree(backup)

    log(
        f'Compacted tile trajectories for window {window_middle}: '
        f'{compacted} compacted, {already_compact} already compact, '
        f'{missing} absent/all-land')


def _consolidate_dated_outputs_to_zarr(
        config, dates, root, output_dtype, window_start, window_end):
    """Move legacy per-date products into one time-window Zarr archive."""
    root = Path(root)
    archive_path = _zarr_archive_path(config, root)
    for date in dates:
        stem = (f"{config.EXP.name_exp_save}_y{date.year}m{date.month:02d}d{date.day:02d}"
                f"h{date.hour:02d}m{date.minute:02d}")
        nc_path = root / f"{stem}.nc"
        dated_zarr_path = root / f"{stem}.zarr"
        source_path = dated_zarr_path if dated_zarr_path.exists() else nc_path
        if not source_path.exists():
            continue
        if source_path == dated_zarr_path:
            context = xr.open_zarr(source_path, consolidated=False)
        else:
            context = xr.open_dataset(source_path)
        with context as source:
            record = source.load()
        if 'time' in record.dims and record.sizes['time'] > 1:
            selected = record.sel(time=pd.Timestamp(date))
            if 'time' in selected.dims:
                selected = selected.isel(time=-1)
            record = selected
        if 'time' in record.coords and 'time' not in record.dims:
            record = record.drop_vars('time')
        if 'time' not in record.dims:
            record = record.expand_dims(time=[pd.Timestamp(date)])
        else:
            record = record.assign_coords(time=[pd.Timestamp(date)])
        for name in record.data_vars:
            if np.issubdtype(record[name].dtype, np.floating):
                record[name] = record[name].astype(output_dtype)
        state.State._save_zarr_record(
            record, str(archive_path), date,
            window_start=window_start, window_end=window_end)
        if dated_zarr_path.exists():
            shutil.rmtree(dated_zarr_path)
        if nc_path.exists():
            nc_path.unlink()


def _convert_subwindow_outputs_to_zarr(
        config, list_date_start, list_date_middle, list_date_end,
        output_dtype):
    """Consolidate legacy files into one Zarr archive per time window."""
    for middle, start, end in zip(
            list_date_middle, list_date_start, list_date_end):
        root = Path(config.EXP.path_save) / f"subwindow_{str(middle)[:10]}"
        dates = generate_dates(start, end, config.EXP.saveoutput_time_step)
        _consolidate_dated_outputs_to_zarr(
            config, dates, root, output_dtype, start, end)

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

def _validate_dated_outputs(
        config, dates, root, zarr_output=False, consolidate_zarr=True):
    """Fail when expected dates are missing or outputs are unreadable."""
    if zarr_output:
        archive_path = _zarr_archive_path(config, root)
        if not archive_path.exists():
            raise RuntimeError(f'Output archive is missing: {archive_path}')
        try:
            # Appends update the live Zarr metadata first. Reading a stale
            # consolidated view here can falsely report missing timestamps.
            with xr.open_zarr(
                    archive_path, consolidated=False) as dataset:
                actual_times = pd.DatetimeIndex(
                    pd.to_datetime(dataset.time.values))
                _ = dataset.sizes
        except Exception as exc:
            raise RuntimeError(
                f'Output archive is unreadable: {archive_path}: {exc}') from exc
        duplicate_times = actual_times[actual_times.duplicated()].unique()
        if len(duplicate_times):
            raise RuntimeError(
                f'Output archive contains duplicate timestamps: '
                f'{archive_path} (first: {duplicate_times[0]})')
        expected_times = pd.DatetimeIndex(pd.to_datetime(dates))
        missing_times = expected_times.difference(actual_times)
        if len(missing_times):
            raise RuntimeError(
                f'Output archive is missing {len(missing_times)} timestamps: '
                f'{archive_path} (first: {missing_times[0]})')
        if consolidate_zarr:
            _consolidate_zarr_metadata(archive_path)
        log(
            f'Validated {len(expected_times)} timestamps in Zarr archive '
            f'{archive_path}')
        return

    missing = []
    unreadable = []
    for date in dates:
        filename = (
            f'{config.EXP.name_exp_save}'
            f'_y{date.year}m{date.month:02d}d{date.day:02d}'
            f'h{date.hour:02d}m{date.minute:02d}.nc'
        )
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            with xr.open_dataset(path) as dataset:
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



def _finalize_spatial_zarr_parts(
        config, dates, root, rank_count, output_dtype, force=False):
    """Assemble rank archives into one atomically published window archive."""
    root = Path(root)
    canonical_archive = _zarr_archive_path(config, root)

    if canonical_archive.exists() and not force:
        try:
            _validate_dated_outputs(
                config, dates, root, zarr_output=True)
        except RuntimeError as exc:
            log(f'Canonical window archive is incomplete: {exc}')
        else:
            parts_root = _zarr_parts_root(config, root, rank_count)
            if parts_root.exists():
                shutil.rmtree(parts_root)
            log(f'Canonical window archive is already complete: {canonical_archive}')
            return

    part_datasets = {}
    combined = None
    temporary_root = None
    try:
        for rank in range(rank_count):
            rank_dates = dates[rank::rank_count]
            if not rank_dates:
                continue
            part_root = _zarr_part_root(
                config, root, rank, rank_count)
            _validate_dated_outputs(
                config, rank_dates, part_root, zarr_output=True)
            part_archive = _zarr_archive_path(config, part_root)
            part_datasets[rank] = xr.open_zarr(
                part_archive, consolidated=False)

        temporary_root = root / (
            f'.{config.EXP.name_exp_save}.spatial-finalize-{os.getpid()}')
        temporary_archive = _zarr_archive_path(config, temporary_root)
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        temporary_root.mkdir(parents=True)

        # Keep rank arrays lazy and publish all temporal chunks in one write.
        # The former record loop reopened metadata and rewrote partially filled
        # time chunks hundreds of times on Lustre.
        combined = xr.concat(
            [part_datasets[rank] for rank in sorted(part_datasets)],
            dim='time',
            data_vars='all',
            coords='minimal',
            compat='override',
            join='outer',
        ).sortby('time')
        actual_times = pd.DatetimeIndex(pd.to_datetime(combined.time.values))
        duplicate_times = actual_times[actual_times.duplicated()].unique()
        if len(duplicate_times):
            raise RuntimeError(
                'Spatial rank parts contain duplicate timestamps '
                f'(first: {duplicate_times[0]})')
        expected_times = pd.DatetimeIndex(pd.to_datetime(dates))
        missing_times = expected_times.difference(actual_times)
        unexpected_times = actual_times.difference(expected_times)
        if len(missing_times) or len(unexpected_times):
            raise RuntimeError(
                'Spatial rank timestamps do not match the Analysis Window: '
                f'missing={len(missing_times)}, '
                f'unexpected={len(unexpected_times)}')
        combined = combined.sel(time=expected_times)
        for name in combined.data_vars:
            if np.issubdtype(combined[name].dtype, np.floating):
                combined[name] = combined[name].astype(output_dtype)
        state._set_zarr_time_encoding(combined)
        state._write_new_zarr(combined, str(temporary_archive))
        combined.close()
        combined = None

        _validate_dated_outputs(
            config, dates, temporary_root, zarr_output=True)

        backup_archive = root / (
            f'.{config.EXP.name_exp_save}.zarr.backup-{os.getpid()}')
        if backup_archive.exists():
            shutil.rmtree(backup_archive)
        had_canonical = canonical_archive.exists()
        if had_canonical:
            os.replace(canonical_archive, backup_archive)
        try:
            os.replace(temporary_archive, canonical_archive)
        except Exception:
            if had_canonical and not canonical_archive.exists():
                os.replace(backup_archive, canonical_archive)
            raise
        else:
            if backup_archive.exists():
                shutil.rmtree(backup_archive)
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)

        _validate_dated_outputs(
            config, dates, root, zarr_output=True)
        _remove_legacy_dated_outputs(config, dates, root)
        for dataset in part_datasets.values():
            dataset.close()
        part_datasets.clear()
        parts_root = _zarr_parts_root(config, root, rank_count)
        if parts_root.exists():
            shutil.rmtree(parts_root)
        log(
            f'Published spatial window archive from {rank_count} rank parts: '
            f'{canonical_archive}')
    finally:
        if combined is not None:
            combined.close()
        for dataset in part_datasets.values():
            dataset.close()
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)

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
    zarr_parts=False,
    finalize_spatial_parts=False,
    zarr_output=False,
    output_float64=False,
    cleanup_tile_zarr=False,
    cleanup_subwindow_outputs=False,
):
    log(f"Loading configuration from {path_config}")
    with open(path_config, "rb") as f:
        config = pickle.load(f)

    config_requests_zarr = bool(
        getattr(config.EXP, 'saveoutputs_zarr', False))
    zarr_output = zarr_output or config_requests_zarr
    log(f"Merged output format: {'Zarr' if zarr_output else 'NetCDF'}")

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

    if cleanup_subwindow_outputs:
        if merge_time_windows or not skip_spatial_merge:
            raise ValueError(
                '--cleanup_subwindow_outputs must be used with '
                '--skip_spatial_merge and without --merge_time_windows')
        final_dates = sorted({
            date
            for start, end in zip(list_date_start, list_date_end)
            for date in generate_dates(
                start, end, config.EXP.saveoutput_time_step)
        })
        # This mode is invoked only after the shell has atomically published
        # experiment_complete.ok.  Revalidate the durable global product
        # before performing the destructive, idempotent cleanup.
        _validate_dated_outputs(
            config, final_dates, config.EXP.path_save,
            zarr_output=zarr_output)
        _remove_subwindow_merged_outputs(
            config, list_date_start, list_date_middle, list_date_end)
        return

    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)
    output_dtype = np.float64 if output_float64 else np.float32
    # output_dtype is resolved above so every spatial merge uses the same precision.

    # Merge all spatial subwindows per time window
    n_windows = len(list_date_start)

    if iw_end is None or iw_end > n_windows:
        iw_end = n_windows
    
    if iw_start < 0 or iw_start >= iw_end:
        raise ValueError(f"Invalid time window range: [{iw_start}, {iw_end})")
    
    if (zarr_parts or finalize_spatial_parts) and not zarr_output:
        raise ValueError('Distributed Zarr merge requires --zarr_output')
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(f'Invalid merge rank/world: {rank}/{world}')

    if not skip_spatial_merge:
        log(f"Spatial merge for time windows {iw_start} → {iw_end}")

        for iw in range(iw_start, iw_end):
            date_start = list_date_start[iw]
            date_middle = list_date_middle[iw]
            date_end = list_date_end[iw]
            State_window = list_State_all[iw]

            log(f"Processing time window {iw}: {date_start} → {date_end}")

            config0 = config.copy()
            config0.EXP = config0.EXP.copy()
            config0.EXP.tmp_DA_path += f'/subwindow_{str(date_middle)[:10]}'
            config0.EXP.path_save += f'/subwindow_{str(date_middle)[:10]}'
            config0.EXP.init_date = date_start
            config0.EXP.final_date = date_end
            config0.EXP.saveoutputs_zarr = zarr_output
            canonical_root = config0.EXP.path_save
            all_window_dates = generate_dates(
                date_start, date_end, config0.EXP.saveoutput_time_step)

            if finalize_spatial_parts:
                _finalize_spatial_zarr_parts(
                    config0, all_window_dates, canonical_root, world,
                    output_dtype, force=force)
                if cleanup_tile_zarr and iw > 0:
                    _compact_tile_zarr_trajectories(
                        config,
                        list_date_middle[iw - 1],
                        list_date_start[iw],
                        list_tile_paths,
                    )
                continue

            dates_window = all_window_dates
            if zarr_parts:
                canonical_archive = _zarr_archive_path(
                    config0, canonical_root)
                if canonical_archive.exists() and not force:
                    try:
                        _validate_dated_outputs(
                            config0, all_window_dates, canonical_root,
                            zarr_output=True,
                            consolidate_zarr=False)
                    except RuntimeError as exc:
                        log(f"Time window {iw} is incomplete: {exc}")
                    else:
                        log(
                            f"Skipping rank {rank}: canonical Zarr archive "
                            f"for time window {iw} is complete")
                        continue

                part_root = _zarr_part_root(
                    config0, canonical_root, rank, world)
                if force and part_root.exists():
                    shutil.rmtree(part_root)
                config0.EXP.path_save = str(part_root)
                dates_window = all_window_dates[rank::world]
                log(
                    f"  Zarr part rank {rank}/{world}: "
                    f"{len(dates_window)} dates assigned")
            elif world > 1:
                dates_window = all_window_dates[rank::world]
                log(f"  rank {rank}/{world}: {len(dates_window)} dates assigned")

            if not dates_window:
                log(f"  rank {rank}/{world}: no date assigned")
                continue

            State0 = state.State(config0)

            if not force:
                if zarr_output:
                    archive_path = _zarr_archive_path(
                        config0, config0.EXP.path_save)
                    if archive_path.exists():
                        try:
                            _validate_dated_outputs(
                                config0, dates_window,
                                config0.EXP.path_save, zarr_output=True)
                        except RuntimeError as exc:
                            log(f"Time window {iw} is incomplete: {exc}")
                        else:
                            log(
                                f"Skipping time window {iw}: "
                                f"Zarr archive is complete")
                            if not zarr_parts:
                                _remove_legacy_dated_outputs(
                                    config0, dates_window,
                                    config0.EXP.path_save)
                            continue
                else:
                    expected_outputs = [
                        os.path.join(
                            config0.EXP.path_save,
                            f'{config0.EXP.name_exp_save}'
                            f'_y{date.year}m{date.month:02d}d{date.day:02d}'
                            f'h{date.hour:02d}m{date.minute:02d}.nc')
                        for date in dates_window
                    ]
                    if expected_outputs and all(
                            os.path.exists(path) for path in expected_outputs):
                        log(
                            f"Skipping time window {iw}: "
                            f"merged outputs already exist")
                        _validate_dated_outputs(
                            config0, dates_window, config0.EXP.path_save)
                        continue

            parallel_merge(
                dates_window, State0, State_window, name_var_save, kernel,
                None, weights_space_sum, None,
                list_tile_paths=list_tile_paths,
                num_workers=num_workers, output_dtype=output_dtype)
            _validate_dated_outputs(
                config0, dates_window, config0.EXP.path_save,
                zarr_output=zarr_output)
            if zarr_output and not zarr_parts:
                _remove_legacy_dated_outputs(
                    config0, dates_window, config0.EXP.path_save)
    else:
        log("Skipping spatial merge (already done)")

    # Convert spatially merged products before the time-window merge.
    # output_dtype is resolved above so every spatial merge uses the same precision.
    if zarr_output and merge_time_windows:
        log("Consolidating legacy outputs into one Zarr archive per time window")
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
        if zarr_output:
            _remove_legacy_dated_outputs(
                config, final_dates, config.EXP.path_save)
            if cleanup_tile_zarr:
                # Make the operation idempotent and cover merge-only/recovery
                # runs where progressive compaction may not have happened.
                for iw, middle in enumerate(list_date_middle):
                    checkpoint_date = (
                        list_date_start[iw + 1]
                        if iw + 1 < len(list_date_start)
                        else list_date_end[iw])
                    _compact_tile_zarr_trajectories(
                        config, middle, checkpoint_date, list_tile_paths)
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
    parser.add_argument(
        "--zarr_parts",
        action="store_true",
        help="Write this rank's date shard to an independent Zarr archive",
    )
    parser.add_argument(
        "--finalize_spatial_parts",
        action="store_true",
        help="Assemble all spatial rank archives into the window archive",
    )
    parser.add_argument("--zarr_output", action="store_true", help="Use Zarr for merged outputs and remove intermediate NetCDF files")
    parser.add_argument("--output_float64", action="store_true", help="Save merged floating-point data as float64 (default: float32)")
    parser.add_argument(
        "--cleanup_tile_zarr",
        action="store_true",
        help=("After validated merges, replace complete tile Zarr trajectories "
              "with one-record restart checkpoints"),
    )
    parser.add_argument(
        "--cleanup_subwindow_outputs",
        action="store_true",
        help=("After final completion is published, validate the global "
              "output and remove root-level subwindow merge products"),
    )

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
        zarr_parts=args.zarr_parts,
        finalize_spatial_parts=args.finalize_spatial_parts,
        zarr_output=args.zarr_output,
        output_float64=args.output_float64,
        cleanup_tile_zarr=args.cleanup_tile_zarr,
        cleanup_subwindow_outputs=args.cleanup_subwindow_outputs,
    )
    log("Merge finished successfully")
