#!/usr/bin/env python3
"""Export a VarDyn Zarr product to daily NetCDF and benchmark diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib


# Diagnostics call ``plt.show()`` internally.  Use a non-interactive backend
# so the benchmark can run unattended outside a Jupyter frontend.
matplotlib.use("Agg")


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "mapping"))

from src import diag, exp, state  # noqa: E402


def _netcdf_filename(stem: str, timestamp: pd.Timestamp) -> str:
    return (
        f"{stem}_y{timestamp.year}m{timestamp.month:02d}"
        f"d{timestamp.day:02d}h{timestamp.hour:02d}"
        f"m{timestamp.minute:02d}.nc"
    )


def export_netcdf(config, destination: Path, overwrite: bool = False) -> dict:
    """Write one default-style NetCDF file per Zarr timestamp."""
    source = Path(config.EXP.path_save) / f"{config.EXP.name_exp_save}.zarr"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source Zarr archive: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    config_source = Path(config.EXP.path_save) / "config.py"
    if config_source.is_file():
        shutil.copy2(config_source, destination / "config.py")

    started = time.perf_counter()
    written = skipped = 0
    with xr.open_zarr(source, consolidated=False, chunks="auto") as dataset:
        timestamps = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
        if timestamps.has_duplicates:
            raise RuntimeError("Source Zarr archive contains duplicate timestamps")
        for index, timestamp in enumerate(timestamps):
            target = destination / _netcdf_filename(
                config.EXP.name_exp_save, timestamp)
            if target.exists() and not overwrite:
                try:
                    with xr.open_dataset(target) as existing:
                        valid = (
                            existing.sizes.get("time") == 1
                            and pd.Timestamp(existing.time.values[0]) == timestamp
                        )
                    if valid:
                        skipped += 1
                        continue
                except Exception:
                    pass

            temporary = Path(f"{target}.tmp-{os.getpid()}")
            if temporary.exists():
                temporary.unlink()
            record = dataset.isel(time=slice(index, index + 1)).load()
            try:
                record.to_netcdf(temporary, unlimited_dims={"time"})
                with xr.open_dataset(temporary) as candidate:
                    if (candidate.sizes.get("time") != 1
                            or pd.Timestamp(candidate.time.values[0]) != timestamp):
                        raise RuntimeError(f"Invalid temporary NetCDF: {temporary}")
                os.replace(temporary, target)
            finally:
                record.close()
                if temporary.exists():
                    temporary.unlink()
            written += 1
            if written % 10 == 0 or index == len(timestamps) - 1:
                print(
                    f"[NetCDF export] {index + 1}/{len(timestamps)} "
                    f"({written} written, {skipped} reused)",
                    flush=True,
                )

    return {
        "source": str(source),
        "destination": str(destination),
        "timestamps": len(timestamps),
        "written": written,
        "reused": skipped,
        "seconds": time.perf_counter() - started,
    }


def _format_config(base_config, output_path: Path, diag_path: Path,
                   use_zarr: bool):
    config = base_config.copy()
    config.EXP = base_config.EXP.copy()
    config.EXP.path_save = str(output_path)
    config.EXP.saveoutputs_zarr = use_zarr
    config.DIAG = base_config.DIAG.copy()
    config.DIAG.dir_output = str(diag_path)
    return config


def benchmark_diagnostics(config) -> dict:
    """Run and time the same diagnostic sequence used by the notebook."""
    Path(config.DIAG.dir_output).mkdir(parents=True, exist_ok=True)
    timings = {}

    started = time.perf_counter()
    experiment_state = state.State(config, verbose=0)
    diagnostic = diag.Diag(config, experiment_state)
    timings["initialization"] = time.perf_counter() - started

    stages = (
        ("regrid_exp", lambda: diagnostic.regrid_exp()),
        ("rmse_based_scores", lambda: diagnostic.rmse_based_scores(plot=True)),
        ("psd_based_scores", lambda: diagnostic.psd_based_scores(plot=True)),
        ("movie", lambda: diagnostic.movie(Display=False)),
        ("leaderboard", lambda: diagnostic.Leaderboard()),
    )
    for name, operation in stages:
        print(f"[Diagnostics] {name}", flush=True)
        started = time.perf_counter()
        operation()
        timings[name] = time.perf_counter() - started
        print(f"[Diagnostics] {name}: {timings[name]:.3f} s", flush=True)

    timings["total"] = sum(timings.values())
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--netcdf-output", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--overwrite-netcdf", action="store_true")
    args = parser.parse_args()

    base_config = exp.Exp(args.config)
    zarr_output = Path(base_config.EXP.path_save)
    netcdf_output = Path(args.netcdf_output)
    benchmark_root = Path(args.benchmark_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    timings_path = benchmark_root / "timings.json"

    results = {
        "experiment": base_config.EXP.name_experiment,
        "netcdf_export": export_netcdf(
            base_config, netcdf_output, args.overwrite_netcdf),
        "diagnostics": {},
    }
    timings_path.write_text(json.dumps(results, indent=2) + "\n")

    for label, output_path, use_zarr in (
            ("zarr", zarr_output, True),
            ("netcdf", netcdf_output, False)):
        print(f"[Benchmark] diagnostics for {label}", flush=True)
        current_config = _format_config(
            base_config,
            output_path,
            benchmark_root / label,
            use_zarr,
        )
        results["diagnostics"][label] = benchmark_diagnostics(current_config)
        timings_path.write_text(json.dumps(results, indent=2) + "\n")

    zarr_total = results["diagnostics"]["zarr"]["total"]
    netcdf_total = results["diagnostics"]["netcdf"]["total"]
    results["diagnostic_total_ratio_netcdf_over_zarr"] = (
        netcdf_total / zarr_total if zarr_total else np.nan)
    timings_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
