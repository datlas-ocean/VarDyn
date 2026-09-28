# VarDyn SLURM

Scripts to run [VarDyn/MASSH](https://github.com/leguillf/MASSH) SSH mapping experiments on HPC clusters using SLURM GPU arrays. Lives in `slurm/` inside the MASSH repository.

## Overview

Large-scale SSH mapping with MASSH (e.g. global VarDyn runs) is parallelised over:
- **Space** — the domain is split into overlapping spatial tiles
- **Time** — the time period is split into overlapping time windows

Each running SLURM array task (one GPU) processes its deterministic shard of the sorted tile list. For Zarr output, spatial-merge date shards are also dynamically claimed: each active task uses its own CPU allocation to create an independent rank archive. One task then assembles the validated rank archives into the single archive for that temporal window. Finally, one task merges all temporal windows into the full output.

```
sbatch slurm/run/VarDyn_GLO.sh [--skip-prepare] [--restart] [--force-merge] [--tile-scope all|equatorial] [--name_exp <name>]
```

## Repository structure

```
slurm/
├── run/
│   └── VarDyn_GLO.sh       # Example SLURM array job script (copy & edit per experiment)
└── src/
    ├── prepare_VarDyn.py   # Prepare spatial/temporal subwindows and save pickles
    ├── run_tile.py         # Run one assimilation tile (called per-GPU in parallel)
    └── merge_outputs.py    # Merge spatial tiles and time windows into final output
```

> Config files (`.py`) live in a sibling `configs/` directory and are not part of this repo.

## Workflow

```text
prepare_VarDyn.py (one dynamic owner)
                 │
                 ▼
for each temporal window:
  deterministic assimilation-tile shards (one owner per tile)
                 │
                 ▼
  dynamic spatial-merge rank queue
  ├── rank 0 → timestamps[0::N] → rank-0000/<name>.zarr
  ├── rank 1 → timestamps[1::N] → rank-0001/<name>.zarr
  └── ...
                 │
                 ▼
  validate + atomically publish one window <name>.zarr
                 │
                 ▼
merge_time_windows (one task) → final <name>.zarr
```

## Scripts

### `VarDyn_GLO.sh`

Example SLURM submission script — copy and edit the **USER SETTINGS** block for each experiment.

| Variable | Description |
|---|---|
| `MASH_DIR` | **Absolute path to the MASSH repo root** — required because SLURM copies the script to a spool directory before execution, making `$0`-relative paths unreliable |
| `NUM_GPUS` | Number of GPU array tasks (also update `#SBATCH --array`) |
| `NUM_MERGE_WORKERS` | CPU workers used by each active merge rank; normally keep this at or below `--cpus-per-task` |
| `DIR_SAVE_PICKLE` | Root directory for all pickle/output files |
| `PATH_CONFIG` | Path to the main MASSH config `.py` |
| `PATH_CONFIG_EQ` | Path to the equatorial MASSH config `.py` |
| `INIT_DATE` / `FINAL_DATE` | Experiment date range |
| `NAME_VAR` | Comma-separated list of variables to save |
| `GRID_TYPE` / `NX_PROC` … | Spatial subwindow grid parameters |
| `SPACE_WIN_X/Y`, `SPACE_OVERLAP_X/Y` | Spatial window size and overlap (degrees) |
| `TIME_WIN`, `TIME_OVERLAP` | Temporal window size and overlap (days) |
| `FLAG_INIT` / `FLAG_BACKGROUND` / `NAME_EXP` | Initialise from / use background from a previous experiment |
| `NAME_EXP_BACKGROUND` | Source experiment for the inversion background; alternatively pass `--name_exp_background` |
| `BARRIER_TIMEOUT` | Maximum barrier wait or interval without tile completion progress, in seconds (default: 7200) |
| `TILE_TIMEOUT` | Maximum runtime of one tile in seconds (default: 172800); timeout stops its process group |
| `STAGE_TIMEOUT` | Maximum runtime of one preparation/merge command in seconds (default: 172800) |
| `ZARR_OUTPUT` | If this shell option or `EXP.saveoutputs_zarr` is `true`, store each merged temporal window in one Zarr archive and the final experiment in one global Zarr archive |
| `OUTPUT_FLOAT64` | If `true`, save merged floating-point data as float64; otherwise float32 (default: false) |
| `CLEANUP_TILE_ZARR` | If `true` (default), compact validated tile trajectories to the single record needed to restart the following window |
| `ZARR_TIME_CHUNK` | Number of time records per Zarr chunk (default: 4) |
| `ZARR_SPATIAL_CHUNK` | Maximum size of each spatial Zarr chunk dimension (default: 256) |
| `ZARR_COMPRESSION_LEVEL` | Zstd compression level from 0 to 9 (default: 3); bitshuffle is always enabled |

**CLI flags** (passed after the script name):

| Flag | Effect |
|---|---|
| `--skip-prepare` | Skip `prepare_VarDyn.py` if pickles already exist |
| `--restart` | Pass `--restart` to `run_tile.py` (resume from checkpoint) |
| `--force-merge` | Force re-merge even if output files already exist |
| `--merge-only` | Skip preparation and assimilation, only run spatial and time-window merges |
| `--tile-scope all\|equatorial` | Restrict assimilation dispatch to all tiles (default) or only tiles crossing latitude 0; spatial and temporal merges still use every tile output |
| `--name_exp <name>` | Override experiment name (default: read from config or filename) |

**`EXP_NAME` resolution order:**
1. `--name_exp` CLI flag
2. `name_experiment = '...'` variable in `PATH_CONFIG`
3. Config filename with `config_` prefix stripped

**Background controls:** Set `FLAG_BACKGROUND=true` and `NAME_EXP_BACKGROUND`
in the shell config, or pass `--name_exp_background SOURCE` when submitting.
Preparation sets each tile's `INV.path_background` automatically from its
`INV.path_save_control_vectors` root. For a current control root
`/path/controls/CURRENT`, it reads
`/path/controls/SOURCE/subwindow_<date>/<tile>/Xres.nc`.
The main and equatorial Python configs need a `path_save_control_vectors`
root; they do not need `path_background`. `NAME_EXP` remains the legacy
fallback source name when background mode is enabled without
`NAME_EXP_BACKGROUND`.

**Barrier robustness** (Lustre/GPFS):
- Barrier directory creation is retried up to five times with backoff.
- Preparation, queue publication, merge parts, finalization and final merge
  waits stop on a shared failure, a confirmed dead owner, or `BARRIER_TIMEOUT`.
- Assimilation waits reset their deadline only when the number of incomplete
  tiles changes. Pending owners are not replaced by another worker. Set
  `BARRIER_TIMEOUT` above the expected gap between tile completions/startups.
- Slurm queries have a 30-second timeout (plus 5 seconds to force termination).
  A failed targeted query falls back to the full active-task listing, including
  for `--predecessor-job`. Unknown ownership never authorizes stealing a lock.
- A tile failure stops new launches and propagates to other tasks through
  generation-specific `run.failed` and window failure markers. Running tile
  and stage process groups are terminated and reaped before releasing tile
  locks; TERM is followed by KILL after a ten-second grace period if needed.
- Successful tiles remain resumable. Previous `.tile_failed` markers are
  cleared once during queue publication for the next generation, before any
  workers are released. Barriers are retained for diagnostics after completion.
- Deterministic failures do not submit automatic continuations. The Slurm
  wall-time warning still submits a dependent continuation.
- `--force-merge` applies to the submitted run only and is not propagated to
  automatic continuations.

The launcher requires `setsid` and GNU `timeout` on the compute nodes.
All three timeouts can be overridden in the experiment shell config.

Automatic continuations carry the preceding array ID in an internal
`--predecessor-job` argument. The continuation checks `squeue` before doing any
work, providing a runtime guard in addition to Slurm's `afterany` dependency.
Per-tile `.tile_running.lock` leases also prevent two job generations from
running the same tile concurrently. Locks are released when a tile process
exits and are reclaimed only after their owning array is no longer active.

After spatial merging succeeds, each time window receives a durable
`.window_complete_<scope>.ok` marker. Normal continuation runs skip these
windows before launching tile or merge subprocesses. Explicit `--restart` and
`--force-merge` runs bypass the marker.

### `prepare_VarDyn.py`

Reads the MASSH config files and generates the pickle tree under `DIR_SAVE_PICKLE/<EXP_NAME>/`:
```
<EXP_NAME>/
  config.pkl
  subwindow_<date>/
    subwindow_<space>/
      config.pkl
      state.pkl
      weights.pkl
```

### `run_tile.py`

Loads one `subwindow_<space>` pickle directory and runs the full MASSH assimilation (forward + inverse). Writes `Xres.nc` on completion. The atomic `.tile_complete.ok` contains
`COMPLETED` for calculated/reused output or `SKIPPED_LAND` for a tile whose
entire state mask is land. Land tiles intentionally have no trajectory.
The fusion checks the state mask too, so old prepared states and completion
markers remain compatible.

### `merge_outputs.py`

Two-stage merge:
1. **Spatial merge**: distributes timestamp shards over dynamically claimed ranks. Each rank uses `NUM_MERGE_WORKERS` CPU processes and writes an independent temporary Zarr archive; the validated parts are then atomically assembled into one `<name_exp>.zarr` archive for the temporal window.
2. **Time-window merge** (one task only): combines spatial merges across all time windows and validates every expected timestamp.

Spatial fusion excludes all-land tiles from both contributions and weight
normalization. An unreadable ocean trajectory, missing requested variable or
variable read/projection error fails the merge. Other merge workers are stopped
on the first reported error; no window-success marker is published. Existing
merged products are still reused normally: use `--force-merge` to rebuild a
previously generated product after auditing its inputs.

New Zarr stores use explicit Zstd/bitshuffle compression and configurable
time/spatial chunks. After the next window has completed and the current
spatial merge is validated, complete per-tile trajectories from the previous
window are atomically replaced by one-record restart checkpoints. The final
window is compacted after validation of the global archive.

## Usage example

```bash
# Submit with 6 GPUs (array 0-5 in the script)
sbatch VarDyn_GLO.sh

# Skip re-preparation if pickles already exist
sbatch VarDyn_GLO.sh --skip-prepare

# Resume a crashed run
sbatch VarDyn_GLO.sh --skip-prepare --restart

# Re-run only equatorial 4DVar tiles and rebuild merged outputs
sbatch VarDyn_GLO.sh --skip-prepare --restart --force-merge --tile-scope equatorial

# Override experiment name
sbatch VarDyn_GLO.sh --name_exp my_custom_name
```

## Stopping a job without triggering another continuation

`VarDyn_GLO.sh` uses `USR1` exclusively for the warning sent five minutes
before the wall-time limit. Only this signal submits an automatic
continuation. The regular `TERM` signal sent by `scancel` exits without a
continuation.

Stop a job or a complete array with the standard command:

```bash
scancel <JOB_ID>
```

No experiment name, VarDyn path, stop marker, or custom shell function is
required. If a continuation was already visible in `squeue` before the
cancellation, it is an independent Slurm job and its numeric ID must also be
cancelled:

```bash
scancel <JOB_ID> <CONTINUATION_JOB_ID>
```

## Requirements

- SLURM with GPU support (`--gpus=v100_32g:1` or similar)
- MASSH repository — `mapping/` is located automatically relative to `slurm/` (no `MASSH_PATH` env var needed when running from within the repo)
- Python environment with: `numpy`, `xarray`, `scipy`, `astropy`, `jax`, `cartopy`
- Lustre/GPFS shared filesystem (barrier mechanism uses atomic `mkdir`)

## Notes

- Tile claiming uses `mkdir` (atomic on all POSIX filesystems including Lustre/GPFS) — no NFS locking required.
- Set `HDF5_USE_FILE_LOCKING=FALSE` if you encounter NetCDF read errors on shared filesystems (already handled inside `run_assimilation.py`).
- Logs are written to `./logs/<EXP_NAME>_job-<JOB_ID>/gpu<ARRAY_ID>.log` and per-tile under subdirectories.
