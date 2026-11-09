# SW reduced-basis learning

The `VarDyn_Tropics` branch combines `Dev-SW` with the Tropics fixes, including
empty L4 observations in compiled scans. Existing SW review corrections are
preserved. The historical working copy `/home/flo/VarDyn` is not a runtime
source dependency.

## Source layout

`mapping/src/` contains exactly the 13 Python files tracked by local `main`.
Its imports never depend on the optional learning sources. The 17 additional
modules have been moved to `learning_local/`, which is ignored by Git.
They remain available in this working copy; a fresh clone does not include them.
The four local CLI drivers use the optional `learning_local` package, and
learning-specific tests skip when that local package is absent.

The archived `learning_local/covariance.py` is retained as a local copy;
active covariance classes are defined only in the tracked `src.tools` module.
The provenance inventory `sw_learning_source_inventory.json` describes the
historical integration before this move.

- `mapping/src/basis.py`: daily SW forcing, AR(1) covariance, generic/calibrated
  forcing bases and divergent/rotational/compensated-height spatial operators.
- `mapping/src/tools.py`: shared independent covariance operators.
- `learning_local/sw_learning_empirical.py`: MAP1, empirical means, standard
  deviations and temporal statistics; physical RMS normalization.
- `learning_local/sw_reduced_temporal.py`: per-atom window ACF fits and compact
  temporal controls for reduced MAP2.
- `learning_local/sw_joint_control.py`, `sw_joint_optax.py` and
  `sw_learning_joint_forward.py`: joint forcing/physical-parameter controls,
  the existing Optax minimizer and differentiable continuous SW trajectories.
- `learning_local/sw_learning.py`, `sw_learning_moments.py`,
  `sw_learning_windowed.py`, `sw_learning_forward.py`: historical approximate
  EM / posterior-moment learning and fixed-physics trajectories.
- `learning_local/sw_learning_spectral.py`, `sw_learning_host_spectral.py`,
  `sw_learning_cg_audit.py`, `sw_learning_pilot.py`, `sw_learning_profile.py`:
  posterior preconditioning, audits, pilots and GPU resource profiling.
- `learning_local/sw_barotropic_component.py` and `sw_barotropic_joint_forward.py`:
  optional barotropic SSH component and associated trajectory.

Imports use the classes in `basis.py`; the former `sw_daily_forcing.py`,
`sw_forcing_basis.py` and `sw_forcing_calibrated.py` are not required. Removed
BMaux cross-variable correlations and the block-coordinate minimizer stay
removed. Temporal autocorrelation remains available.

## Entry points

Four local drivers are in `scripts/`: `learn_sw_daily.py`,
`learn_sw_joint_physics.py`, `learn_sw_reduced_joint_physics.py` and
`learn_sw_windowed.py`. Each accepts `--help` without importing JAX.
`--config` is required: historical experiment configurations and their GLORYS,
ERA5 and boundary data are not bundled with these sources.

Use `learn_sw_joint_physics.py` for daily MAP1 with active physics, then
`learn_sw_reduced_joint_physics.py` for reduced MAP2 with transferred physics.
For example, with an existing compatible configuration and data:

```bash
nohup env JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
  HDF5_USE_FILE_LOCKING=FALSE XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -u scripts/learn_sw_joint_physics.py \
  --config /absolute/path/config.py --output runs/sw_map1 \
  --use-config-optimizer > runs/sw_map1.log 2>&1 < /dev/null &
echo $! > runs/sw_map1.pid
```

The joint configuration must define `JOINT_PHYSICS`, a `BASIS_SW_DAILY` basis,
matching physical controls and observation/boundary inputs. Use `--prepare-only`
to check configuration/data preparation before optimization. See each driver's
`--help` for MAP1 statistics, reduced MAP2, continuation and optimizer options.
Production GPU checks and detached execution follow `AGENTS.md`.

The source inventory and verification logs for this integration are in
`runs/merge_dev_sw_20261109/`. Small synthetic CPU tests validate the learning
operators and gradients; they do not reproduce annual GLORYS training or an OSE.

The current layout was verified separately: 40 core CPU tests pass with imports
of the relocated modules explicitly blocked (eight optional test files skip),
and 43 local learning tests pass with the local package available. Static import
auditing finds no core dependency on the relocated modules. All 34 source/driver
files compile; all four CLI help commands succeed. Details and the move inventory
are in `runs/src_main_layout_20261109/`.

The root-level `scripts/`, `tests/` and `learning_local/` directories are
ignored by Git. Their files remain on disk in this worktree. The three main
regression files remain tracked under `mapping/tests/`, making the entire
`mapping/` file inventory identical to local main. The root `tests/` directory
contains 15 additional test files. Run both test directories for the complete
local suite: `python -m pytest tests mapping/tests -q`, with both the repository
root and `mapping/` on PYTHONPATH. A fresh clone includes only the three main
regression files, without the local drivers or learning modules.

Root-level relocation verification: all 43 `mapping/` paths match local main
both on disk and in the Git index; mapping file contents were preserved. The
40 core tests and 43 local learning tests pass after relocation, and all four
CLI help commands work from `/tmp`. Logs: `runs/root_local_layout_20261109/`.
