# WHIRLS VarDyn examples

This directory contains two VarDyn 4DVar configuration templates for the WHIRLS
Agulhas-region experiments:

- `config_VarDyn-QG.py`: quasi-geostrophic setup using `MOD_QG1L`.
- `config_VarDyn-SW.py`: shallow-water setup using the `MOD_QGSW` wrapper with
  the `sw` model core.

The two configurations use the same regional domain, analysis window,
observations, and observation operators. They mainly differ in the dynamical
model and in the controlled variables represented by the reduced basis.

## Associated notebooks

- `run_VarDyn-QG.ipynb` runs the QG configuration.
- `run_VarDyn-SW.ipynb` runs the SW configuration.

Use the notebook matching the configuration you want to test. The notebooks are
intended as runnable examples for preparing the experiment, launching the
assimilation, and inspecting the saved outputs.

## Configuration summary

### `config_VarDyn-QG.py`

The QG experiment uses a 1.5-layer quasi-geostrophic model with streamfunction
formulation:

- model block: `QG`
- model default: `MOD_QG1L`
- prognostic/controlled fields include SSH anomaly as `sla` and SST as `sst`
- external boundary conditions are read for SSH and SST
- reduced basis blocks control large-scale SSH, small-scale balanced SSH, and
  fast/slow/small-scale SST corrections

This configuration is useful when the target experiment should remain close to
the balanced QG dynamics used in the MASSH-derived workflow.

### `config_VarDyn-SW.py`

The SW experiment uses the shallow-water core through the QG/SW model wrapper:

- model block: `SW`
- model default: `MOD_QGSW`
- model core: `name_class = 'sw'`
- prognostic/controlled fields include SSH anomaly as `sla`, C-grid velocities `u` and `v`,
  SST as `sst`, and equivalent-depth corrections `H`
- DUACS boundary conditions provide SSH and geostrophic velocities, while SST is
  read from a separate product
- reduced basis blocks control SSH and SST as in QG, plus spatial and offset
  controls for `H`

This configuration is useful when testing the shallow-water dynamics and the
impact of correcting the equivalent-depth parameter.

## Paths to update before running

Both configuration files are templates. Update the paths below to match your
machine, storage layout, and data access.

### Local experiment directories

In both configs:

- `myPath`: root directory used for relative outputs and cache files.
- `EXP['path_save']`: derived from `myPath`; output NetCDF directory.
- `EXP['tmp_DA_path']`: derived from `myPath`; scratch, restart, and cached
  assimilation files.
- `EXP['path_obs']`: derived from `myPath`; cached preprocessed observations.
- `OBSOP_SSH['path_save']` and `OBSOP_SST['path_save']`: derived from `myPath`;
  cached observation operators.
- `myINV['path_save_control_vectors']`: derived from `myPath`; saved 4DVar
  control vectors.

If you only want all generated files under a different local root, changing
`myPath` is usually enough.

### Static auxiliary files

In both configs:

- `path_mdt`: MDT file shared by the grid mask and the QG/SW model blocks.
- `myGRID["name_init_mask"]`: derived from `path_mdt`; MDT file used to define the land/sea mask.
- `QG["path_mdt"]` and `SW["path_mdt"]`: derived from `path_mdt`; MDT file used by the models. In SW, `name_var_mdt` also maps `mdu` and `mdv` via `var_u` and `var_v`.
- `QG["filec_aux"]` or `SW["filec_aux"]`: first baroclinic phase-speed file.
- `SmallScales_SSH["file_aux"]`: balanced-motion reduced-basis auxiliary file.
- `SmallScales_SST["file_aux"]`: balanced-motion reduced-basis auxiliary file.

The relative `../../aux_files/...` files point to the repository auxiliary directory
when running from `mapping/examples/WHIRLS`. Change them if you launch from a
different working directory or keep auxiliary files elsewhere.

### Boundary-condition products

In `config_VarDyn-QG.py`:

- `QG['bc_files']['SSH']['file']`: DUACS L4 SSH boundary-condition files.
- `QG['bc_files']['SST']['file']`: L4 SST boundary-condition files.

In `config_VarDyn-SW.py`:

- `SW['bc_files']['DUACS']['file']`: DUACS L4 SSH and geostrophic-velocity
  boundary-condition files.
- `SW['bc_files']['SST']['file']`: L4 SST boundary-condition files.

### Observation products

In both configs:

- `SST_MW['path']`: microwave SST observation files.
- `AL['path']`: AltiKa along-track SSH files.
- `C2N['path']`: CryoSat-2/Nadir along-track SSH files.
- `H2B['path']`: HY-2B along-track SSH files.
- `S3A['path']`: Sentinel-3A along-track SSH files.
- `S3B['path']`: Sentinel-3B along-track SSH files.
- `S6A['path']`: Sentinel-6A along-track SSH files.
- `SWON['path']`: SWOT nadir along-track SSH files.

### SSH error products

In both configs:

- `path_err`: directory containing the per-satellite SSH noise files.
- `AL['path_err']`, `C2N['path_err']`, `H2B['path_err']`,
  `S3A['path_err']`, `S3B['path_err']`, `S6A['path_err']`, and
  `SWON['path_err']`: derived from `path_err`.

If no per-satellite error files are available, use the constant
`sigma_nadir` fallback consistently for the SSH observation blocks.

## Other settings users commonly adapt

- `myINV['minimizer']`: `scipy` keeps the Historical SciPy Minimizer;
  `optax-decoupled` selects the Device-Resident Optax Minimizer and requires
  `device_resident_state=True`, `jit_cost_and_grad=True`, and
  `cost_and_grad_schedule='scan'`.
- `myINV['relative_gradient_tolerance']`, `myINV['convergence_nit']`, and
  `myINV['minimum_iterations']`: common stopping rule for the Optax path.
- `myINV['save_minimization']`: keep it `False` for the scalar-only Optax hot
  path. Set it to `True` only when per-iteration restart files justify
  transferring and writing the complete Control Vector; `Xres.nc` is always
  saved at the end.
- `EXP['init_date']` and `EXP['final_date']`: experiment time window.
- `myGRID['lon_min']`, `myGRID['lon_max']`, `myGRID['lat_min']`,
  `myGRID['lat_max']`, `myGRID['dlon']`, and `myGRID['dlat']`: regional grid.
- `compute_obs`: set to `True` to rebuild cached observations/operators, or
  `False` to reuse existing files when available.
- `myINV['maxiter']`, `myINV['restart_4Dvar']`, and convergence tolerances:
  minimization behavior.

Generated directories such as `H/`, `obs/`, `outputs/`, `scratch/`, and
`controls/` are runtime products. They should be regenerated for a new machine,
date window, or data layout rather than treated as portable configuration.
