# CSWM VarDyn Example

This directory contains VarDyn 4DVar examples for the JET BM-IT OSSE, using either the standalone internal-tide model or the coupled BM-IT model.

## Files

- `config_VarDyn-CSWM-JET.py`: configuration for the CSW internal-tide model with Balanced Motion coupling.
- `run_VarDyn-CSWM-JET.ipynb`: notebook that loads the config, builds the state/model/observations/operators/basis, runs 4DVar, and optionally runs diagnostics.
- `config_VarDyn-BMIT-JET.py`: coupled `MOD_BMIT` configuration using a QG balanced-motion component and a CSW internal-tide component.
- `run_VarDyn-BMIT-JET.ipynb`: notebook that runs the coupled BM-IT inversion against total SLA observations.

## Configuration Summary

The experiment uses the JET BM-IT reference data as both forcing context and OSSE truth:

- grid: `GRID_FROM_FILE`, read from `path_modes`
- model: `MOD_CSW1L`
- state variables: `u_it`, `v_it`, `ssh_it`
- controlled parameters: `He_mean`, `hbc`, and `alpha`
- observations: gridded L4 SSH from `path_ref`, observing `ssh_it`
- observation operator: `OBSOP_INTERP_L4`
- reduced basis: `BASIS_GAUSS3D` for `He_mean`, `BASIS_OFFSET` for `alpha`, and `BASIS_HBC` for internal-tide boundary waves
- diagnostics: `DIAG_OSSE` comparing output `ssh_it` with reference `ssh_it`

## Paths To Check

Before running, check these paths in `config_VarDyn-CSWM-JET.py`:

- `path_modes`: NetCDF file containing the grid, vertical modes, MDT, and first baroclinic phase speed.
- `path_ref`: NetCDF file containing reference surface fields, Balanced Motion SSH, and OSSE SSH observations.
- The reference and vertical-mode files can be downloaded from the MEOM OpenDAP catalog: https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/JET_BM-IT/catalog.html
- `myPath`: local root for generated `outputs/`, `scratch/`, `obs/`, `H/`, `controls/`, and `diags/` directories.
- `path_images2mp4`: optional movie helper used by diagnostics.

Changing `myPath` is usually enough to redirect generated files. Changing `path_modes` and `path_ref` is needed if the JET data are stored outside `/data1/data/models/JET_BM-IT/full_resolution`.

## Running The Example

From `mapping/examples/CSWM`, open and run:

```bash
jupyter lab run_VarDyn-CSWM-JET.ipynb
```

The notebook performs the workflow step by step:

1. Select a GPU through `CUDA_VISIBLE_DEVICES`.
2. Load and merge the config with VarDyn defaults.
3. Build the grid/state from `path_modes`.
4. Instantiate `MOD_CSW1L`.
5. Load OSSE observations from `path_ref`.
6. Build the L4 observation operator.
7. Build the reduced basis.
8. Run the configured 4DVar inversion.
9. Run OSSE diagnostics after outputs have been written.

## Common Settings To Adapt

- `EXP['init_date']` and `EXP['final_date']`: analysis window.
- `EXP['assimilation_time_step']`: control/analysis time spacing.
- `EXP['saveoutput_time_step']`: output cadence.
- `compute_obs`: set to `False` to reuse cached observations/operators.
- `NR['subsampling']`: temporal thinning of the OSSE SSH observations.
- `CSWM['dist_sponge_bc']` and `CSWM['sponge_coef']`: sponge width and damping strength.
- `CSWM['extend_it_open_boundary_sponge']`: whether to extend the entering-wave medium across open-boundary sponge bands.
- `Basis_He`, `Basis_alpha`, and `Basis_hbc`: prior scales and basis spacing for the controlled parameters.
- `myINV['maxiter']`, `myINV['ftol']`, and `myINV['restart_4Dvar']`: minimizer behavior.

Generated directories such as `H/`, `obs/`, `outputs/`, `scratch/`, `controls/`, and `diags/` are runtime products and should not be treated as portable source files.
