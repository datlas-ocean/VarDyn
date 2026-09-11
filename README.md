# VarDyn

Variational reconstruction of ocean dynamical variables.

VarDyn is a condensed version of MASSH focused on dynamical, variational mapping
of sea-surface height and related ocean variables. It keeps the core data
assimilation machinery: configurable grids, dynamical model operators,
observation loading and interpolation, reduced-basis control vectors, 4DVar
inversion, windowed orchestration, and diagnostics.

## Main features

- **4DVar assimilation** with a reduced-basis control vector and two maintained
  minimizers: historical SciPy L-BFGS-B and device-resident Optax L-BFGS with a
  decoupled scalar line search.
- **Forward-only integrations** when no inversion block is configured.
- **Dynamical model cores** for diffusion/identity mappings, 1.5-layer QG,
  1.5-layer shallow-water, QG/SW variants, and balanced-motion/internal-tide
  experiments.
- **JAX-enabled model and cost-function paths** for automatic differentiation
  and accelerated model propagation.
- **External boundary conditions** from NetCDF files, including optional sponge
  zones near open boundaries and coastal masks.
- **Observation support** for gridded L4 fields, nadir altimetry, and swath
  altimetry, with optional MDT correction, noise/error handling, and date-range
  file filtering.
- **Observation operators** for L3 along-track/swath interpolation and L4
  gridded interpolation.
- **Reduced bases** including offset, 2D/3D Gaussian, balanced-motion auxiliary
  wavelet bases, and open-boundary/internal-tide basis controls.
- **Windowed assimilation orchestration** over time and space, including
  multiprocessing, GPU-device assignment, observation caching, and output
  merging utilities.
- **Diagnostics** for OSSE and OSE experiments.

## Repository layout

```text
mapping/src/          Core library: config, state, models, observations, basis,
                      inversion, diagnostics, orchestration
mapping/models/       Dynamical kernels for QG, shallow-water, and QG/SW models
mapping/examples/     Example notebooks and experiment configurations
mapping/aux/          Auxiliary NetCDF fields used by examples and defaults
pyproject.toml        Package metadata and Python dependencies
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/datlas-ocean/VarDyn.git
   cd VarDyn
   ```

2. Create and activate a Python environment:

   ```bash
   conda create -n env_vardyn python=3.10
   conda activate env_vardyn
   ```

3. Install `pyinterp` with conda-forge.

   `pyinterp` provides interpolation tools for geo-referenced data. Installation
   can be long because of its dependency stack; `mamba` is usually faster.

   ```bash
   conda install -c conda-forge pyinterp
   ```

4. Optionally install a GPU/TPU build of `jax`.

   Users with accelerator access should install the matching JAX build before
   installing VarDyn. If no specialized JAX build is present, the standard
   CPU-only packages are installed by default.

5. Install VarDyn and the remaining Python dependencies:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

## Running an example

The main bundled example is in:

```text
mapping/examples/2021a_SSH_mapping_OSE/
```

It contains:

- `2021a_VarDyn_QG.ipynb`: notebook entry point.
- `config_2021a_VarDynQG.py`: full experiment configuration.
- `data_access.ipynb`: helper notebook for data access.

The example config runs a Gulf Stream SSH OSE using a Cartesian grid,
1.5-layer QG dynamics, external DUACS boundary conditions, nadir altimetry
observations, a balanced-motion auxiliary reduced basis, and 4DVar inversion.

## Configuration model

Experiments are driven by Python configuration blocks. Defaults live in
`mapping/src/config_default.py`; an experiment config selects blocks with names
such as:

- `NAME_GRID`
- `NAME_MOD`
- `NAME_BC`
- `NAME_OBS`
- `NAME_OBSOP`
- `NAME_BASIS`
- `NAME_INV`
- `NAME_DIAG`

Each selected block normally has a `super` field pointing to a default block
family. For example, a custom model block can set `super = 'MOD_QG1L'` and then
override only the fields needed for the experiment.

### Choosing the 4DVar minimizer

`INV_4DVAR.minimizer` selects the minimization adapter. Existing configurations
keep the historical default:

```python
myINV = dict(
    super='INV_4DVAR',
    minimizer='scipy',
)
```

The device-resident path is explicit:

```python
myINV = dict(
    super='INV_4DVAR',
    minimizer='optax-decoupled',
    jit_cost_and_grad=True,
    cost_and_grad_schedule='scan',
    gtol=0.1,
    convergence_nit=2,
    minimum_iterations=5,
)
```

The Optax adapter keeps the control, gradient, direction, and L-BFGS history on
the JAX device. With `save_minimization=False`, Python receives only scalar
line-search diagnostics. Enabling per-iteration saves transfers and writes the
complete Control Vector after every accepted iteration; the final `Xres.nc` is
saved in either mode. See
[the GPU minimization benchmark](docs/benchmarks/gpu-4dvar-minimization.md) for
performance results and configuration constraints.

## Testing and checks

Unit checks for the compiled variational path and minimizer live under
`mapping/tests`. Run them with an environment containing `pytest`, or at least
run syntax checks on touched modules, for example:

```bash
python -m py_compile mapping/src/mod.py mapping/src/inv.py
```

For behavior changes, run the relevant notebook or a reduced experiment config
against a small domain/time window.
