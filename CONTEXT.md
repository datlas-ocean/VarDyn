# VarDyn

VarDyn reconstructs ocean dynamical variables by combining a reduced control
space, dynamical propagation, observations, and variational inversion.

## Language

**Analysis Window**:
The time interval jointly optimized by one 4DVar inversion.
_Avoid_: Batch, run period

**Checkpoint**:
A time in the Analysis Window where VarDyn applies the Reduced Basis, evaluates
observation misfits, or stores state needed by the adjoint.
_Avoid_: Snapshot, save time

**Control Vector**:
The coefficients optimized by 4DVar; the Reduced Basis maps them to model
variables and parameters.
_Avoid_: Parameter vector, weights

**Reduced Basis**:
The mapping between the Control Vector and corrections to model variables or
parameters over space and time.
_Avoid_: Encoder, feature basis

**Model-Error Forcing**:
A time-dependent additive correction to the model tendencies, inferred by the
variational analysis. Its components follow the model variables they force.
_Avoid_: State correction, analyzed state

**Forcing Bias**:
The time mean of a Model-Error Forcing over its calibration period, stored
separately from the zero-mean covariance represented by BMaux.
_Avoid_: Mean BMaux coefficient, background state

**BMaux Temporal Scale**:
The local half-width of a compact temporal basis function, calibrated from the
autocovariance of zero-mean Model-Error Forcing at one spatial scale.
_Avoid_: E-folding time, regional decorrelation time

**BMaux Calibration Domain**:
The wet model domain used to estimate Model-Error Forcing statistics, including
sponge zones and excluding land.
_Avoid_: Diagnostic interior, propagation-safe domain

**Spatial Filter Support**:
The fraction of a scale-dependent filter's weighted footprint covered by valid
calibration data. A band is locally supported when this fraction reaches 50%.
_Avoid_: Temporal sample coverage, wet-point fraction

**BMaux Scale Grid**:
The eight spatial wavelengths actually represented by BMaux between 80 and
1000 km; denser filter bands are calibration diagnostics rather than controls.
_Avoid_: Diagnostic wavelength grid, arbitrary spectral bins

**Interface-Displacement Forcing**:
The vertical Model-Error Forcing expressed in the SW model's interface
coordinate, distinct from physical sea-surface-height forcing.
_Avoid_: SSH tendency, sea-level forcing

**BMaux Auxiliary Set**:
The three variable-specific auxiliary files that jointly define the covariance
of zonal, meridional, and interface-displacement Model-Error Forcing.
_Avoid_: Combined auxiliary file, shared forcing prior

**Calibrated BMaux Temporal Scale**:
A BMaux Temporal Scale written in physical days and consumed without an
empirical multiplier, within the identifiable range of 1 to 60 days.
_Avoid_: Scaled Tdec, heuristic decorrelation factor

**Stationary Forcing Covariance**:
The covariance of temporally centered Model-Error Forcing over the calibration
period, without removing a trend or seasonal harmonic.
_Avoid_: Seasonal covariance, detrended forcing covariance

**Coupled Spectral Calibration**:
The joint inference of BMaux component variances and temporal scales from
overlapping diagnostic bands using the basis's spectral response.
_Avoid_: One-band-one-component fit, independent band calibration

**Log-Spectral Variance Fit**:
The strictly positive inference of BMaux component variances by matching the
logarithms of measured and reconstructed band variances, with curvature
regularization across wavelength. It gives comparable influence to each
resolved spectral decade while admitting smooth power-law spectra.
_Avoid_: Absolute-variance NNLS, zero-filled component spectrum

**Cross-Forcing Diagnostic**:
The scale-dependent correlation among zonal, meridional, and
interface-displacement Model-Error Forcing, evaluated on a common h grid. It is
reported as a diagnostic because the current variable-specific BMaux bases do
not encode cross-variable covariance.
_Avoid_: Coupled BMaux prior, multivariate auxiliary covariance

**Completed Auxiliary Field**:
A BMaux field completed independently at each wavelength from directly
supported wet points by spatial regularization within the connected ocean.
Direct and completed values remain distinguishable through support, fit-quality,
and completion masks; land fill exists only to make interpolation well-defined.
_Avoid_: Unflagged extrapolation, land estimate

**Temporal-Scale Uncertainty**:
The 10th-to-90th-percentile interval of a BMaux Temporal Scale obtained by
resampling the stationary calibration record in 30-day temporal blocks.
Boundary-hitting or weakly constrained estimates are flagged and completed
spatially before operational use.
_Avoid_: Formal posterior interval, independent-day bootstrap

**Source-Basis Resolution Diagnostic**:
The comparison of observed forcing spectra with the synthesis response and
conditioning of the Gaussian source basis. It distinguishes spectral energy
present in analyzed forcings from scales that its coefficient lattice resolves
independently; requested 80 km components remain represented and quality-flagged.
_Avoid_: Hard nominal-scale cutoff, energy-only resolution claim

**Cost-Gradient Evaluation**:
One complete background-cost, forward-model, observation-misfit, and adjoint
calculation for a Control Vector.
_Avoid_: Objective call, model pass

**4DVar Minimizer**:
The algorithm that proposes Control Vectors and accepts steps using
Cost-Gradient Evaluations.
_Avoid_: Solver, optimizer backend

**Historical SciPy Minimizer**:
The host-resident SciPy L-BFGS-B 4DVar Minimizer retained for compatibility and
reference comparisons.
_Avoid_: CPU mode, legacy solver

**Device-Resident Optax Minimizer**:
The L-BFGS 4DVar Minimizer whose Control Vector and history remain on the JAX
device while Python decides the scalar Armijo line search.
_Avoid_: Full-GPU solver, JAXopt mode

**Converged Analysis**:
An Analysis Window whose accepted cost changes satisfy the configured relative
cost tolerance for the required consecutive iterations.
_Avoid_: Small-gradient analysis, exhausted iteration budget
