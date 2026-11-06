# VarDyn

VarDyn reconstructs ocean dynamical variables by combining a reduced control
space, dynamical propagation, observations, and variational inversion.

## Branch-specific context

When the active Git branch is `Dev-SW`, **read
[CONTEXT_Dev-SW.md](/home/flo/VarDyn/CONTEXT_Dev-SW.md) in addition to this file
before working on the branch**. It records the GLORYS shallow-water learning,
MAP1/MAP2 results, active SSH nadir experiments, covariance coupling, and
follow-up instructions. Recheck its dated run status against the live outputs.

## Language

**North Atlantic Midlatitude Domain**:
The Atlantic region from 80°W to 0° and 30°N to 60°N used for the
GLORYS learning experiment.
_Avoid_: Tropical Atlantic domain

**North Atlantic Absolute SSH**:
The GLORYS sea-surface height used without subtraction of a spatial mean in
the North Atlantic calibration, boundary fields, and synthetic nadirs.
_Avoid_: North Atlantic SSH anomaly

**Admissible Reference Depth**:
The uniform initial SW layer depth chosen so the year's absolute GLORYS SSH
starts with positive total layer thickness throughout the North Atlantic domain.
_Avoid_: Chelton-derived regional depth, learned final depth

**OSE**:
An ocean reconstruction experiment assimilating measurements from real observing systems.
_Avoid_: OSSE, synthetic-observation experiment

**Withheld Real Nadir**:
Along-track sea-level measurements from a real altimeter mission excluded from the OSE observation cost and used to assess reconstructed SSH.
_Avoid_: Assimilated nadir, synthetic nadir

**First-Baroclinic-Mode Current**:
The part of a three-dimensional current associated with the first vertical baroclinic mode, whose shape depends on water-column stratification and bottom depth.
_Avoid_: Current at a single depth, sub-mixed-layer current

**Sub-Mixed-Layer Target Current**:
A GLORYS current target averaged over a thin layer immediately below the density-defined mixed layer, rather than sampled at the surface or interpreted as a full-depth baroclinic mode.
_Avoid_: First-baroclinic-mode current, surface current

**Mixed-Layer-Offset Current**:
A GLORYS current sampled at a specified depth below the density-defined mixed-layer base, distinct from a mean over a layer beneath that base.
_Avoid_: First-baroclinic-mode current, fixed-depth current

**OSSE**:
An ocean reconstruction experiment assimilating observations sampled from a reference simulation.
_Avoid_: OSE, real-observation experiment

**MAP1-Centered OSE Physical Prior**:
A prior for real-nadir reconstruction whose physical-parameter fields are centered on the GLORYS-trained MAP1 estimate, with reduced uncertainty so real observations can adjust them.
_Avoid_: MAP1 physics fixed for OSE, original-reference physical prior

**GLORYS OSE Boundary SLA**:
A GLORYS height boundary field expressed relative to GLORYS mean dynamic topography, matching the real-nadir SLA convention.
_Avoid_: Absolute GLORYS height in an SLA analysis

**Direct GLORYS Boundary Current**:
A GLORYS current boundary field supplied as a full current without subtracting a mean current.
_Avoid_: Current anomaly

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

**Analyzed-Forcing Calibration**:
The estimation of Model-Error Forcing statistics from corrections inferred by
variational analysis. These statistics describe regularized analyzed corrections,
not an unbiased model-error covariance or equation defects evaluated on reference states.
_Avoid_: Direct GLORYS residual calibration, observed physical forcing

**Prescribed Forcing Bias**:
The learned temporal mean of Model-Error Forcing applied as a fixed background
correction, separately from the covariance of centered forcing fluctuations.
_Avoid_: Zero-mean constant control, bias standard deviation

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

**Basin-Mean Height Forcing**:
The area-weighted spatial mean of Interface-Displacement Forcing within each
connected basin, which may vary in time. Its spatially uniform contribution is
distinct from the remaining zero-integral height forcing and from a temporal Forcing Bias.
_Avoid_: Temporal mean forcing, physical SSH forcing

**Local Multiscale SW Forcing Calibration**:
The estimation of spatially varying amplitudes and temporal scales for
divergent, rotational, and zero-integral height Model-Error Forcing at multiple
coexisting spatial scales. Basin-Mean Height Forcing is characterized separately
for each connected basin.
_Avoid_: Homogeneous forcing calibration, single local spatial scale

**SW Forcing Wavelength**:
The spatial scale of a forcing component identified by the peak of variance
per logarithmic wavenumber interval. It is distinct from the compact support
radius of the potential or compensated-height function that represents it.
_Avoid_: Potential radius, compact half-width

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

**SW Coefficient Decorrelation Time**:
The lag at which the modeled temporal correlation of SW forcing coefficients
reaches `1/e`. The original daily AR1 model uses an exponential. The reduced
local MAP2 fits the covariance induced by its actual temporal-window bank,
matching temporal centering in the empirical ACF; its reported `tdec` is the
1/e crossing before sample-mean subtraction. Window width and node spacing
are separate quantities.
_Avoid_: Temporal node spacing, cosine-envelope half-width

**SW Coefficient Standard Deviation**:
The stationary standard deviation of centered SW forcing coefficients at
temporal control nodes, for a specified spatial-function normalization.
_Avoid_: Innovation standard deviation, pointwise forcing amplitude

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

**Annual Shared Forcing Statistics**:
The spatially local forcing-coefficient mean, stationary standard deviation,
and temporal decay scale inferred jointly from the year's Analysis Windows.
They are common across windows, rather than averages of independently fitted
window parameters.
_Avoid_: Monthly parameter average, spatially uniform annual prior

**Window Moment Ownership**:
The assignment of each analysis date to one Analysis Window for annual
statistical aggregation. Observations can inform multiple overlapping windows
while the selected posterior moments contribute only once.
_Avoid_: Observation exclusion, duplicated annual moments

**Continuous Annual Empirical Calibration**:
The October 7, 2026 experiment reconstructs one uninterrupted annual trajectory
with daily forcing controls and a permissive physical prior (tdec 2 days).
The revised MAP2 learns mean, variance and ACF locally for every spatial atom,
fits the ACF of compact temporal windows, and reduces temporal centers as their
width increases. It starts from zero physical forcing controls, with the same
initial ocean state as MAP1, and performs no posterior covariance solves.
The earlier pooled-statistics, warm-start MAP2 and its sensitivity chain are
superseded; their outputs remain archived. See
`docs/sw_reduced_local_map2_20261007.md` for the revised strategy and
`docs/sw_annual_empirical_learning.md` for the historical daily strategy.
_Avoid_: Windowed EM, posterior-corrected variance, assumed prior independence

**Equal Physical Scale Amplitude**:
A common expected wet-area RMS of the total forcing contribution at each
wavelength, separately for divergent momentum, rotational momentum and height.
The coefficient prior accounts for derivatives, masks and overlapping atoms.
The October 7 experiment uses 0.1 (m/s)/day for each momentum component and
1 m interface/day for height, with amplitude sensitivity factors 0.5 and 2.
_Avoid_: Equal potential coefficient std, equal amplitude per isolated atom

**GLORYS-Informed Initial Physical Amplitude**:
An experiment-specific initialization measured from daily GLORYS equation
defects with the current fixed Chelton/g'=0.025 model, jointly projected onto
the daily spatial potential basis. The October 7 configuration
`config_sw_daily_glorys_prior.py` uses physical RMS values 0.0304 and 0.0198
(m/s)/day for divergent and rotational forcing, 13.5 m interface/day for
compensated height and 1.23 m interface/day for basin means, with tdec=2 days.
Each component keeps the same RMS across wavelengths. These are approximate
initial amplitudes, not learned posterior uncertainties or SSH standard
deviations. See `docs/glorys_daily_prior_20261007.md` for support and limitations.
