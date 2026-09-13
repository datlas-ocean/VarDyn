# VarDyn

VarDyn reconstructs ocean dynamical variables by combining prior dynamical structure with observations. This glossary fixes the language used to discuss controlled ocean parameters and physically diagnosed fields.

## Language

**Equivalent Depth**:
The positive layer-depth scale used by the QG/SW dynamics to set wave speed and pressure coupling.
_Avoid_: H anomaly, depth anomaly

**Equivalent-Depth Control**:
A dimensionless, unconstrained log-multiplier used to adjust Equivalent Depth during inversion.
_Avoid_: Total H, physical H, meter-valued H correction

**Reference Equivalent Depth**:
The background or climatological Equivalent Depth that represents the neutral state before inversion changes it.
_Avoid_: Raw H0, uncontrolled H

**Bathymetry**:
The spatial ocean-depth field describing the physical water-column depth.
_Avoid_: Equivalent Depth, layer depth, H control

**Layer Depth**:
The shallow-water layer thickness used by a Balanced Motion component.
_Avoid_: Bathymetry, ocean depth, topographic depth

**Physical Reduced-Gravity Layer Depth**:
The reference thickness of the active layer in a physical reduced-gravity shallow-water model; together with Reduced Gravity it determines the first-baroclinic wave speed. It is configured and diagnosed in metres.
_Avoid_: Equivalent Depth, bathymetry, mixed-layer depth

**Physical Layer-Depth Control**:
A dimensionless logarithmic Control Vector coordinate that maps a Reference Physical Reduced-Gravity Layer Depth to a positive Controlled Layer Depth in metres.
_Avoid_: Metre-valued depth correction, additive H anomaly

**Controlled Layer Depth**:
The positive physical reduced-gravity reference thickness in metres after applying the Physical Layer-Depth Control; it excludes prognostic Interface Displacement.
_Avoid_: Interface displacement, instantaneous thickness, dimensionless H control

**Instantaneous Layer Thickness**:
The sum of Controlled Layer Depth and prognostic Interface Displacement, `H_total + eta`, expressed in metres.
_Avoid_: Controlled layer depth, SSH, mixed-layer depth

**One-Layer Baroclinic QGSW**:
The public MOD_QGSW representation: one physical reduced-gravity layer advanced by shallow-water dynamics, with no QG, Ekman, MLD, slab, or layer-role configuration.
_Avoid_: QGSW QG mode, modal layer stack, Ekman–barocline model

**Generic Multilayer SW Core**:
The internal layer-indexed shallow-water implementation retained for future extensions; it supports multiple physical layers and passive tracers without exposing Ekman, MLD, or layer-role semantics through MOD_QGSW.
_Avoid_: Public multilayer MOD_QGSW configuration, physical layer-role stack

**Wind-Forcing Depth**:
The positive depth used to convert wind stress to acceleration: Controlled Layer Depth, a prescribed `h_wind`, or Instantaneous Layer Thickness according to one exclusive configuration choice.
_Avoid_: Implicit wind-depth fallback, mixed-layer depth by assumption

**Layer-Resolved Passive Tracer Content**:
The conservative tracer coordinate `Q_k = (H_total,k + eta_k) C_k` advanced independently in each Generic Multilayer SW Core layer.
_Avoid_: Bare concentration tendency, upper-layer MLD aggregate

**Layer-Resolved Depth Controls**:
Positive controls of the physical Ekman, total mixed-layer, and first-baroclinic
reference depths. Their public names are `H_ek`, `H_ml`, and `H_bc1`; a distinct
Ekman layer nested inside the mixed layer uses the bounded fraction `r_ek =
H_ek/H_ml` instead of controlling two potentially inconsistent depths.
_Avoid_: One shared depth control, additive depth correction, stacked public H

**Tropical Two-Layer Reference Depths**:
The default reference depths for the tropical two-layer modal reduced-gravity
stack: 40 m for the mixed layer and 80 m for the first-baroclinic layer.
_Avoid_: Full thermocline depth, fixed global layer depths

**Layer-Resolved State**:
The prognostic thickness and velocity fields `h_*`, `u_*`, and `v_*` for every
active physical layer in a Modal Reduced-Gravity Stack. Layer roles are named
Ekman, mixed layer, shared Ekman–mixed layer, and first baroclinic; diagnosed
surface SSH/U/V remain separate public products.
_Avoid_: Layer-zero state, surface-only restart

**Control-Preserving Restart**:
A multi-window restart that reloads canonical layer-depth controls before
physical depth diagnostics, so the optimizer's depth-control coordinates are
continuous across subwindows.
_Avoid_: Control rebasing, physical-depth-only restart

**Reference First-Baroclinic Speed**:
The spatial `c1` field that calibrates the first (fastest) baroclinic mode of the
two-layer modal reduced-gravity stack at its reference depths.
_Avoid_: Barotropic speed, arbitrary constant wave speed

**Fixed Reduced Gravity**:
The reference reduced-gravity field remains fixed while Layer-Resolved Depth
Controls change the layer depths and therefore the modal dynamics.
_Avoid_: Gravity compensation, fixed wave speed under depth control

**Reduced-Gravity Control**:
A dimensionless logarithmic Control Vector coordinate that maps Reference
Reduced Gravity to a positive spatially varying Controlled Reduced Gravity.
It is static during a forward integration; phase speed is diagnosed as
`sqrt(Controlled Reduced Gravity * Controlled Layer Depth)`.
_Avoid_: Advected density, additive reduced-gravity anomaly, independent phase-speed control

**Reference Reduced Gravity**:
The positive reduced-gravity field before inversion. It is configured directly
or diagnosed from Reference First-Baroclinic Speed and Reference Physical
Reduced-Gravity Layer Depth through `g_prime = c1**2 / H`.
_Avoid_: Physical gravity, time-evolving density, controlled reduced gravity

**Diagnosed Modal Fields**:
The derived Interface Amplification Factor and the two modal wave-speed fields
saved with output for physical verification; they are recomputed from resumed
depth controls and Fixed Reduced Gravity.
_Avoid_: Modal controls, restart inputs

**Interface Displacement**:
The prognostic displacement of the active reduced-gravity interface or modal surface.
_Avoid_: SSH, altimetric height, layer-depth control

**Interface Amplification Factor**:
The ratio used to diagnose internal Interface Displacement from Diagnosed
Sea-Surface Height when a surface field initializes a multilayer state. It is
uncontrolled and diagnosed from the current layer depths and fixed reduced
gravities; it may be supplied by a vertical-mode product when available.
_Avoid_: SSH scale factor, arbitrary layer split

**Physical Reduced-Gravity Layer Stack**:
The ordered active-layer representation used by a multilayer physical
reduced-gravity experiment. Supported roles are a wind-driven Ekman layer, a
tracer-carrying mixed-layer remainder, a shared Ekman–mixed-layer surface
layer, and a first-baroclinic layer. With distinct Ekman and mixed-layer roles,
`H_ml` is the total surface-to-ML-base depth and the disjoint model thicknesses
are `H_ek = r_ek H_ml` and `H_ml,remainder = (1-r_ek) H_ml`.
_Avoid_: Arbitrary layer numbering, overlapping prognostic thicknesses, equivalent-depth stack

**Fixed-Depth Ekman Slab**:
A momentum-only surface sublayer with prescribed, time-invariant thickness. It carries horizontal velocity but neither a prognostic thickness nor pressure contribution to SSH.
_Avoid_: Ekman shallow-water layer, Ekman depth control

**Two-Slab Ekman Closure**:
The ordered pair of Fixed-Depth Ekman Slabs: an upper wind-forced slab and a lower slab, coupled by interfacial drag; the lower slab is frictionally coupled to the prognostic first-baroclinic velocity.
_Avoid_: Two Ekman SW layers, two free-surface Ekman layers


**Baroclinic-Referenced Ekman Coriolis**:
The Coriolis tendency of each Fixed-Depth Ekman Slab acts on its velocity departure from the prognostic first-baroclinic velocity. The barocline supplies the pressure-bearing balanced-current reference.
_Avoid_: Isolated slab Coriolis, diagnosed-geostrophic reference


**Terminal Ekman Drag**:
A non-conservative drag on the lower Fixed-Depth Ekman Slab against a resting unresolved deep reservoir. It is the column momentum sink required where Coriolis cannot balance wind stress.
_Avoid_: Conservative Ekman–baroclinic drag, horizontal viscosity


**Fixed-Slab Ekman Pumping**:
The conservative transfer of the Ekman transport relative to the first-baroclinic velocity into the first-baroclinic thickness tendency. It closes the column mass budget without counting the baroclinic transport twice or giving the slabs prognostic thicknesses.
_Avoid_: Ekman slab thickness evolution, absolute slab-transport pumping, drag-only Ekman coupling

**Modal Reduced-Gravity Stack**:
A multilayer reduced-gravity system that advances slow internal modes and
diagnoses Sea-Surface Height from them, rather than advancing a fast physical
free-surface mode.
_Avoid_: Free-surface stack, barotropic model

**Upper-Layer Tracer Content**:
The conservative tracer coordinate `Q = T D_upper`, with `D_upper` equal to the
instantaneous total thickness of the tracer-carrying surface layers. SST is
vertically uniform across those layers and its flux uses their summed mass
transport; explicit reduced-basis SST error fluxes are deliberate sources.
_Avoid_: Layer-zero passive concentration, fixed-depth SST transport

**Unfiltered MLD Reference**:
The analysis-window mean of daily GLORYS `mlotst`, interpolated to the model
grid without spatial filtering. Missing-value completion and optional
area-weighted collapse through `constant_MLD=True` are not spatial smoothing.
_Avoid_: Smoothed MLD, boundary-condition MLD

**Filtered Boundary Condition Source**:
The `smooth-10d-10xy` GLORYS product used for SSH, velocity, and SST boundary
conditions. It is distinct from the unfiltered daily GLORYS source used for SST
observations and the Unfiltered MLD Reference.
_Avoid_: Observation source, MLD reference

**Role-Selective Dynamic Initialization**:
SSH and velocity boundary fields initialize only the first-baroclinic layer;
Ekman and mixed-layer dynamic anomalies start from zero.
_Avoid_: Primary-mode initialization, zero-transport shear initialization

**Baroclinic Dynamic Correction**:
The dynamic error tendencies `Fu`, `Fv`, and `Fh`, which act only on the
first-baroclinic layer and never directly force Ekman or mixed-layer dynamics.
_Avoid_: Surface correction, vertically projected correction

**Role-Selective Boundary Condition**:
SSH and velocity constrain only the first-baroclinic boundary, while SST
constrains only the upper-layer tracer boundary.
_Avoid_: Primary-mode boundary condition, identical boundary targets by layer

**Fixed SSH Reference**:
The time-independent initial SSH boundary field retained as the public
background when no MDT is prescribed; the layer stack carries its anomaly.
_Avoid_: Prognostic mean SSH, MDT, barotropic mode

**Diagnosed Sea-Surface Height**:
The free-surface observable used for boundary and altimetric comparisons. For a
Physical Reduced-Gravity Layer Stack it is diagnosed hydrostatically as
`p_deep/g`, where deep pressure is obtained from all interface displacements
and Fixed Reduced Gravity. If no MDT is prescribed the public field and misfit
coordinate are SSH; with an MDT they are SLA. Individual layer states remain
unobserved.
_Avoid_: Sum of layer anomalies, interface displacement, top-layer internal pressure

**Antilles Domain**:
The regional western tropical Atlantic domain used for eNATL60 Antilles experiments.
_Avoid_: Toy domain, global domain, generic regional box

**Equivalent-Depth Floor**:
The lower physical limit used to keep Total Equivalent Depth non-negative.
_Avoid_: Numerical epsilon, hidden lower bound

**Total Equivalent Depth**:
The physically meaningful Equivalent Depth after applying the control transformation and any lower or upper physical limits.
_Avoid_: H correction, H anomaly

**Balanced Motion**:
The slowly evolving, balanced ocean signal that provides the BM contribution to SSH and may provide velocity and height fields for coupling to internal tides.
_Avoid_: BM model, QG field

**Internal Tide Coupling**:
The relationship by which Balanced Motion modifies internal-tide equivalent depth and advective terms; advective coupling uses full Balanced Motion velocity, while equivalent-depth coupling may use either anomaly or full Balanced Motion height.
_Avoid_: BM forcing, IT correction

**Internal Tide Generation Term**:
The mechanism by which Bathymetry contributes to the creation of internal-tide height variability.
_Avoid_: Bathymetry forcing, topographic source

**Balanced-Motion Coupling Control**:
A spatial control field that tunes how Balanced Motion modifies internal-tide equivalent depth or advective coupling.
_Avoid_: Local stratification, BM stratification control, alpha knob

**Sponge Interior Edge**:
The grid line on the interior side of the boundary sponge layer, still inside the sponge, used as the reference value when Entering-Wave Medium fields are extended through the sponge.
_Avoid_: Open-ocean interior value, sponge core, boundary value

**Sponge Corner Blend**:
A smooth combination of the two relevant Sponge Interior Edge values in corner sponge cells, using the same side partitioning concept as internal-tide boundary blending.
_Avoid_: Corner overwrite, arbitrary side priority

**Entering-Wave Medium**:
The fields seen by internal-tide boundary-wave construction inside the sponge layer; it may differ from the fields used by the interior dynamics.
_Avoid_: Model medium, sponge dynamics, boundary stratification

**Open-Boundary Sponge Extension**:
The extension of Entering-Wave Medium values from the Sponge Interior Edge across the south, north, west, and east boundary sponge bands.
_Avoid_: Coastal sponge extension, island sponge extension, land extension

**Balanced Motion State**:
The set of Balanced Motion fields advanced by the chosen dynamics; it may be SSH-only for balanced SSH models or velocity-plus-SSH for shallow-water dynamics.
_Avoid_: BM variables, model internals

**Surface Balanced Motion**:
The two-dimensional top-layer or surface Balanced Motion fields used by observations, saving, controls, and Internal Tide Coupling.
_Avoid_: Layer-zero output, coupling slice

**Balanced Motion Control Surface**:
The Balanced Motion tendency fields exposed to inversion as correctable controls.
_Avoid_: BM parameters, flux knobs

**Balanced Motion Boundary Condition**:
An external Balanced Motion field used to constrain the BM component at model boundaries.
_Avoid_: BM BC file, edge forcing

## Configuration conventions

- Whenever a configuration key is added, removed, renamed, or changes meaning, update `mapping/src/config_default.py` in the same change. Keep the relevant default block complete and synchronized with every factory or runtime consumer.

## 4DVar and inversion terminology

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

**Cost-Gradient Evaluation**:
One complete background-cost, forward-model, observation-misfit, and adjoint
calculation for a Control Vector.
_Avoid_: Objective call, model pass

**Historical SciPy Minimizer**:
The host-resident SciPy L-BFGS-B 4DVar Minimizer retained for compatibility and
reference comparisons.
_Avoid_: CPU mode, legacy solver

**Device-Resident Optax Minimizer**:
The L-BFGS 4DVar Minimizer whose Control Vector and history remain on the JAX
