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
The reference thickness of the active layer in a physical reduced-gravity shallow-water model; together with Reduced Gravity it determines the first-baroclinic wave speed. For the current one-layer implementation, it is 80 m and Reduced Gravity is diagnosed as c1²/H.
_Avoid_: Equivalent Depth, bathymetry, mixed-layer depth

**Layer-Resolved Depth Controls**:
Independent, positive controls of the reference depths of the mixed layer and
the first-baroclinic layer. Their public names are `H_ml` and `H_bc1`.
_Avoid_: One shared depth control, additive depth correction, stacked public H

**Tropical Two-Layer Reference Depths**:
The default reference depths for the tropical two-layer modal reduced-gravity
stack: 40 m for the mixed layer and 80 m for the first-baroclinic layer.
_Avoid_: Full thermocline depth, fixed global layer depths

**Layer-Resolved State**:
The six prognostic fields of the two-layer modal reduced-gravity stack:
`h_ml`, `u_ml`, `v_ml`, `h_bc1`, `u_bc1`, and `v_bc1`. Diagnosed surface
fields remain separate observational products.
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
reduced-gravity experiment: mixed layer at the surface, then the
first-baroclinic layer below. The Ekman response is represented within the
mixed-layer dynamics, not as a separate prognostic layer.
_Avoid_: Arbitrary layer numbering, equivalent-depth stack

**Modal Reduced-Gravity Stack**:
A multilayer reduced-gravity system that advances slow internal modes and
diagnoses Sea-Surface Height from them, rather than advancing a fast physical
free-surface mode.
_Avoid_: Free-surface stack, barotropic model

**Baroclinic Shear Initialization**:
The initialization of unobserved lower-layer velocity from observed mixed-layer
velocity so depth-integrated transport is zero; the two layers carry opposing
transport in proportion to their reference depths.
_Avoid_: Zero lower-layer velocity, implicit barotropic current

**Surface-Projected Baroclinic Forcing**:
A single surface correction projected deterministically into both layers: its
height follows the selected baroclinic mode and its momentum follows
zero-transport baroclinic shear.
_Avoid_: Independent lower-layer forcing, surface-only layer correction

**Surface-Projected Boundary Condition**:
An open-boundary field obtained from SSH and mixed-layer velocity, then lifted
to both layers using the same modal-height and baroclinic-shear projections as
initialization.
_Avoid_: Independent lower-layer boundary condition, layer-zero copy

**Diagnosed Sea-Surface Height**:
The free-surface observable used for boundary and altimetric comparisons. In a
one-layer reduced-gravity model it can be diagnosed from Interface
Displacement using a physical-gravity scaling. In the two-layer modal stack it
is the explicit sum of mixed-layer and first-baroclinic thickness anomalies;
the individual layer states are unobserved.
_Avoid_: Model height, interface displacement, top-layer internal pressure

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
