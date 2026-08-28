# Physical Modal Layer Stack and Upper-Layer Tracer Conservation

**Status:** Superseded for the public `MOD_QGSW` interface by ADR-0004. The
multilayer SW machinery remains an internal implementation reference.

VarDyn QGSW experiments may use an explicit `modal_layer_stack` whose ordered
roles are selected from Ekman, mixed layer, shared Ekman–mixed layer, and first
baroclinic. A distinct Ekman layer is nested within the total mixed-layer depth:
`H_ek = r_ek H_ml`, while the disjoint mixed-layer remainder has thickness
`(1-r_ek) H_ml`. The shared Ekman–mixed-layer case uses one prognostic surface
layer for both wind response and tracer conservation. Layer depths use positive
log controls; `r_ek` uses a bounded logistic control. Fixed reduced gravities
are calibrated at the reference depths from the prescribed modal speeds.

Public SSH is diagnosed inside the model as deep hydrostatic pressure divided
by physical gravity. SSH/U/V boundary fields and the dynamic corrections
`Fu/Fv/Fh` act only on the first-baroclinic layer; upper-layer dynamic boundary
targets are zero. Wind acts only on a layer with an Ekman role and uses its
instantaneous positive thickness. When no MDT is prescribed, the initial SSH
boundary field is retained as a fixed diagnostic reference so only its anomaly
enters the prognostic baroclinic thickness.

SST is vertically uniform over the configured tracer-carrying upper layers and
is advanced through the conservative content `Q = T D_upper`. Its advective
flux reuses the exact mass fluxes from the upper-layer continuity equations.
Reduced-basis SST error fluxes are explicit sources expressed as temperature
tendencies and converted to content tendencies with instantaneous `D_upper`.

**Considered Options**

- Treat Ekman and mixed-layer depths as overlapping independent prognostic
  layers.
- Set `H_ek = H_ml` while retaining a separate zero-thickness mixed-layer
  remainder.
- Advect SST as a layer-zero concentration with a fixed transport depth.
- Project boundary fields and dynamic corrections onto a coupled primary mode
  or a zero-depth-transport shear spanning the upper and baroclinic layers.

**Consequences**

Three-layer experiments control `H_ml`, `r_ek`, and `H_bc1`; shared
Ekman–mixed-layer experiments control `H_ml` and `H_bc1`. Daily GLORYS `mlotst`
is averaged over the analysis window and interpolated without spatial
filtering; the two SST experiments then use `constant_MLD=True` to collapse it
to one area-weighted domain value and prevent locally vanishing upper-layer
thickness. The filtered `smooth-10d-10xy` GLORYS product supplies SSH/U/V for
the first-baroclinic boundary and SST for the upper-tracer boundary, while
unfiltered daily `thetao` supplies L4 SST observations. The SST correction
`Fsst` is an explicit source in the conservative tracer equation, not a dynamic
layer forcing. Outputs must retain named layer states, physical depths,
controls, and diagnosed modal speeds.
