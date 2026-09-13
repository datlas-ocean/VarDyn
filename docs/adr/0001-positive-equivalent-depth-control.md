# Positive Equivalent Depth Control

VarDyn will enforce physical consistency of QG/SW Equivalent Depth through a smooth positive control transform relative to Reference Equivalent Depth, rather than relying on hard clipping of an additive correction. The inversion may still optimize an unconstrained Equivalent-Depth Control, but the SW core should receive a positive Total Equivalent Depth so gradients remain useful while negative layer depths are impossible and the neutral control preserves the climatological scale. When a phase-speed floor is configured, the Equivalent-Depth Floor should be derived from it; when no phase-speed floor is configured, the floor is zero unless explicitly overridden. The transform is exponential: `H_total = H_floor + (H_ref - H_floor) * exp(H_control)`, with `H_max` retained only as an optional safety rail.

**Considered Options**

- Additive corrections with optional hard `H_min`/`H_max` clipping.
- Smooth positive transforms such as log or softplus parameterizations.

**Consequences**

Saved outputs should distinguish the dimensionless Equivalent-Depth Control from Total Equivalent Depth, because only Total Equivalent Depth is physically meaningful. Prior widths for this control are interpreted on a log scale, so small values correspond approximately to fractional uncertainty around the reference state.
The existing `name_params = ["H"]` behavior is intentionally redefined: it now denotes Equivalent-Depth Control, not an additive meter-valued correction. Existing experiments that estimated `H` must review their prior widths because `sigma_Q` and `sigma_B` move from metre units to log/fractional units. Saved datasets should use `H` for Total Equivalent Depth and `H_control` for the dimensionless Equivalent-Depth Control; every output state should include both fields when `H` is controlled so it is self-describing and restartable. Grid-from-file initialization should read `H_control`, not Total Equivalent Depth, when restarting a controlled `H` experiment. If a controlled `H` restart source lacks `H_control`, VarDyn should fail loudly rather than infer control values from `H`, because old files used `H` for a different quantity.
