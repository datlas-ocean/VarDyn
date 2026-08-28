# One-Layer Public QGSW Interface

**Status:** Accepted

Public `MOD_QGSW` configuration represents one baroclinic layer. Its height is
either the historical direct SSH coordinate or a physical reduced-gravity
Interface Displacement. Modal layer stacks, mixed-layer and Ekman roles,
fixed-slab closures, layer-resolved depth controls, modal tracer closures, and
their diagnostics are not part of this interface.

The shallow-water implementation is selected unconditionally. QG experiments
use the separate `MOD_QG1L` interface; `MOD_QGSW` has no model-class selector.

The Generic Multilayer SW Core remains internal so its numerical implementation
and tests can support future model modules. It is not selected or parameterised
through `MOD_QGSW`. A future public multilayer model must introduce a separate,
coherent configuration block instead of widening `MOD_QGSW` again.

**Considered options**

- Keep every experimental modal and Ekman parameter in `MOD_QGSW` defaults.
- Remove defaults while silently accepting modal height representations.
- Retain the Generic Multilayer SW Core internally behind the smaller public
  one-layer interface.

**Consequences**

`MOD_QGSW` accepts only `height_representation='ssh'` and
`height_representation='interface_displacement'`. Modal and Ekman configuration
now fails at model construction with a direct migration error. Internal SW
defaults use no fixed Ekman slabs, top-layer wind stress, concentration tracer
advection, the Boundary Condition sponge target, and the reference pressure
gradient. Existing QG experiments must inherit `MOD_QG1L`.
