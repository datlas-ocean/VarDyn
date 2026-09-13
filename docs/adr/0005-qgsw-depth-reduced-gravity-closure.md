# QGSW Depth–Reduced-Gravity Closure

**Status:** Accepted

The public one-layer `MOD_QGSW` interface uses the physical closure
`c**2 = g_prime * H`. Reference Layer Depth and Reference Reduced Gravity are
the authoritative dynamical parameters when both are configured; Reference
First-Baroclinic Speed is then diagnosed. When exactly one of `H` and
`g_prime` is absent, the configured phase-speed field diagnoses it. The
historical direct-SSH default with both absent retains physical gravity and
diagnoses Equivalent Depth from phase speed.

`H` and `g_prime` may both appear in `name_params`. Each denotes a
dimensionless logarithmic control and therefore produces a strictly positive
physical field. Controlled phase speed is always diagnosed as
`sqrt(g_prime * H)`; it is not an independent control.

Reduced-Gravity Control is static during each forward integration. It is not
an advected density variable and does not turn the model into thermal shallow
water. The shallow-water pressure implementation receives Controlled Reduced
Gravity on the differentiated JAX path.

In physical Interface Displacement mode, `State` retains a fixed reference
pressure coordinate so that physical SSH observations do not change merely
because the parameter control changes. The QGSW wrapper converts this
coordinate to instantaneous Interface Displacement with Controlled Reduced
Gravity inside the differentiated step, and converts it back afterward.

**Consequences**

Outputs and restarts distinguish `g_prime`, the physical Controlled Reduced
Gravity, from `g_prime_control`, its dimensionless logarithmic coordinate.
Likewise, controlled phase speed is a diagnostic. Configurations specifying
all three reference quantities are accepted only through the authoritative
`H`–`g_prime` pair; any phase-speed input is replaced by their diagnosed,
physically consistent value.
