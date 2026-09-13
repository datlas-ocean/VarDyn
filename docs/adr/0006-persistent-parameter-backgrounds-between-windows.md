# Persistent Parameter Backgrounds Between Assimilation Windows

**Status:** Accepted

Windowed assimilation restarts the prognostic state from the preceding
window. Selected controlled model parameters may also persist, but their
reduced-basis coefficients remain window-local increments. Each reduced-basis
component opts in next to the parameter that it controls:

```python
myBASIS1_H = dict(
    super='BASIS_GAUSS2D',
    name_mod_var='H',
    use_state_background=True,
)
```

The default is `False`, which preserves the historical behavior. The option
works for a directly configured single basis and for each component of a
`Basis_multi` configuration.

For every enabled basis, `set_basis(..., State=State)` captures the control
field already loaded into `State.params` from the state restart. Each
subsequent basis projection constructs

```text
parameter_control = restart_background + G @ window_control
```

in model-control coordinates. For positive QGSW parameters this addition is
performed in logarithmic control space, before the physical transforms for
Total Equivalent Depth and Controlled Reduced Gravity.

The restart background is constant within the current variational problem.
Consequently its derivative with respect to the new reduced control is zero,
the tangent operator remains `G`, and the reduced-space adjoint remains
`G.T`. The background must be restored before every projection rather than
accumulated onto the current parameter field.

Only basis targets present in `MOD.name_params` are accepted. Prognostic
correction controls such as `ssh`, `u` and `v` should keep
`use_state_background=False`: their physical state is restarted separately,
while their time-dependent basis coefficients describe new corrections for
each window.

When several basis components target the same model parameter, all components
must configure the same `use_state_background` value. `Basis_multi` restores
the background once per target and then sums every component increment:

```text
parameter_control = restart_background + G1 @ x1 + G2 @ x2 + ...
```

This prevents the restart background from being counted once per component.

**Consequences**

- A zero reduced control preserves every enabled parameter field from the
  preceding window, including when only one basis is configured directly.
- `Xres.nc` represents the reduced increment/background vector of the current
  window; trajectory NetCDF files contain the resulting total physical fields
  and their total control coordinates.
- `INV.path_background` or `INV.path_init_4Dvar` must not reintroduce the same
  persistent parameter contribution, otherwise it is counted twice.
- Same-window minimizer restarts remain valid because they reload the current
  window's reduced control while the parameter background is reconstructed
  from the preceding-window state file.
