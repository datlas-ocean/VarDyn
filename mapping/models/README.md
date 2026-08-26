# Dynamical models

This directory contains three JAX implementations used by VarDyn. The equations below describe physical variables; `qgsw` internally uses metric-scaled C-grid variables on non-uniform grids.

Notation: $\eta$ is sea-surface/interface displacement, $h$ a layer-thickness anomaly, $H$ its reference thickness, $\mathbf{u}=(u,v)$ horizontal velocity, $f$ the Coriolis parameter, $f_0$ its reference value, $g$ gravity, $g'_k$ reduced gravity, and $\zeta=v_x-u_y$ relative vorticity.

## `model_qg1l`: one-and-a-half-layer QG

`jqgm.py` advances potential-vorticity (PV) anomalies and diagnoses SSH and geostrophic velocity. With the default `formulation='ssh'`,

```math
\mathbf{u}_g = \frac{g}{f_0}\,\mathbf{k}\times\nabla\eta,
\qquad
q = \frac{g}{f_0}\nabla^2\eta - \frac{g f_0}{c^2}\eta.
```

Here $c$ is the first baroclinic phase speed, so $R_d=c/|f_0|$. Equivalently, the optional streamfunction formulation, $\psi=g\eta/f$, uses

```math
\mathbf{u}_g=\mathbf{k}\times\nabla\psi,
\qquad q=\nabla^2\psi-\frac{\psi}{R_d^2}.
```

The interior PV tendency is

```math
\frac{\partial q}{\partial t}
+ \mathbf{u}_g\cdot\nabla q
+ \beta v
+ \mathbf{u}_g\cdot\nabla q_b
= K_q\nabla^2q.
```

The $\beta v$ term is used with spatially varying Coriolis parameter, and $q_b=f_0 b$ (or $fb$ in streamfunction form) is the optional bathymetric PV contribution. PV advection and diffusion are optional; after each update, the Helmholtz relation is inverted using SSH boundary values. Optional passive tracers satisfy

```math
\frac{\partial C}{\partial t}
+ (\mathbf{u}_g+\mathbf{u}_a)\cdot\nabla C
=K_C\nabla^2C,
```

where $\mathbf{u}_a$ is included only when ageostrophic velocities are configured. Transport uses a selectable upwind stencil; Euler, RK2, and RK3 are available.

## `model_sw1l`: linear one-layer shallow water

`jswm.py` (`CSWm`) advances C-grid $u$, $v$, and $h$ about an equivalent depth $H_e$. Without optional background-flow terms, it solves

```math
\begin{aligned}
u_t - fv &= -g\,\eta_x,\\
v_t + fu &= -g\,\eta_y,\\
\eta_t + H_e\,(u_x+v_y) &= 0.
\end{aligned}
```

The gravity-wave speed is $c=\sqrt{gH_e}$. The model can add linear dynamics about prescribed mean fields: mean horizontal advection of $u$, $v$, and $h$; vertical-shear terms proportional to perturbation divergence in momentum; and prescribed boundary-wave forcing. Those advection terms use a third-order upwind scheme. Euler, RK3, RK4, and leapfrog are implemented (the configured default is RK4).

## `model_qgsw`: multilayer rotating shallow water and projected QG

`model_qgsw/sw.py` provides an $N$-layer nonlinear rotating shallow-water model. In physical form, layer $k$ obeys

```math
\begin{aligned}
\partial_t\mathbf{u}_k
+ (\mathbf{u}_k\cdot\nabla)\mathbf{u}_k
+ f\,\mathbf{k}\times\mathbf{u}_k
&= -\nabla p_k + \mathbf{F}_k + \nu\nabla^2\mathbf{u}_k,\\
\partial_t h_k + \nabla\cdot[(H_k+h_k)\mathbf{u}_k]
&= \kappa\nabla^2h_k.
\end{aligned}
```

Hydrostatic pressure is built from stacked interfaces,

```math
\eta_k=-\sum_m H_m+\sum_{m\geq k}(H_m+h_m),
\qquad p_k=\sum_{m\leq k}g'_m\eta_m.
```

The code uses a vector-invariant momentum form (vortex force, kinetic-energy, and pressure-gradient terms), finite-volume thickness fluxes, and C-grid land masks. Thickness fluxes and vortex-force reconstruction are WENO by default; fixed upwind alternatives are available. Optional terms are top-layer wind stress, bottom linear drag, Laplacian viscosity/diffusion, sponge relaxation to boundary fields, and a barotropic-wave filter. Time integration is SSP-RK3 by default, with RK2 variants available.

### Single-layer reduced-gravity case (`nl=1`)

With `nl=1`, `model_qgsw` represents one active layer of reference (equivalent) depth $H$ over a motionless deep layer. The sole interface displacement is $\eta=h$, so the pressure is $p=g'\eta$, where $g'$ is the reduced gravity. The nonlinear rotating shallow-water equations therefore reduce to

```math
\begin{aligned}
\partial_t\mathbf{u}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
+ f\,\mathbf{k}\times\mathbf{u}
&= -g'\nabla\eta + \mathbf{F} + \nu\nabla^2\mathbf{u},\\
\partial_t\eta + \nabla\cdot[(H+\eta)\mathbf{u}]
&= \kappa\nabla^2\eta.
\end{aligned}
```

The associated internal gravity-wave speed and deformation radius are $c=\sqrt{g'H}$ and $R_d=c/|f_0|$. Here $H$ is an equivalent depth: when wind stress is enabled, `h_wind` can be set separately to the physical mixed-layer depth used to convert stress to acceleration.

For `name_class='QG'`, the same `nl=1` setup is projected onto the 1.5-layer QG balance,

```math
\mathbf{u}=\frac{g'}{f_0}\mathbf{k}\times\nabla\eta,
\qquad
(\nabla^2-R_d^{-2})\,p=q,
\qquad p=g'\eta.
```

This is the reduced-gravity counterpart of the multilayer QG relation below; it is implemented by the single coupling coefficient $A=(Hg')^{-1}$.

Selecting `name_class='QG'` uses `model_qgsw/qg.py`. It advances the same forcing/tendency machinery but projects it onto the multilayer QG manifold at each stage:

```math
\mathbf{u}_k=\frac{1}{f_0}\mathbf{k}\times\nabla p_k,
\qquad h_k=H_k\sum_m A_{km}p_m,
```

where $A$ is the layer-coupling matrix constructed from $H_k$ and $g'_k$. The elliptic PV relation is diagonalised into vertical modes,

```math
(\nabla^2-f_0^2\lambda_r)\,p_r=q_r,
```

with $\lambda_r$ the eigenvalues of $A$. Each mode is solved with a discrete Helmholtz inversion, optionally corrected with a capacitance matrix at irregular coastlines, then transformed back to layers. Thus this QG option enforces balanced multilayer evolution; it is distinct from the separate 1.5-layer PV-advection model above.

## Numerical and differentiation notes

All three implementations expose tangent-linear and adjoint paths through JAX automatic differentiation. Boundary handling, masks, and supplied forcing fields are part of the discrete model; this README records the continuous equations and principal discrete choices.
