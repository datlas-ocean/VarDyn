import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


MAPPING_DIR = Path(__file__).resolve().parents[1]
if str(MAPPING_DIR) not in sys.path:
    sys.path.insert(0, str(MAPPING_DIR))
MODEL_DIR = MAPPING_DIR / "models" / "model_qgsw"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from sw import SW, replicate_pad


def _params(forcing_momentum):
    nx, ny = 5, 4
    return dict(
        nx=nx, ny=ny, nl=1,
        dx=jnp.ones((nx, ny)) * 10_000.,
        dy=jnp.ones((nx, ny)) * 10_000.,
        H=jnp.full((1, 1, 1), 300.),
        g_prime=jnp.array([0.01]),
        f=jnp.zeros((nx + 1, ny + 1)),
        taux=0., tauy=0., bottom_drag_coef=0., rho_water=1025.,
        h_wind=100.,
        dtype=jnp.float32, mask=np.ones((nx, ny), dtype=int), compile=False,
        slip_coef=1., visc_coef=0., diff_coef=0., dt=900.,
        barotropic_filter=False, sponge_coef=0.,
        forcing_momentum=forcing_momentum,
        H_min=None, H_max=None, diff_coef_trac=0., time_scheme='rk2',
        h_adv_scheme='rusanov1', mom_adv_scheme='upwind3',
        tracer_adv_scheme='rusanov1', solver='dst',
        wind_use_instantaneous_top_depth=True,
    )


def _face_thicknesses(model, h, h_ref_u, h_ref_v):
    h_pad = replicate_pad(h, model.masks.h)
    h_u = 0.5 * (h_pad[..., 1:, 1:-1] + h_pad[..., :-1, 1:-1])
    h_v = 0.5 * (h_pad[..., 1:-1, 1:] + h_pad[..., 1:-1, :-1])
    return h_ref_u + h_u, h_ref_v + h_v


def test_zero_momentum_mass_source_preserves_face_momentum_exactly():
    model = SW(_params('zero_momentum_mass_source'))
    nx, ny = model.nx, model.ny
    u_phys = jnp.full((1, 1, nx + 1, ny), 0.4)
    v_phys = jnp.full((1, 1, nx, ny + 1), -0.2)
    h_phys = jnp.full((1, 1, nx, ny), 2.)
    Fh_phys = jnp.full_like(h_phys, 0.01)
    zero_u = jnp.zeros_like(u_phys)
    zero_v = jnp.zeros_like(v_phys)

    u, v, h = model.set_input_uvh(u_phys, v_phys, h_phys)
    Fu, Fv, Fh = model.set_input_uvh(zero_u, zero_v, Fh_phys)
    ref_vals = model._compute_ref_values(model.H)
    old_h_u, old_h_v = _face_thicknesses(model, h, ref_vals[1], ref_vals[2])

    new_u, new_v, new_h = model._apply_external_forcing(
        u, v, h, Fu, Fv, Fh, ref_vals)
    new_h_u, new_h_v = _face_thicknesses(
        model, new_h, ref_vals[1], ref_vals[2])

    np.testing.assert_allclose(
        np.asarray(new_h_u * new_u), np.asarray(old_h_u * u), rtol=2.e-6)
    np.testing.assert_allclose(
        np.asarray(new_h_v * new_v), np.asarray(old_h_v * v), rtol=2.e-6)


def test_mass_consistent_is_a_deprecated_alias():
    with pytest.warns(DeprecationWarning, match='zero_momentum_mass_source'):
        model = SW(_params('mass_consistent'))
    assert model.forcing_momentum == 'zero_momentum_mass_source'


def test_invalid_forcing_momentum_is_rejected():
    with pytest.raises(ValueError, match='forcing_momentum must be one of'):
        SW(_params('mass_conistent'))
