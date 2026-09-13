import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from src.mod import Model_qgsw


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "model_qgsw"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from sw import SW


def test_one_layer_closure_uses_exactly_two_independent_fields():
    c = np.full((1, 3, 2), 2.0)
    H = np.full((1, 3, 2), 400.0)
    g_prime = np.full((1, 3, 2), 0.01)

    c_out, H_out, gp_out = Model_qgsw._resolve_one_layer_closure(c, H, None)
    np.testing.assert_allclose(H_out, H)
    np.testing.assert_allclose(gp_out, g_prime)
    np.testing.assert_allclose(c_out**2, gp_out * H_out)

    c_out, H_out, gp_out = Model_qgsw._resolve_one_layer_closure(c, None, g_prime)
    np.testing.assert_allclose(H_out, H)
    np.testing.assert_allclose(gp_out, g_prime)

    # H and g' are authoritative: an inconsistent c input is replaced.
    c_out, H_out, gp_out = Model_qgsw._resolve_one_layer_closure(
        np.full_like(c, 99.0), H, g_prime)
    np.testing.assert_allclose(c_out, c)
    np.testing.assert_allclose(c_out**2, gp_out * H_out)


def test_reduced_gravity_log_control_is_positive_and_diagnoses_c():
    model = Model_qgsw.__new__(Model_qgsw)
    model.dtype = jnp.float32
    model._modal_multilayer = False
    model.g_prime = np.full((1, 3, 2), 0.01, dtype=np.float32)
    model.H0 = np.full((1, 3, 2), 400.0, dtype=np.float32)
    model.H_floor = np.zeros((1, 3, 2), dtype=np.float32)
    model.H_max_bound = None
    model.name_params = ['H', 'g_prime']

    control = jnp.full((2, 3), jnp.log(2.0), dtype=jnp.float32)
    total = model._g_prime_control_to_total(control)
    np.testing.assert_allclose(total, 0.02, rtol=1.e-6)
    assert np.all(np.asarray(total) > 0.)

    state = SimpleNamespace(params={
        'H': jnp.zeros((2, 3), dtype=jnp.float32),
        'g_prime': control,
    })
    np.testing.assert_allclose(
        model._controlled_phase_speed_state(state),
        np.sqrt(0.02 * 400.0),
        rtol=1.e-6,
    )

    derivative = jax.grad(
        lambda x: model._g_prime_control_to_total(x).sum())(control)
    np.testing.assert_allclose(derivative, 0.02, rtol=1.e-6)


def test_sw_step_is_differentiable_with_respect_to_static_g_prime():
    nx, ny = 8, 7
    params = dict(
        nx=nx, ny=ny, nl=1,
        dx=jnp.ones((nx, ny))*10_000.,
        dy=jnp.ones((nx, ny))*10_000.,
        H=jnp.full((1, 1, 1), 400.),
        g_prime=jnp.array([0.01]),
        f=jnp.ones((nx+1, ny+1))*1.e-4,
        taux=0., tauy=0., bottom_drag_coef=0., rho_water=1025.,
        dtype=jnp.float32, mask=np.ones((nx, ny), dtype=int), compile=False,
        slip_coef=1., visc_coef=0., diff_coef=0., dt=10.,
        barotropic_filter=False, sponge_coef=0., forcing_momentum='direct',
        H_min=None, H_max=None, diff_coef_trac=0., time_scheme='rk2',
        h_adv_scheme='rusanov1', mom_adv_scheme='upwind3',
        tracer_adv_scheme='rusanov1', solver='dst',
    )
    model = SW(params)
    u0 = jnp.zeros((1, 1, nx+1, ny), dtype=jnp.float32)
    v0 = jnp.zeros((1, 1, nx, ny+1), dtype=jnp.float32)
    x = jnp.linspace(-1., 1., nx)[:, None]
    h0 = jnp.broadcast_to(0.1*x, (1, 1, nx, ny))

    def response(log_control):
        gp = jnp.full((1, nx, ny), 0.01*jnp.exp(log_control))
        u1, _, _ = model.step(u0, v0, h0, nstep=1, g_prime=gp)
        return jnp.vdot(u1, u1)

    value, derivative = jax.value_and_grad(response)(jnp.asarray(0., dtype=jnp.float32))
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
    assert abs(float(derivative)) > 0.


def test_qgsw_wrapper_tangent_and_adjoint_include_g_prime_control():
    nx, ny = 6, 5
    params = dict(
        nx=nx, ny=ny, nl=1,
        dx=jnp.ones((nx, ny))*10_000., dy=jnp.ones((nx, ny))*10_000.,
        H=jnp.full((1, 1, 1), 400.), g_prime=jnp.array([0.01]),
        f=jnp.ones((nx+1, ny+1))*1.e-4,
        taux=0., tauy=0., bottom_drag_coef=0., rho_water=1025.,
        dtype=jnp.float32, mask=np.ones((nx, ny), dtype=int), compile=False,
        slip_coef=1., visc_coef=0., diff_coef=0., dt=10.,
        barotropic_filter=False, sponge_coef=0., forcing_momentum='direct',
        H_min=None, H_max=None, diff_coef_trac=0., time_scheme='rk2',
        h_adv_scheme='rusanov1', mom_adv_scheme='upwind3',
        tracer_adv_scheme='rusanov1', solver='dst',
    )
    core = SW(params)
    model = Model_qgsw.__new__(Model_qgsw)
    model.nx, model.ny, model.nl = nx, ny, 1
    model.dtype = jnp.float32
    model.model = core
    model.model_step = core.step
    model.jstep_core_jit = model.jstep_core
    model.g_prime = np.full((1, 1, 1), 0.01, dtype=np.float32)
    model.H0 = np.full((1, 1, 1), 400., dtype=np.float32)
    model.H_floor = np.zeros((1, 1, 1), dtype=np.float32)
    model.H_max_bound = None
    model._modal_multilayer = False
    model._modal_layer_stack = False
    model._physical_interface_height = False
    model.anomaly_reference_state = False
    model._anomaly_reference_initialized = False
    model.mdt = model.mdu = model.mdv = None
    model.qg_balanced_sponge_bc = False
    model.sponge_target = 'bc'
    model.wind_use_instantaneous_top_depth = False
    model._controls_h_wind = False
    model._prescribes_h_wind = False
    model._wind_depth_from_H = False

    u0 = jnp.zeros((1, 1, nx+1, ny), dtype=jnp.float32)
    v0 = jnp.zeros((1, 1, nx, ny+1), dtype=jnp.float32)
    h0 = jnp.broadcast_to(
        jnp.linspace(-0.1, 0.1, nx)[:, None], (1, 1, nx, ny))
    gp = jnp.zeros((1, nx, ny), dtype=jnp.float32)
    dgp = jnp.ones_like(gp)*1.e-3
    Fu = jnp.zeros((ny, nx+1), dtype=jnp.float32)
    Fv = jnp.zeros((ny+1, nx), dtype=jnp.float32)
    Fh = jnp.zeros((ny, nx), dtype=jnp.float32)
    ub, vb, hb = jnp.zeros_like(Fu), jnp.zeros_like(Fv), jnp.zeros_like(Fh)

    du, dv, dh = model.jstep_tgl(
        0., jnp.zeros_like(u0), jnp.zeros_like(v0), jnp.zeros_like(h0),
        None, dgp, Fu, Fv, Fh,
        u0, v0, h0, None, gp, Fu, Fv, Fh, ub, vb, hb, nstep=1)
    assert np.isfinite(np.asarray(du)).all()
    assert np.linalg.norm(np.asarray(du)) > 0.

    cot_u = jnp.ones_like(du)
    cot_v = jnp.zeros_like(dv)
    cot_h = jnp.zeros_like(dh)
    zeros_gp = jnp.zeros_like(gp)
    result = model.jstep_adj(
        0., cot_u, cot_v, cot_h, None, zeros_gp,
        Fu, Fv, Fh, u0, v0, h0, None, gp, Fu, Fv, Fh,
        ub, vb, hb, nstep=1)
    adgp = result[4]
    lhs = jnp.vdot(du, cot_u) + jnp.vdot(dv, cot_v) + jnp.vdot(dh, cot_h)
    rhs = jnp.vdot(dgp, adgp)
    np.testing.assert_allclose(lhs, rhs, rtol=2.e-4, atol=1.e-8)
