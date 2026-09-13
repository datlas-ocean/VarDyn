import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.mod import Model_qgsw


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "model_qgsw"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from sw import SW


def test_three_layer_modal_closure_and_ssh_normalization():
    H = np.stack([
        np.full((3, 2), 10.),
        np.full((3, 2), 10.),
        np.full((3, 2), 100.),
    ])
    c1 = np.full((3, 2), 3.)
    g_prime = Model_qgsw._diagnose_three_layer_gprime(H, c1, .08, .03)

    model = Model_qgsw.__new__(Model_qgsw)
    model.dtype = jnp.float32
    model.nl = 3
    model.g_prime = g_prime
    model.g = 9.81
    height_mode, velocity_mode, speeds = model._stack_eigenmodes(jnp.asarray(H))

    np.testing.assert_allclose(np.asarray(speeds[:, 0, 0]), [3., .24, .09], rtol=2e-6, atol=2e-6)
    ssh = model._stack_diagnose_ssh(np.asarray(height_mode).transpose(0, 2, 1))
    np.testing.assert_allclose(np.asarray(ssh), 1., rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(velocity_mode[0]), 1., rtol=2e-6, atol=2e-6)


def test_upper_layer_tracer_reuses_continuity_mass_flux():
    nx, ny, nl = 12, 10, 3
    params = dict(
        nx=nx, ny=ny, nl=nl,
        dx=jnp.ones((nx, ny))*10_000., dy=jnp.ones((nx, ny))*10_000.,
        H=jnp.array([10., 10., 100.]).reshape(3, 1, 1),
        g_prime=jnp.array([1.e-4, 5.e-4, 1.e-2]),
        f=jnp.ones((nx+1, ny+1))*1.e-5,
        taux=0., tauy=0., bottom_drag_coef=0., rho_water=1025.,
        dtype=jnp.float32, mask=np.ones((nx, ny), dtype=int), compile=False,
        slip_coef=1., visc_coef=0., diff_coef=0., dt=10.,
        barotropic_filter=False, sponge_coef=0., forcing_momentum="direct",
        H_min=None, H_max=None, diff_coef_trac=0., time_scheme="rk2",
        h_adv_scheme="rusanov1", mom_adv_scheme="upwind5",
        tracer_adv_scheme="rusanov1", tracer_conservation="upper_layer",
        tracer_upper_layers=2, solver="dst",
        wind_use_instantaneous_top_depth=True,
    )
    model = SW(params)
    rng = np.random.default_rng(4)
    U = rng.normal(0., 1.e-5, (1, nl, nx+1, ny)).astype(np.float32)
    V = rng.normal(0., 1.e-5, (1, nl, nx, ny+1)).astype(np.float32)
    U[:, :, 1, :] = U[:, :, -2, :] = 0.
    V[:, :, :, 1] = V[:, :, :, -2] = 0.
    U, V = jnp.asarray(U), jnp.asarray(V)
    h = jnp.zeros((1, nl, nx, ny), dtype=jnp.float32)
    h_ref = model._compute_ref_values(model.H)[0]
    temperature = 27.
    upper_depth_area = jnp.sum(h_ref[:2][None], axis=1, keepdims=True)
    heat_content = jnp.ones((1, 1, nx, ny), dtype=jnp.float32)*temperature*upper_depth_area

    d_heat = model.advection_upper_layer_tracer(U, V, heat_content, h, h_ref)
    d_h = model.advection_h(U, V, h, h_ref)
    d_upper_depth = jnp.sum(d_h[:, :2], axis=1, keepdims=True)

    scale = float(np.max(np.abs(np.asarray(d_heat))))
    error = float(np.max(np.abs(np.asarray(d_heat-temperature*d_upper_depth))))
    assert error/max(scale, 1.e-30) < 2.e-6
    assert abs(float(np.asarray(d_heat).sum())) < 2.e-6*float(np.abs(np.asarray(d_heat)).sum())

    # Fsst is a concentration tendency: the conservative integrator converts
    # it to heat-content tendency with the instantaneous upper-layer depth.
    zero_u = jnp.zeros_like(U)
    zero_v = jnp.zeros_like(V)
    zero_h = jnp.zeros_like(h)
    c0 = jnp.ones((1, 1, nx, ny), dtype=jnp.float32)*temperature
    fsst = jnp.ones_like(c0)*0.02
    _, _, _, c1 = model.step_with_tracer(
        zero_u, zero_v, zero_h, c0,
        nstep=1,
        u_b=zero_u, v_b=zero_v, h_b=zero_h, c_b=c0,
        Fu=zero_u, Fv=zero_v, Fh=zero_h, Fc=fsst,
    )
    np.testing.assert_allclose(
        np.asarray(c1), temperature+model.dt*0.02, rtol=2.e-6, atol=2.e-6)


def test_multilayer_step_accepts_traced_state_arrays():
    class DummyState:
        def __init__(self, u_layers, v_layers, h_layers):
            self.var = {
                "u_layers": u_layers,
                "v_layers": v_layers,
                "h_layers": h_layers,
            }
            self.params = {
                "U": jnp.zeros(u_layers.shape[1:]),
                "V": jnp.zeros(v_layers.shape[1:]),
                "SSH": jnp.zeros(h_layers.shape[1:]),
            }

        def setvar(self, value, name_var=None, add=False):
            self.var[name_var] = self.var.get(name_var, 0.) + value if add else value

    model = Model_qgsw.__new__(Model_qgsw)
    model.nl = 2
    model.dtype = jnp.float32
    model.name_var = {"U": "U", "V": "V", "SSH": "SSH"}
    model.name_params = []
    model._modal_multilayer = False
    model._modal_layer_stack = False
    model.anomaly_reference_state = False
    model._anomaly_reference_initialized = False
    model.advect_tracer = False
    model.max_nstep = 0
    model.dt = 1.
    model._chunk_forcing = lambda *args, **kwargs: (None, None, None, None, None, None)
    model.jstep_jit = lambda t, u, v, h, *args, **kwargs: (u, v, h)
    model._set_land_nan = lambda state: None

    def traced_step(u_layers, v_layers, h_layers):
        state = DummyState(u_layers, v_layers, h_layers)
        model.step(state, nstep=1)
        return (state.var["u_layers"], state.var["v_layers"],
                state.var["h_layers"], state.var["SSH"])

    shape = (2, 4, 5)
    inputs = tuple(jnp.ones(shape, dtype=jnp.float32)*(index+1) for index in range(3))
    outputs = jax.jit(traced_step)(*inputs)
    for output in outputs:
        assert bool(jnp.all(jnp.isfinite(output)))

    objective = lambda u: sum(jnp.sum(value) for value in traced_step(u, inputs[1], inputs[2]))
    gradient = jax.jit(jax.grad(objective))(inputs[0])
    np.testing.assert_allclose(np.asarray(gradient), 1., rtol=0., atol=0.)



def test_role_selective_dynamic_projection_is_baroclinic_only():
    nx, ny, nl = 3, 2, 3
    model = Model_qgsw.__new__(Model_qgsw)
    model.dtype = jnp.float32
    model.nl = nl
    model.g = 9.81
    model.g_prime = jnp.stack([
        jnp.full((nx, ny), 1.e-4),
        jnp.full((nx, ny), 5.e-4),
        jnp.full((nx, ny), 1.e-2),
    ])

    u = jnp.arange((nx+1)*ny, dtype=jnp.float32).reshape(nx+1, ny)/10.
    v = jnp.arange(nx*(ny+1), dtype=jnp.float32).reshape(nx, ny+1)/10.
    ssh = jnp.arange(nx*ny, dtype=jnp.float32).reshape(nx, ny)/100.
    H = jnp.ones((nl, nx, ny), dtype=jnp.float32)

    for projector in (model._stack_project_boundary_sw,
                      lambda uu, vv, hh: model._stack_project_surface_sw(uu, vv, hh, H)):
        u_layers, v_layers, h_layers = projector(u, v, ssh)
        np.testing.assert_allclose(np.asarray(u_layers[:-1]), 0.)
        np.testing.assert_allclose(np.asarray(v_layers[:-1]), 0.)
        np.testing.assert_allclose(np.asarray(h_layers[:-1]), 0.)
        np.testing.assert_allclose(np.asarray(u_layers[-1]), np.asarray(u))
        np.testing.assert_allclose(np.asarray(v_layers[-1]), np.asarray(v))
        diagnosed = model._stack_diagnose_ssh(h_layers.transpose(0, 2, 1))
        np.testing.assert_allclose(np.asarray(diagnosed), np.asarray(ssh.T), rtol=2.e-6, atol=2.e-6)


def test_layer_masking_preserves_ocean_nonfinite_values():
    class DummyState:
        mask = np.array([[True, False, False], [False, False, False]])

    model = Model_qgsw.__new__(Model_qgsw)
    values = jnp.array([[jnp.nan, 1., 2.], [3., jnp.nan, 5.]], dtype=jnp.float32)
    masked = np.asarray(model._masked_jax_model_array(DummyState(), values))
    assert masked[0, 0] == 0.
    assert np.isnan(masked[1, 1])


def _wind_depth_selector(**overrides):
    model = Model_qgsw.__new__(Model_qgsw)
    model.dtype = jnp.float32
    model._modal_multilayer = False
    model.wind_use_instantaneous_top_depth = False
    model._controls_h_wind = False
    model._prescribes_h_wind = False
    model._wind_depth_from_H = False
    model.H0 = np.full((1, 2, 3), 80., dtype=np.float32)
    model.H_floor = np.zeros((1, 2, 3), dtype=np.float32)
    model.H_max_bound = None
    for name, value in overrides.items():
        setattr(model, name, value)
    return model


def test_controlled_one_layer_depth_is_selected_for_wind_forcing():
    model = _wind_depth_selector(_wind_depth_from_H=True)
    control = jnp.full((3, 2), jnp.log(1.5), dtype=jnp.float32)

    depth = model._wind_depth_to_model(control, None)

    np.testing.assert_allclose(np.asarray(depth), 120., rtol=2.e-6)
    gradient = jax.grad(
        lambda value: jnp.mean(model._wind_depth_to_model(value, None))
    )(control)
    np.testing.assert_allclose(np.asarray(gradient), 20., rtol=2.e-6)


def test_prescribed_wind_depth_takes_precedence_over_controlled_H():
    model = _wind_depth_selector(
        _prescribes_h_wind=True,
        _wind_depth_from_H=False,
    )
    control = jnp.full((3, 2), jnp.log(1.5), dtype=jnp.float32)

    # None tells SW.step to use its prescribed self.h_wind. Passing the total
    # H here would incorrectly make SW.step add it to that prescribed depth.
    assert model._wind_depth_to_model(control, None) is None


def test_independent_wind_depth_control_returns_model_increment():
    model = _wind_depth_selector(_controls_h_wind=True)
    model.h_wind_ref_sw = jnp.full((2, 3), 100., dtype=jnp.float32)
    model.h_wind_floor_sw = jnp.zeros((2, 3), dtype=jnp.float32)
    model.h_wind_max_bound_sw = None
    control = jnp.full((2, 3), jnp.log(1.2), dtype=jnp.float32)

    increment = model._wind_depth_to_model(None, control)

    np.testing.assert_allclose(np.asarray(increment), 20., rtol=2.e-6)
