from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from src.mod import M, Model_diffusion, Model_multi, Model_qgsw


def _scan_model():
    model = Model_qgsw.__new__(Model_qgsw)
    model.ny = 2
    model.nx = 3
    model.dt = 10
    model.max_nstep = 2
    model.dtype = jnp.float32
    model.advect_tracer = True
    model._apply_bc = lambda t0, t1: (
        np.full((2, 4), t0, dtype=np.float32),
        np.full((3, 3), t0 + 1, dtype=np.float32),
        np.full((2, 3), t0 + 2, dtype=np.float32),
    )
    model._apply_bc_tracer = lambda t0, t1: jnp.full(
        (1, 2, 3), t0 + 3, dtype=jnp.float32
    )
    model._get_wind_stress = lambda t0: (
        jnp.full((2, 4), t0 + 4, dtype=jnp.float32),
        jnp.full((3, 3), t0 + 5, dtype=jnp.float32),
    )
    return model


def test_qgsw_scan_forcing_covers_intervals_and_internal_chunks():
    model = _scan_model()
    forcing = model.prepare_scan_boundary_conditions(
        np.array([0, 40]),
        nstep=5,
    )

    expected_shapes = (
        (2, 3, 2, 4),
        (2, 3, 3, 3),
        (2, 3, 2, 3),
        (2, 3, 1, 2, 3),
        (2, 3, 2, 4),
        (2, 3, 3, 3),
    )
    assert tuple(value.shape for value in forcing) == expected_shapes

    second_interval = tuple(value[1] for value in forcing)
    selected = model._chunk_forcing(second_interval, 2, None, None)
    np.testing.assert_allclose(selected[2], 82.0)
    np.testing.assert_allclose(selected[3], 83.0)
    np.testing.assert_allclose(selected[4], 84.0)


def test_finite_model_array_is_traceable():
    model = M.__new__(M)

    @jax.jit
    def finite(values):
        state = SimpleNamespace(var={'x': values})
        return model._finite_model_array(state, 'x')

    result = finite(jnp.array([1.0, jnp.nan, jnp.inf], dtype=jnp.float32))
    np.testing.assert_allclose(result, np.array([1.0, 0.0, 0.0]))


def test_qgsw_scan_forcing_preserves_constant_wind():
    model = _scan_model()
    model._get_wind_stress = lambda t0: (None, None)
    model.model = SimpleNamespace(taux=0.25, tauy=-0.5)

    forcing = model.prepare_scan_boundary_conditions(
        np.array([0]),
        nstep=2,
    )

    np.testing.assert_allclose(forcing[4], 0.25)
    np.testing.assert_allclose(forcing[5], -0.5)


class _ScanAwareComponent:
    dt = 5
    name_var = {}

    def prepare_scan_boundary_conditions(self, times, nstep):
        self.prepared_nstep = nstep
        return jnp.asarray(times, dtype=jnp.float32) + 1

    def step(self, state, nstep=1, t=None, Xb=None):
        assert Xb is not None
        state.var['x'] = state.var['x'] + Xb

    def step_adj(self, adstate, state, nstep=1, t=None, Xb=None):
        assert Xb is not None
        adstate.var['x'] = adstate.var['x'] + Xb


class _LegacyComponent:
    dt = 10
    name_var = {}

    def step(self, state, nstep=1, t=None):
        pass

    def step_adj(self, adstate, state, nstep=1, t=None):
        pass


def test_multi_model_forwards_scan_boundary_conditions_under_jit():
    scan_aware = _ScanAwareComponent()
    model = Model_multi.__new__(Model_multi)
    model.Models = [scan_aware, _LegacyComponent()]
    model.dt = 10
    model.name_var = {}
    model.name_var_tot = {}
    times = np.array([0, 10])
    forcing = model.prepare_scan_boundary_conditions(times, nstep=2)
    assert scan_aware.prepared_nstep == 4
    assert forcing[1] is None

    @jax.jit
    def run(initial):
        def forward_body(value, inputs):
            t, boundary = inputs
            state = SimpleNamespace(var={'x': value})
            model.step(state, nstep=2, t=t, Xb=boundary)
            return state.var['x'], None

        forward, _ = jax.lax.scan(
            forward_body, initial, (jnp.asarray(times), forcing)
        )

        def backward_body(value, inputs):
            t, boundary = inputs
            state = SimpleNamespace(var={'x': jnp.asarray(0.0)})
            adjoint = SimpleNamespace(var={'x': value})
            model.step_adj(
                adjoint, state, nstep=2, t=t, Xb=boundary
            )
            return adjoint.var['x'], None

        adjoint, _ = jax.lax.scan(
            backward_body, initial, (jnp.asarray(times), forcing), reverse=True
        )
        return forward, adjoint

    forward, adjoint = run(jnp.asarray(0.0, dtype=jnp.float32))
    np.testing.assert_allclose(forward, 12.0)
    np.testing.assert_allclose(adjoint, 12.0)


class _TraceState:
    def __init__(self, var, params):
        self.var = var
        self.params = params

    def getvar(self, name):
        return self.var[name]

    def setvar(self, value, name):
        self.var[name] = value


def test_diffusion_adjoint_is_traceable_with_identity_configuration():
    model = Model_diffusion.__new__(Model_diffusion)
    model.name_var = {'SSH': 'sla_barotrop'}
    model.Kdiffus = 0
    model.dt = 1800

    @jax.jit
    def run(adjoint_value):
        zeros = jnp.zeros_like(adjoint_value)
        state = _TraceState({'sla_barotrop': zeros}, {'sla_barotrop': zeros})
        adjoint = _TraceState({'sla_barotrop': adjoint_value}, {'sla_barotrop': zeros})
        model.step_adj(adjoint, state, nstep=12, t=jnp.asarray(0))
        return adjoint.var['sla_barotrop'], adjoint.params['sla_barotrop']

    values = jnp.array([[1.0, jnp.nan], [2.0, 3.0]], dtype=jnp.float32)
    state_adjoint, parameter_adjoint = run(values)
    expected = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(state_adjoint, expected)
    np.testing.assert_allclose(parameter_adjoint, expected * 0.25)


def test_diffusion_boundary_conditions_use_model_time_keys():
    model = Model_diffusion.__new__(Model_diffusion)
    model._bc_fields = SimpleNamespace(
        interp=lambda dates: {
            'SSH': np.stack([
                np.full((2, 2), 3.0),
                np.full((2, 2), 4.0),
            ])
        })
    model.name_var = {'SSH': 'sla'}
    model.bc = {'SSH': {}}
    model.init_from_bc = True
    model._set_land_nan = lambda state: None

    dates = np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]')
    model.set_bc(dates, t_bc=np.array([0, 86400]))

    state = _TraceState({'sla': np.zeros((2, 2))}, {})
    model.init(state, t0=0)
    np.testing.assert_allclose(state.var['sla'], 3.0)
