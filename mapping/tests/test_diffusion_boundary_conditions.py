from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from src.mod import Model_diffusion


class _State:
    def __init__(self, values):
        self.var = {'ssh': values}
        self.params = {'ssh': jnp.zeros_like(values)}
        self.mask = None

    def getvar(self, name):
        return self.var[name]

    def setvar(self, value, name):
        self.var[name] = value


def _model(*, sponge_active=False):
    model = Model_diffusion.__new__(Model_diffusion)
    model.name_var = {'SSH': 'ssh'}
    model.ny = 2
    model.nx = 2
    model.dt = 10
    model.Kdiffus = 0
    model.sponge_active = sponge_active
    model.sponge_coef = 0.5
    model.Wbc_device = jnp.array([[1.0, 0.0], [0.5, 0.0]])
    model.bc = {'SSH': {}}
    model.timestamps = np.asarray([datetime(2024, 1, 1)])
    model.init_from_bc = False
    return model


def test_diffusion_sponge_requires_positive_distance_and_coefficient():
    enabled = Model_diffusion._sponge_is_configured
    assert enabled(100, 0.05)
    for distance, coefficient in (
        (None, 0.05), (0, 0.05), (-1, 0.05),
        (100, None), (100, 0), (100, -0.05),
    ):
        assert not enabled(distance, coefficient)


def test_diffusion_initialization_uses_numeric_model_time_keys():
    model = _model()
    model.init_from_bc = True
    dates = np.asarray(['2024-01-01', '2024-01-02'], dtype='datetime64[D]')
    values = np.asarray([
        np.full((2, 2), 3.0),
        np.full((2, 2), 7.0),
    ])
    model.set_bc(dates, {'SSH': values}, t_bc=np.asarray([0, 86400]))

    state = _State(jnp.zeros((2, 2)))
    model.init(state, t0=1)

    np.testing.assert_allclose(state.var['ssh'], 3.0)
    assert list(model.bc['SSH']) == [0, 86400]


def test_diffusion_sponge_relaxes_once_per_model_step():
    model = _model(sponge_active=True)
    boundary = (jnp.full((1, 2, 2), 8.0), jnp.asarray([True]))
    state = _State(jnp.zeros((2, 2)))

    model.step(state, nstep=2, t=0, Xb=boundary)

    np.testing.assert_allclose(
        state.var['ssh'],
        np.asarray([[6.0, 0.0], [3.5, 0.0]]),
    )


def test_diffusion_sponge_is_inert_when_not_active():
    model = _model(sponge_active=False)
    boundary = (jnp.full((1, 2, 2), 8.0), jnp.asarray([True]))
    state = _State(jnp.zeros((2, 2)))

    model.step(state, nstep=2, t=0, Xb=boundary)

    np.testing.assert_allclose(state.var['ssh'], 0.0)

    scan_values, scan_available = model.prepare_scan_boundary_conditions(
        np.asarray([0, 10]), nstep=1)
    assert scan_values.shape == (2, 0)
    assert scan_available.shape == (2, 0)


def test_diffusion_sponge_tangent_and_adjoint_match():
    model = _model(sponge_active=True)
    boundary = (jnp.full((1, 2, 2), 8.0), jnp.asarray([True]))
    tangent_input = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    adjoint_input = jnp.asarray([[4.0, 3.0], [2.0, 1.0]])

    tangent = _State(tangent_input)
    model.step_tgl(tangent, _State(jnp.zeros((2, 2))), nstep=2, t=0, Xb=boundary)
    adjoint = _State(adjoint_input)
    model.step_adj(adjoint, _State(jnp.zeros((2, 2))), nstep=2, t=0, Xb=boundary)

    lhs = jnp.vdot(tangent.var['ssh'], adjoint_input)
    rhs = jnp.vdot(tangent_input, adjoint.var['ssh'])
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6)

    # The explicit boundary input also remains usable inside a compiled step.
    @jax.jit
    def compiled(values):
        state = _State(values)
        model.step(state, nstep=1, t=jnp.asarray(0), Xb=boundary)
        return state.var['ssh']

    np.testing.assert_allclose(compiled(jnp.zeros((2, 2)))[0, 0], 4.0)


def test_diffusion_scan_boundary_uses_interval_end_time():
    model = _model(sponge_active=True)
    model.bc['SSH'] = {
        10: np.full((2, 2), 1.0),
        30: np.full((2, 2), 3.0),
    }

    values, available = model.prepare_scan_boundary_conditions(
        np.asarray([0, 20]), nstep=1)

    np.testing.assert_allclose(values[:, 0, 0, 0], [1.0, 3.0])
    np.testing.assert_array_equal(available, [[True], [True]])


def test_diffusion_does_not_propagate_land_nans_into_ocean():
    model = _model()
    model.ny = model.nx = 4
    model.dt = 1
    model.Kdiffus = 0.1
    model.dx = np.ones((4, 4))
    model.dy = np.ones((4, 4))
    land = np.zeros((4, 4), dtype=bool)
    land[1, 1] = True
    model.land_mask_device = jnp.asarray(land)
    model.Wbc_device = jnp.zeros((4, 4))

    values = np.ones((4, 4))
    values[land] = np.nan
    state = _State(jnp.asarray(values))
    model.step(state, nstep=1, t=0)

    result = np.asarray(state.var['ssh'])
    assert np.isnan(result[land]).all()
    np.testing.assert_allclose(result[~land], 1.0)


def test_mask_aware_diffusion_tangent_and_adjoint_match():
    model = _model()
    model.ny = model.nx = 4
    model.dt = 1
    model.Kdiffus = 0.1
    model.dx = np.ones((4, 4))
    model.dy = np.ones((4, 4))
    land = np.zeros((4, 4), dtype=bool)
    land[1, 1] = True
    model.land_mask_device = jnp.asarray(land)
    model.Wbc_device = jnp.zeros((4, 4))
    tangent_input = jnp.arange(16, dtype=float).reshape(4, 4)
    adjoint_input = jnp.arange(16, 32, dtype=float).reshape(4, 4)

    tangent = _State(tangent_input)
    model.step_tgl(tangent, _State(jnp.zeros((4, 4))), nstep=2, t=0)
    adjoint = _State(adjoint_input)
    model.step_adj(adjoint, _State(jnp.zeros((4, 4))), nstep=2, t=0)

    lhs = jnp.vdot(tangent.var['ssh'], adjoint_input)
    rhs = jnp.vdot(tangent_input, adjoint.var['ssh'])
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6)
