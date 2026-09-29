"""Empty L4 operators must be neutral inside compiled checkpoint scans."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.obsop import Obsop_interp_l4, Obsop_multi


def _operator(empty):
    operator = object.__new__(Obsop_interp_l4)
    operator.gradients = False
    operator.name_var = 'SSH'
    operator.name_mod_var = {'SSH': 'ssh'}
    operator.observation_role = None
    operator.model_to_observation_scale = 1.0
    operator.DX = jnp.ones((1, 2))
    operator.t_obs_jax = jnp.asarray([] if empty else [0.0])
    operator.varobs_arr = jnp.zeros((0 if empty else 1, 2))
    operator.errobs_arr = jnp.ones_like(operator.varobs_arr)
    return operator


@pytest.mark.parametrize('empty_first', [True, False])
def test_empty_l4_is_neutral_in_mixed_compiled_scan(empty_first):
    empty, active = _operator(True), _operator(False)
    combined = object.__new__(Obsop_multi)
    combined.Obsop = [empty, active] if empty_first else [active, empty]
    field = jnp.asarray([[2.0, -3.0]])

    def cost(value):
        def step(total, t):
            residual = combined.scan_misfit(t, {'ssh': value})
            return total + 0.5 * jnp.vdot(residual, residual), residual
        return jax.lax.scan(step, jnp.asarray(0.0), jnp.asarray([0.0, 1.0]))

    total, residuals = jax.jit(cost)(field)
    assert empty.scan_misfit_size() == 0
    assert combined.scan_misfit_size() == 2
    np.testing.assert_allclose(total, 6.5)
    np.testing.assert_array_equal(residuals, [[2.0, -3.0], [0.0, 0.0]])
    gradient = jax.jit(jax.grad(lambda value: cost(value)[0]))(field)
    adjoint = jax.jit(lambda value: combined.scan_adj(
        0.0, {'ssh': jnp.zeros_like(value)}, {'ssh': value},
        combined.scan_misfit(0.0, {'ssh': value})))(field)
    np.testing.assert_allclose(adjoint['ssh'], gradient)


def test_empty_l4_compiled_scan_has_zero_cost_and_preserves_adjoint():
    operator = _operator(True)
    field = jnp.asarray([[2.0, -3.0]])
    seed = jnp.asarray([[4.0, 5.0]])

    def cost(value):
        _, residuals = jax.lax.scan(
            lambda carry, t: (carry, operator.scan_misfit(t, {'ssh': value})),
            None, jnp.asarray([0.0, 1.0]))
        return jnp.sum(residuals ** 2)

    np.testing.assert_array_equal(jax.jit(jax.grad(cost))(field), 0.0)
    assert float(jax.jit(cost)(field)) == 0.0
    result = jax.jit(lambda t: operator.scan_adj(
        t, {'ssh': seed}, {'ssh': field}, operator.scan_misfit(t, {'ssh': field})))(0.0)
    np.testing.assert_array_equal(result['ssh'], seed)
