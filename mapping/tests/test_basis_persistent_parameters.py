from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from src.basis import Basis, Basis_multi


class _Config(dict):
    __getattr__ = dict.__getitem__


class _AdditiveUnitBasis:
    """Small stand-in for a multi-mode basis component."""

    def __init__(self, name):
        self.name = name
        self.nbasis = 1

    def set_basis(self, time, return_q=False, **kwargs):
        del time, kwargs
        if return_q:
            return np.zeros(1), np.ones(1)

    def operg(self, time, control, State=None):
        del time
        value = control[0] * jnp.ones_like(State[self.name])
        State[self.name] += value

    def operg_transpose(self, time, adState):
        del time
        return jnp.asarray([jnp.sum(adState[self.name])])


def _basis_multi_for_test():
    basis = Basis_multi.__new__(Basis_multi)
    basis.Basis = [_AdditiveUnitBasis('H'), _AdditiveUnitBasis('ssh')]
    basis.name_mod_var = ['H', 'ssh']
    basis.state_background_names = ('H',)
    basis.parameter_background = {}
    return basis


def test_persistent_parameter_is_background_but_window_control_is_zero_based():
    basis = _basis_multi_for_test()
    initial_H = jnp.asarray([[0.2, -0.1]])
    state = SimpleNamespace(params={
        'H': initial_H,
        'ssh': jnp.asarray([[4.0, 5.0]]),
    })
    basis.set_basis(jnp.asarray([0.0]), return_q=True, State=state)

    # Mutating the source State after set_basis must not alter the snapshot.
    state.params['H'] = jnp.zeros_like(initial_H)
    projected = {
        'H': jnp.full_like(initial_H, 99.0),
        'ssh': jnp.full_like(initial_H, 99.0),
    }
    basis.operg(0.0, jnp.asarray([0.0, 0.0]), State=projected)

    np.testing.assert_allclose(projected['H'], initial_H)
    np.testing.assert_allclose(projected['ssh'], 0.0)


def test_direct_single_basis_uses_the_same_state_background_rule():
    background = jnp.asarray([[0.2, -0.1]])
    config = _Config(
        BASIS=_Config(
            super='BASIS_OFFSET',
            name_mod_var='H',
            sigma_B=0.1,
            use_state_background=True,
        ),
        MOD=_Config(super='MOD_QGSW', name_params=['H']),
    )
    state = SimpleNamespace(
        params={'H': background},
        ny=1,
        nx=2,
    )
    direct_basis = Basis(config, state, verbose=False)
    direct_basis.set_basis(
        jnp.asarray([0.0]), return_q=True, State=state)

    projected = {'H': jnp.zeros_like(background)}
    direct_basis.operg(
        0.0, jnp.asarray([0.3]), State=projected)

    np.testing.assert_allclose(
        projected['H'], background + 0.3)


def test_persistent_background_does_not_change_control_jacobian():
    basis = _basis_multi_for_test()
    background = jnp.asarray([[0.2, -0.1]])
    state = SimpleNamespace(params={
        'H': background,
        'ssh': jnp.zeros_like(background),
    })
    basis.set_basis(jnp.asarray([0.0]), return_q=True, State=state)

    def objective(control):
        projected = {
            'H': jnp.zeros_like(background),
            'ssh': jnp.zeros_like(background),
        }
        basis.operg(0.0, control, State=projected)
        return 0.5 * (
            jnp.vdot(projected['H'], projected['H'])
            + jnp.vdot(projected['ssh'], projected['ssh']))

    control = jnp.asarray([0.3, -0.4])
    gradient = jax.grad(objective)(control)
    expected = jnp.asarray([
        jnp.sum(background + control[0]),
        background.size * control[1],
    ])
    np.testing.assert_allclose(gradient, expected, rtol=1.e-6, atol=1.e-7)

    # The constant background contributes nothing to the tangent, so the
    # existing multi-basis transpose remains its exact adjoint.
    direction = jnp.asarray([0.7, -0.2])
    tangent = {
        'H': direction[0] * jnp.ones_like(background),
        'ssh': direction[1] * jnp.ones_like(background),
    }
    cotangent = {
        'H': jnp.asarray([[1.1, -0.3]]),
        'ssh': jnp.asarray([[0.4, 0.8]]),
    }
    lhs = (
        jnp.vdot(tangent['H'], cotangent['H'])
        + jnp.vdot(tangent['ssh'], cotangent['ssh']))
    reduced_cotangent = basis.operg_transpose(
        0.0, adState={name: value.copy()
                      for name, value in cotangent.items()})
    rhs = jnp.vdot(direction, reduced_cotangent)
    np.testing.assert_allclose(lhs, rhs, rtol=1.e-6, atol=1.e-7)
