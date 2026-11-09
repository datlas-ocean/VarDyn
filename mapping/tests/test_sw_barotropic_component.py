"""The UpperDyn flux component must reproduce MOD_DIFF's checkpoint integral."""

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import CSR

from src.sw_barotropic_component import BarotropicComponent, cumulative_flux_weights


class _SmallBasis:
    flux = True
    time_spinup = None
    sigma_T = 3.
    normalize_fact = True
    norm_fact = 1.7
    ENST = np.arange(-2., 5.)
    Nx = 2
    nbasis = len(ENST) * Nx
    shape_phys = (1, 2)
    Gauss_xy = CSR.fromdense(jnp.array([[1., .25], [.1, 1.]]))


def test_six_hour_flux_accumulates_into_daily_static_ssh():
    basis = _SmallBasis()
    coefficients = jnp.arange(basis.nbasis, dtype=jnp.float32) / 10.
    component = BarotropicComponent(basis, np.ones(basis.nbasis), days=2)
    states = np.asarray(component.states(coefficients))

    expected = np.zeros_like(states)
    flux_state = np.zeros((basis.Nx,))
    matrix = np.asarray(basis.Gauss_xy.todense())
    for step in range(8):
        t = step / 4.
        distance = (t - basis.ENST) / basis.sigma_T
        weights = np.where(np.abs(distance) < 1.,
                           -np.pi * np.sin(np.pi * distance / 2.)
                           * np.cos(np.pi * distance / 2.) / basis.norm_fact,
                           0.)
        flux_state += .25 * (weights @ np.asarray(coefficients).reshape(-1, basis.Nx))
        if (step + 1) % 4 == 0:
            expected[(step + 1) // 4] = (matrix @ flux_state).reshape(basis.shape_phys)

    np.testing.assert_allclose(states, expected, rtol=2e-6, atol=2e-6)
    gradient = jax.grad(lambda x: jnp.sum(component.states(x) ** 2))(coefficients)
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.max(np.abs(np.asarray(gradient))) > 0.
    assert np.array_equal(cumulative_flux_weights(basis, 2)[0],
                          np.zeros(len(basis.ENST), dtype='float32'))
