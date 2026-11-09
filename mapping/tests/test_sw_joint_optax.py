"""Joint MAP driver must preserve the Gaussian optimum and resumable controls."""
import json

import jax.numpy as jnp
import numpy as np
import pytest

from src import config_default as defaults
from src.exp import Config
from src.sw_joint_optax import joint_optax_map


@pytest.mark.parametrize('grouped', [False, True])
def test_joint_optax_gaussian_map_and_resume(tmp_path, grouped):
    cfg = Config(defaults.INV_4DVAR | dict(
        minimizer='optax-decoupled', jit_cost_and_grad=True,
        cost_and_grad_schedule='scan', maxiter=100, convergence_nit=1,
        gtol=1.e-5, ftol=1.e-8, gradient_max_norm=None, max_retries=0))
    target = jnp.array([.3, -.7, .5])
    error = jnp.array([.5, 1., 2.])
    residual = lambda z: (z-target)/error
    initial = np.zeros(3, dtype='float32')
    groups = np.array([0, 0, 1]) if grouped else None
    result, report = joint_optax_map(residual, initial, groups, tmp_path, 'gaussian', cfg)
    expected = np.asarray(target/(1+error**2))
    np.testing.assert_allclose(result, expected, atol=5.e-5)
    assert report['success']
    assert json.loads((tmp_path/'map_result.json').read_text())['posterior_covariance'] is False
    resumed, info = joint_optax_map(residual, initial, groups, tmp_path, 'gaussian', cfg, resume=True)
    np.testing.assert_allclose(resumed, result, atol=5.e-5)
    with pytest.raises(ValueError, match='provenance'):
        joint_optax_map(residual, initial, groups, tmp_path, 'changed', cfg, resume=True)
