"""The pilot preserves Gaussian MAP inference and rejects incompatible resumes."""
import pytest
pytest.importorskip("learning_local.sw_learning", reason="Optional local SW learning sources are absent")

from types import SimpleNamespace
import json
import numpy as np
import jax.numpy as jnp
from src.basis import DailyAR1Covariance
from learning_local.sw_learning_pilot import run_pilot
from learning_local.sw_learning_cg_audit import audit


def test_pilot_dense_map_and_restart(tmp_path):
    basis = SimpleNamespace(std=np.array([1., 2.]), tdec=np.array([2., 3.]),
                            ntime=3, natom=2, mean=np.zeros(2), layout_signature='tiny')
    residual = lambda a: 2*a-jnp.ones_like(a)
    rows = run_pilot(residual, basis, tmp_path, maxiter=200, probes=2)
    assert all(row['accepted'] for row in rows)
    covariance = DailyAR1Covariance(basis.std, basis.tdec, basis.ntime)
    L = np.column_stack([np.asarray(covariance.sqrt(jnp.eye(6)[i])) for i in range(6)])
    expected = np.linalg.solve(np.eye(6)+4*L.T@L, 2*L.T@np.ones(6))
    with np.load(tmp_path/'pilot_map.npz') as saved:
        np.testing.assert_allclose(saved['z'], expected, atol=2.e-5)
    pcg = audit(residual, basis, tmp_path, probes=2)
    assert all(row['accepted'] for row in pcg)
    assert json.loads((tmp_path/'pilot_preconditioner_diagonal.json').read_text())['diagonal_quantiles'][0] >= 1.
    before = (tmp_path/'pilot_map_result.json').read_text()
    run_pilot(residual, basis, tmp_path, maxiter=200, probes=2)
    assert (tmp_path/'pilot_map_result.json').read_text() == before
    with pytest.raises(ValueError, match='provenance'):
        run_pilot(residual, basis, tmp_path, provenance='changed')


def test_incomplete_map_is_saved_but_does_not_start_cg(tmp_path):
    basis = SimpleNamespace(std=np.array([1., 2.]), tdec=np.array([2., 3.]),
                            ntime=3, natom=2, mean=np.zeros(2), layout_signature='tiny')
    residual = lambda a: 2*a-jnp.ones_like(a)
    with pytest.raises(RuntimeError, match='MAP incomplete'):
        run_pilot(residual, basis, tmp_path, maxiter=1, probes=2)
    with np.load(tmp_path/'pilot_map.npz') as saved:
        assert not bool(saved['converged'])
        assert np.linalg.norm(saved['z']) > 0
    assert not (tmp_path/'pilot_cg.json').exists()
    run_pilot(residual, basis, tmp_path, maxiter=200, probes=2)
    assert json.loads((tmp_path/'pilot_map_result.json').read_text())['success']


def test_spectral_inverse_matches_dense_precision():
    import jax
    from learning_local.sw_learning_spectral import spectral_inverse
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    precision = (q*np.geomspace(1., 1000., 20))@q.T
    inverse, report = spectral_inverse(jax.jit(lambda x: jnp.asarray(precision)@x), jnp.zeros(20), rank=20)
    matrix = np.column_stack([np.asarray(inverse(jnp.eye(20)[i])) for i in range(20)])
    np.testing.assert_allclose(matrix@precision, np.eye(20), atol=2.e-4)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert report['orthogonality_max'] < 1.e-5


def test_host_spectral_inverse_and_compiled_cg():
    import jax
    from learning_local.sw_learning_host_spectral import host_spectral_inverse
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(20, 20)))
    precision = (q*np.geomspace(1., 1000., 20))@q.T
    hessian = jax.jit(lambda v: jnp.asarray(precision)@v)
    inverse, report = host_spectral_inverse(hessian, jnp.zeros(20), rank=20)
    matrix = np.column_stack([np.asarray(inverse(jnp.eye(20)[i])) for i in range(20)])
    np.testing.assert_allclose(matrix@precision, np.eye(20), atol=2.e-4)
    assert np.linalg.eigvalsh(matrix).min() > 0
    solve = jax.jit(lambda b: jax.scipy.sparse.linalg.cg(hessian, b, M=inverse, tol=1.e-4)[0])
    b = jnp.ones(20)
    x = solve(b)
    assert float(jnp.linalg.norm(hessian(x)-b)/jnp.linalg.norm(b)) < 1.e-4
    assert report['storage'] == 'host'
