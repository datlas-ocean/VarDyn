"""Reduced local temporal controls: covariance, dimensions and MAP invariants."""
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

from src.sw_reduced_temporal import (ReducedTemporalBasis, local_statistics,
                                    window_matrix, window_acf_table, summarize_local_statistics)
from src.sw_learning_empirical import map_only, MapOptions


def test_local_width_controls_dimension_and_preserves_mean_variance_and_adjoint():
    times=np.arange(41.)
    std=np.array([2.,3.]); mean=np.array([1.,-4.]); width=np.array([4.,16.])
    b=ReducedTemporalBasis(times,std,[3.,10.],mean,window_width=width)
    assert b.counts[0] > b.counts[1]
    assert b.nbasis < 41*2
    np.testing.assert_allclose(b.expand(jnp.asarray(b.control_mean)), np.tile(mean,(41,1)),atol=1e-6)
    np.testing.assert_array_equal(b.expand(jnp.zeros(b.nbasis)),np.zeros((41,2)))
    for i in range(2):
        w=window_matrix(times,width[i])
        expected=np.mean(np.sum(w*w,axis=1))*(std[i]*b.gain[i])**2
        np.testing.assert_allclose(expected,std[i]**2,rtol=1e-6)
        q=np.zeros(b.nbasis,dtype='float32')
        q[b.offsets[i]:b.offsets[i+1]]=np.arange(b.counts[i])
        np.testing.assert_allclose(np.asarray(b.expand(jnp.asarray(q)))[:,i],w@np.arange(b.counts[i]),atol=2e-6)
    rng=np.random.default_rng(1)
    q=jnp.asarray(rng.normal(size=b.nbasis),dtype=jnp.float32)
    y=jnp.asarray(rng.normal(size=(41,2)),dtype=jnp.float32)
    # VJP exercises the scan transpose used by the optimizer. JAX's direct
    # linear_transpose does not support this scan with nonlinear index setup.
    _,transpose=jax.vjp(b.expand,jnp.zeros(b.nbasis))
    np.testing.assert_allclose(jnp.vdot(b.expand(q),y),jnp.vdot(q,transpose(y)[0]),rtol=2e-5)


def test_fitted_acf_is_covariance_of_actual_windows_not_exponential():
    times=np.arange(81.)
    widths,acf,tau=window_acf_table(times,30,candidates=12)
    i=4; w=window_matrix(times,widths[i]); centered=w-w.mean(axis=0)
    cov=centered@centered.T
    expected=np.array([np.diag(cov,k).mean() for k in range(31)])
    expected/=expected[0]
    np.testing.assert_allclose(acf[i],expected,atol=1e-13)
    assert not np.allclose(acf[i],np.exp(-np.arange(31)/tau[i]),atol=.01)


def test_local_estimates_vary_in_space_and_recover_window_width():
    rng=np.random.default_rng(32);times=np.arange(365.)
    values=[]
    for width in (6.,24.):
        w=window_matrix(times,width)
        values.append(w@rng.normal(size=(w.shape[1],40)))
    a=np.concatenate(values,axis=1)
    stats=local_statistics(a,np.ones(80),np.ones(80)*2)
    assert np.median(stats['window_width_days'][:40]) < np.median(stats['window_width_days'][40:])
    np.testing.assert_allclose(np.median(stats['window_width_days'][:40]),6.,rtol=.3)
    np.testing.assert_allclose(np.median(stats['window_width_days'][40:]),24.,rtol=.3)
    assert np.ptp(stats['std']) > .1
    np.testing.assert_allclose(stats['mean'],a.mean(axis=0))
    empty=local_statistics(np.zeros((8,2)),np.ones(2),np.ones(2)*2)
    assert all(empty['fit_status']=='zero_variance_reference_scale')
    np.testing.assert_allclose(empty['std'],1.)
    assert empty['std_prior_retained'].all()
    weak=local_statistics(a,np.ones(80),np.ones(80)*2,floor_fraction=1.e-6)
    np.testing.assert_allclose(weak['std'],weak['raw_std'])
    assert np.all(empty['window_width_days'] < 10.)
    records=np.zeros((2,8));records[:,1]=80.
    summary=summarize_local_statistics(empty,records)
    assert summary[0]['atoms']==2
    assert summary[0]['floor_active_atoms']==2


def test_reduced_map_from_zero_matches_dense_linear_solution(tmp_path):
    times=np.arange(12.)
    b=ReducedTemporalBasis(times,[.6],[3.],[.2],window_width=[5.])
    target=np.sin(times/3.)[:,None]
    w=window_matrix(times,5.)
    sigma=b.control_std; mean=b.control_mean
    exact=np.linalg.solve(w.T@w/.2**2+np.diag(1/sigma**2),
                          w.T@target.ravel()/.2**2+mean/sigma**2)
    def residual(q):
        return (b.expand(q.ravel())-jnp.asarray(target))/.2
    with patch('jax.scipy.sparse.linalg.cg',side_effect=AssertionError('No CG')):
        solution,report=map_only(residual,sigma,None,mean,np.zeros((1,b.nbasis)),tmp_path,
            'reduced-test',MapOptions(maxiter=100,gtol=1e-5),independent=True)
    np.testing.assert_allclose(solution.ravel(),exact,atol=3e-4)
    np.testing.assert_allclose(report['initial_cost'],.5*np.sum((target/.2)**2)+.5*np.sum((mean/sigma)**2),rtol=1e-5)
    np.testing.assert_allclose(report['initial_observation_cost'],.5*np.sum((target/.2)**2),rtol=1e-5)
    assert report['posterior_covariance'] is False
