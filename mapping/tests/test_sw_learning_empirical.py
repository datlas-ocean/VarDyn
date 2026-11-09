"""Scientific invariants for continuous MAP-only empirical calibration."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from src.sw_learning_empirical import (MapOptions, compare_fields, estimate_statistics,
                                      fit_decay, map_only, physical_scale_prior, validate_trajectory)


def test_physical_scale_normalization_includes_overlap_and_derivatives():
    # Two coincident weak atoms at one scale, one strong atom at another.
    # Equal coefficient amplitudes would give very different physical energy.
    matrix = sparse.csr_matrix([[1., 1., 10., 0.], [2., 2., 20., 0.], [0., 0., 0., 3.]])
    records = np.zeros((4, 8)); records[:, 0] = [0, 0, 0, 2]; records[:, 1] = [80, 80, 160, 80]
    g = {name: np.ones((1, 1)) for name in ('au', 'av', 'wu', 'wv', 'area', 'wet', 'basins')}
    b = SimpleNamespace(g=g, records=records, natom=4, spatial_matrix=matrix, offsets=np.arange(4))
    stats, report = physical_scale_prior(b, .1, 1., 2.)
    for ids, expected in (([0, 1], .1), ([2], .1), ([3], 1.)):
        fields_variance = np.asarray(matrix[:, ids].power(2) @ stats['std'][ids]**2).ravel()
        rms = np.sqrt(fields_variance[:2].mean()) if ids != [3] else np.sqrt(fields_variance[2])
        np.testing.assert_allclose(rms, expected)
    np.testing.assert_allclose(stats['std'][0]/stats['std'][2], 10/np.sqrt(2))
    assert all(s['physical_rms'] > 0 for s in report)
    custom,_=physical_scale_prior(b,amplitudes=dict(divergent=.02,rotational=.03,height=.4,height_mean=.5))
    np.testing.assert_allclose(custom['std'][:3]/stats['std'][:3],.2)
    np.testing.assert_allclose(custom['std'][3]/stats['std'][3],.4)
    a = np.ones((5, 4), dtype='float32')
    comparison = compare_fields(b, 2*a, a)
    for row in comparison.values():
        np.testing.assert_allclose(row['relative_difference'], 1.)


def test_multilag_exponential_and_unresolved_fit():
    lags = np.arange(31)
    tau, report = fit_decay(np.exp(-lags/7.), 365-lags)
    assert tau == pytest.approx(7., rel=1.e-5)
    assert len(report['used_lags']) == 30
    tau, report = fit_decay([1., -.1, -.05], [100, 99, 98])
    assert report['status'] == 'unresolved_short_decay'


def test_statistics_remove_spatial_means_and_recover_ar1():
    rng = np.random.default_rng(2026)
    ntime, natom, tau, sigma = 4000, 30, 5., 2.
    rho = np.exp(-1/tau)
    a = np.empty((ntime, natom)); a[0] = rng.normal(size=natom)*sigma
    noise = rng.normal(size=a.shape)*sigma*np.sqrt(1-rho*rho)
    for i in range(1, ntime):
        a[i] = rho*a[i-1]+noise[i]
    a += np.arange(natom)*1000.
    records = np.zeros((natom, 8)); records[:, 1] = 80.
    stats, report = estimate_statistics(a, records, np.ones(natom)*3, np.ones(natom)*2)
    np.testing.assert_allclose(stats['std'], sigma, rtol=.03)
    np.testing.assert_allclose(stats['tdec'], tau, rtol=.12)
    np.testing.assert_allclose(stats['mean'], a.mean(0))
    assert not report[0]['floor_active']
    empty, flags = estimate_statistics(np.ones((365, natom))*10, records, 3., 2.)
    assert flags[0]['status'] == 'zero_variance_prior_retained'
    np.testing.assert_allclose(empty['std'], .15)


def test_map_only_matches_dense_solution_and_never_runs_cg(tmp_path):
    n, sigma, tau, error = 7, 1.3, 2., .4
    y = np.linspace(-.4, .7, n)
    residual = lambda a: (a.ravel()-jnp.asarray(y))/error
    q = sigma**2*np.exp(-abs(np.arange(n)[:, None]-np.arange(n))/tau)
    exact = np.linalg.solve(np.linalg.inv(q)+np.eye(n)/error**2, y/error**2)
    options = MapOptions(maxiter=100, gtol=1.e-5, checkpoint_every=1)
    with patch('jax.scipy.sparse.linalg.cg', side_effect=AssertionError('No CG allowed')):
        a, report = map_only(residual, [sigma], [tau], [0.], np.zeros((n, 1)), tmp_path, 'test', options)
    np.testing.assert_allclose(a.ravel(), exact, atol=5.e-5)
    assert report['posterior_covariance'] is False
    with patch('src.sw_learning_empirical.minimize', side_effect=AssertionError('Reuse successful MAP')):
        resumed, _ = map_only(residual, [sigma], [tau], [0.], np.zeros((n, 1)), tmp_path, 'test', options, True)
    np.testing.assert_array_equal(a, resumed)
    with pytest.raises(ValueError, match='signature'):
        map_only(residual, [sigma], [tau], [0.], np.zeros((n, 1)), tmp_path, 'other', options, True)


def test_incomplete_map_checkpoint_is_not_success(tmp_path):
    residual = lambda a: (a.ravel()-jnp.arange(8))/jnp.linspace(.1, 2., 8)
    with pytest.raises(RuntimeError, match='incomplete'):
        map_only(residual, [1.], [2.], [0.], np.zeros((8, 1)), tmp_path, 'test',
                 MapOptions(maxiter=1, checkpoint_every=1))
    with np.load(tmp_path/'map_checkpoint.npz') as saved:
        assert not bool(saved['success'])
        assert int(saved['iterations']) == 1
        assert json.loads(str(saved['diagnostics']))['posterior_covariance'] is False


@pytest.mark.parametrize('gradient_only', [False, True])
def test_map_rejects_invalid_line_search_and_recovers(tmp_path, gradient_only):
    import jax
    @jax.custom_jvp
    def limited(x):
        return 10*(x-.4) if gradient_only else jnp.where(x < .6, 10*(x-.4), jnp.nan)

    @limited.defjvp
    def limited_jvp(primals, tangents):
        x, = primals
        dx, = tangents
        return limited(x), jnp.where(x < .6, 10., jnp.nan)*dx

    result, report = map_only(lambda a: limited(a.ravel()), [1.], None, [0.],
        np.zeros((1, 1)), tmp_path, 'invalid-trial',
        MapOptions(maxiter=50, gtol=1.e-4, ftol=1.e-8, checkpoint_every=1), independent=True)
    np.testing.assert_allclose(result.ravel(), [40/101], atol=1.e-5)
    assert report['rejected_trials'] >= 1
    trials = list((tmp_path/'rejected_trials').glob('*.npz'))
    with np.load(trials[0]) as saved:
        assert saved['z'][0] >= .6
        np.testing.assert_array_equal(saved['last_accepted'], [0.])
        failure = json.loads(str(saved['diagnostics']))
        assert failure['cost_finite'] == gradient_only
        assert failure['gradient_nonfinite'] > 0


def test_map_invalid_initial_point_fails_with_diagnostic(tmp_path):
    with pytest.raises(FloatingPointError, match='initial MAP'):
        map_only(lambda a: a*jnp.nan, [1.], None, [0.], np.zeros((1, 1)),
                 tmp_path, 'invalid-start', independent=True)
    assert len(list((tmp_path/'rejected_trials').glob('*.npz'))) == 1


@pytest.mark.parametrize('strategy', ['prior','component_scale','local_atom'])
def test_preconditioners_preserve_physical_map_and_initial_prior(tmp_path,strategy):
    from src.sw_learning_empirical import gradient_group_scaling
    matrix=np.array([[20.,2.,0.],[0.,1.,.3],[1.,0.,.1]])
    target=np.array([2.,-.5,.4]);std=np.array([.4,1.2,2.]);mean=np.array([.3,-.7,.2])
    precision=np.diag(1/std**2)
    exact=np.linalg.solve(matrix.T@matrix+precision,matrix.T@target+precision@mean)
    groups=np.array([0,0,1]) if strategy=='component_scale' else np.arange(3)
    options=MapOptions(maxiter=200,gtol=1.e-4,ftol=1.e-10,checkpoint_every=1,preconditioner=strategy)
    result,report=map_only(lambda a:jnp.asarray(matrix)@a.ravel()-target,std,None,mean,
        np.zeros((1,3)),tmp_path,'preconditioner-invariance',options,independent=True,
        preconditioner_groups=groups)
    np.testing.assert_allclose(result.ravel(),exact,atol=3.e-4)
    np.testing.assert_allclose(report['initial_prior_cost'],.5*np.sum((mean/std)**2),rtol=1.e-6)
    with np.load(tmp_path/'map_checkpoint.npz') as saved:
        np.testing.assert_allclose(mean+std*saved['z'],result.ravel(),atol=1.e-6)
    if strategy!='prior':
        with np.load(tmp_path/'map_preconditioner.npz') as saved:
            scale=saved['scale'];assert np.all((scale>=.1)&(scale<=10))
    with patch('src.sw_learning_empirical.minimize',side_effect=AssertionError('Reuse result')):
        resumed,_=map_only(lambda a:a,std,None,mean,np.zeros((1,3)),tmp_path,
            'preconditioner-invariance',options,True,independent=True,preconditioner_groups=groups)
    np.testing.assert_array_equal(resumed,result)
    scale,_=gradient_group_scaling(np.zeros(3),groups)
    assert np.isfinite(scale).all()


def test_trajectory_validation_distinguishes_land_from_ocean():
    field = np.ones((3, 2, 2))
    land = np.array([[True, False], [False, False]])
    field[:, land] = np.nan
    report = validate_trajectory({'ssh':field}, {'ssh':land})
    assert report['ssh'] == dict(nonfinite_wet=0, nonfinite_land=3)
    field[2, 1, 1] = np.inf
    with pytest.raises(FloatingPointError, match='day index 2'):
        validate_trajectory({'ssh':field}, {'ssh':land})


def test_daily_basis_factory_honors_physical_amplitudes():
    from copy import deepcopy
    from datetime import datetime
    from src import config_default as defaults
    from src.exp import Config
    from src.basis import Basis
    x,y=np.meshgrid(np.arange(8)*25000.,np.arange(7)*25000.)
    state=SimpleNamespace(nx=8,ny=7,X=x,Y=y,DX=x*0+25000,DY=y*0+25000,mask=None)
    targets=dict(divergent=.02,rotational=.03,height=.5,height_mean=.1)
    cfg=Config(BASIS=Config(deepcopy(defaults.BASIS_SW_DAILY)|dict(super='BASIS_SW_DAILY',
        wavelengths_km=[200.],initial_physical_rms=targets,initial_tdec_days=2.)),
        EXP=Config(init_date=datetime(2020,1,1)))
    b=Basis(cfg,state,verbose=False)
    weights=np.concatenate(((b.g['au']*b.g['wu']).ravel(),(b.g['av']*b.g['wv']).ravel(),
                             (b.g['area']*b.g['wet']).ravel()))
    for i,name in enumerate(b.components):
        ids=b.records[:,0]==i
        variance=np.asarray(b.spatial_matrix[:,ids].power(2)@b.std[ids]**2).ravel()
        section=slice(0,b.offsets[2]) if i<2 else slice(b.offsets[2],None)
        np.testing.assert_allclose(np.sqrt(weights[section]@variance[section]/weights[section].sum()),targets[name],rtol=1.e-6)
    np.testing.assert_array_equal(b.tdec,np.full(b.natom,2.))
