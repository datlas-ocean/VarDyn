import pytest
pytest.importorskip("learning_local.sw_learning", reason="Optional local SW learning sources are absent")

from datetime import datetime,timedelta
from pathlib import Path
import json
import numpy as np
from learning_local.sw_learning import LearningOptions, maximize_window_statistics
from learning_local.sw_learning_moments import window_moments, combine_moments, common_mean, expected_ar1
from learning_local.sw_learning_windowed import partition_windows, learn_windows, atomic_npz


def test_partition_overlaps_and_unique_core_dates():
    start=datetime(2020,7,1); windows=partition_windows(start,start+timedelta(days=364))
    counts=np.zeros(365,int)
    for i,w in enumerate(windows):
        left=(datetime.fromisoformat(w['start'])-start).days
        counts[left+w['owned_start']:left+w['days']+1]+=1
        if i:
            previous=windows[i-1]
            assert (datetime.fromisoformat(previous['end'])-datetime.fromisoformat(w['start'])).days==10
            assert w['initial_offset_days']==20
    np.testing.assert_array_equal(counts,np.ones(365))
    assert len(windows)==18
    assert windows[-1]['end']==(start+timedelta(days=364)).isoformat()
    with pytest.raises(ValueError): partition_windows(start,start+timedelta(days=10),10,10)


def test_aggregated_likelihood_matches_separate_dense_ar1_blocks():
    rng=np.random.default_rng(21)
    blocks=[]
    for n in (31,20,1):
        a=10000+rng.normal(size=(n,2));p=np.full_like(a,.3);cross=np.full((n-1,2),.1)
        blocks.append((a,p,cross))
    m=combine_moments(window_moments(*b) for b in blocks)
    std=np.array([1.2,2.]);tau=np.array([3.,8.]);rho=np.exp(-1/tau)
    mean=m['reference']+common_mean(m,rho)
    direct=0.
    for a,p,c in blocks:
        d=a-mean; n=len(a)
        q=d[0]**2+p[0]+np.sum((d[1:]-rho*d[:-1])**2+p[1:]+rho**2*p[:-1]-2*rho*c,axis=0)/(1-rho**2)
        direct+=.5*np.sum(2*n*np.log(std)+(n-1)*np.log(1-rho**2)+q/std**2)
    np.testing.assert_allclose(float(expected_ar1(m,np.log(std),np.log(tau))),direct,rtol=2.e-6)
    assert m['transitions']==sum(len(a)-1 for a,_,_ in blocks)


def test_shared_mstep_preserves_unobserved_prior_including_short_window():
    std=np.array([2.]);tau=np.array([6.])
    moments=combine_moments(window_moments(np.full((n,1),10000.),np.full((n,1),4.),
        np.full((n-1,1),4*np.exp(-1/6))) for n in (31,20,1))
    s,t,mean,_=maximize_window_statistics(moments,std,tau,np.array([0]),np.empty((0,2),int),
        np.log([std,tau]),LearningOptions(regional_penalty=0.))
    np.testing.assert_allclose(s,std,rtol=1.e-4)
    np.testing.assert_allclose(t,tau,rtol=1.e-4)
    np.testing.assert_allclose(mean,[10000.],atol=1.e-5)


def test_window_resume_reuses_completed_workers_and_chains_initial_state(tmp_path):
    start=datetime(2020,7,1); windows=partition_windows(start,start+timedelta(days=7),3,1)
    records=np.zeros((2,8));records[:,1]=[100,200]
    stats=dict(std=np.array([1.,2.]),tdec=np.array([3.,6.]),mean=np.array([.1,.2]))
    options=LearningOptions(outer_iterations=1,pooled_iterations=0,probes=2,regional_penalty=0.)
    calls=[]; fail=[True]
    def worker(w,statistics,warm,folder,seed,preceding):
        if w['id']==1 and fail[0]:
            fail[0]=False;raise RuntimeError('simulated interruption')
        calls.append((folder.parent.name,w['id']))
        if w['id']:
            assert Path(preceding).parent.name==f"window_{w['id']-1:03d}"
        else: assert preceding is None
        with np.load(statistics) as saved: prior={k:saved[k] for k in stats}
        n=w['days']+1
        a=np.tile(prior['mean'],(n,1));p=np.tile(prior['std']**2,(n,1))
        c=np.tile(prior['std']**2*np.exp(-1/prior['tdec']),(n-1,1))
        path=folder/'posterior.npz'
        atomic_npz(path,analysis=a,variance=p,lag_covariance=c,records=records,
                   diagnostics=json.dumps({'cg_max_residual':0.}),**prior)
        return path
    with pytest.raises(RuntimeError,match='interruption'):
        learn_windows(windows,records,stats,worker,tmp_path,options)
    result=learn_windows(windows,records,stats,worker,tmp_path,options,resume=True)
    assert calls.count(('iteration_000',0))==1
    for k in stats: np.testing.assert_allclose(result[k],stats[k],rtol=2.e-4)
    ncall=len(calls)
    learn_windows(windows,records,stats,worker,tmp_path,options,resume=True)
    assert len(calls)==ncall
    with pytest.raises(ValueError,match='mismatch'):
        learn_windows(windows,records,stats,worker,tmp_path,options,provenance='changed',resume=True)


@pytest.mark.parametrize('mode',['spectral','spectral-host'])
def test_preconditioner_preserves_same_posterior_draws_and_reuses_map(tmp_path,mode):
    import jax.numpy as jnp
    from unittest.mock import patch
    from learning_local.sw_learning import posterior_step
    y=jnp.arange(6).reshape(6,1)/10
    residual=lambda a:(a-y)/.4
    std=np.array([2.]);tau=np.array([3.]);mean=np.zeros(1);initial=np.zeros((6,1))
    options=LearningOptions(probes=8,cg_rtol=1.e-6,map_gtol=1.e-5)
    baseline=posterior_step(residual,std,tau,mean,initial,options,12,
                            tmp_path/'map.npz','same-objective')
    options.cg_preconditioner=mode;options.cg_spectral_rank=6
    with patch('learning_local.sw_learning.minimize',side_effect=AssertionError('MAP must be reused')):
        conditioned=posterior_step(residual,std,tau,mean,initial,options,12,
                                    tmp_path/'map.npz','same-objective')
    for a,b in zip(baseline[:3],conditioned[:3]):
        np.testing.assert_allclose(a,b,rtol=3.e-5,atol=3.e-6)
    with pytest.raises(ValueError,match='mismatch'):
        posterior_step(residual,std,tau,mean,initial,options,12,tmp_path/'map.npz','other-objective')


def test_cli_prepare_provenance_ignores_json_key_order(tmp_path,monkeypatch):
    import importlib.util
    import sys
    import subprocess
    from types import SimpleNamespace
    from src import exp
    script=Path(__file__).resolve().parents[1]/'scripts/learn_sw_windowed.py'
    spec=importlib.util.spec_from_file_location('window_cli',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(exp,'Exp',lambda path:SimpleNamespace(EXP=SimpleNamespace(
        init_date=datetime(2020,7,1),final_date=datetime(2020,7,5))))
    calls=[]
    def prepare(command,**kwargs):
        folder=Path(command[command.index('--output')+1]);folder.mkdir(exist_ok=True)
        records=np.zeros((1,8));records[:,1]=200
        np.savez(folder/'basis_metadata.npz',records=records,std=np.ones(1),tdec=np.ones(1)*10,mean=np.zeros(1))
        values={'a':1,'b':2} if not calls else {'b':2,'a':1}
        (folder/'provenance.json').write_text(json.dumps({'inputs':values}))
        (folder/'resolved_config.json').write_text('{}')
        calls.append(1)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',prepare)
    monkeypatch.setattr(sys,'argv',[str(script),'--config',str(tmp_path/'config.py'),'--output',str(tmp_path),'--prepare-only'])
    module.main()
    before=(tmp_path/'window_plan.json').read_text()
    module.main()
    assert (tmp_path/'window_plan.json').read_text()==before
