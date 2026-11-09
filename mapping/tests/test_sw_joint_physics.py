"""Joint static physical fields, canonical control packing and derivatives."""
import pytest
pytest.importorskip("learning_local.sw_learning", reason="Optional local SW learning sources are absent")

from datetime import datetime
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from learning_local.sw_joint_control import JointControl
from learning_local.sw_reduced_temporal import ReducedTemporalBasis




def test_joint_packing_and_fixed_map2():
    records=np.zeros((2,8));records[:,1]=[100,200]
    daily=SimpleNamespace(ntime=5,natom=2,nbasis=10,std=np.array([2.,3.]),
        tdec=np.array([2.,4.]),mean=np.array([.2,-.3]),records=records)
    class Spatial:
        nbasis=2
        def operg(self,t,q):return q.reshape(1,2)
    physics={'c':(Spatial(),np.ones(2)*.15),'h_wind':(Spatial(),np.ones(2)*.3)}
    joint=JointControl(daily,physics=physics)
    a,p=joint.decode(jnp.asarray(joint.initial()))
    np.testing.assert_allclose(a,0,atol=1e-7)
    np.testing.assert_array_equal(p['c'],0)
    z=joint.initial();z[-4:]=[1.,2.,3.,4.]
    a,p=joint.decode(jnp.asarray(z))
    np.testing.assert_allclose(p['c'],[[.15,.3]])
    temporal=ReducedTemporalBasis(np.arange(5.),daily.std,daily.tdec,daily.mean,window_width=[2.,4.])
    reduced=JointControl(daily,temporal=temporal,fixed=p)
    b,fixed=reduced.decode(jnp.asarray(reduced.initial()))
    np.testing.assert_allclose(b,0,atol=2e-7)
    for name in p:np.testing.assert_array_equal(fixed[name],p[name])
    for strategy in ('component_scale','local_atom'):
        assert joint.groups(strategy).shape==(joint.size,)
        assert reduced.groups(strategy).shape==(reduced.size,)
    assert joint.size==14 and reduced.size==temporal.nbasis
    # A reduced MAP may retain active physical coefficients initialized from
    # MAP1, independently of the changed forcing vector length.
    active=JointControl(daily,temporal=temporal,physics=physics)
    start=active.initial();start[active.nforcing:]=z[joint.nforcing:]
    forcing,transferred=active.decode(jnp.asarray(start))
    np.testing.assert_allclose(forcing,0,atol=2e-7)
    for name in p:np.testing.assert_allclose(transferred[name],p[name])
    grad=jax.grad(lambda zz:sum(jnp.sum(v*v) for v in active.decode(zz)[1].values()))(jnp.asarray(start))
    assert np.all(np.asarray(grad[active.nforcing:])!=0)


@pytest.mark.parametrize('names',[('c','h_wind'),('H','g_prime','h_wind'),('H','g_prime','h_wind','bottom_drag_coef')])
def test_joint_forward_physics_gradients_and_fixed_transfer(tmp_path,names):
    from src import config_default as defaults
    from src.exp import Config
    from src.state import State
    from src.mod import Model_qgsw
    from src.basis import Basis
    from learning_local.sw_learning_joint_forward import make_daily_residual
    cfg=Config(EXP=Config(defaults.EXP|dict(path_save=str(tmp_path),tmp_DA_path=str(tmp_path),
        flag_plot=0,init_date=datetime(2024,1,1),final_date=datetime(2024,1,3))),
        GRID=Config(defaults.GRID_GEO|dict(super='GRID_GEO',lon_min=-2.,lon_max=2.,lat_min=-2.,lat_max=2.,
            dlon=.5,dlat=.5,name_init_mask=None)),
        MOD=Config(defaults.MOD_QGSW|dict(super='MOD_QGSW',H=250.,g_prime=.025,dtmodel=1800.,cfl=None,
            name_params=list(names),h_wind=80.,bottom_drag_coef=1./(20*86400.),height_representation='interface_displacement',
            dist_sponge_bc=0.,h_adv_scheme='linear_upwind5',mom_adv_scheme='upwind5')),
        BASIS=Config(defaults.BASIS_SW_DAILY|dict(super='BASIS_SW_DAILY',wavelengths_km=[300.])))
    state=State(cfg,verbose=False);model=Model_qgsw(cfg,state);model.init(state)
    if 'bottom_drag_coef' in names:
        np.testing.assert_allclose(model._coefficient_control_to_total_state(
            'bottom_drag_coef', state.params['bottom_drag_coef']), 1./(20*86400.), rtol=1e-6)
        assert abs(model.adjoint_test_jstep(nstep=1)-1.) < 2.e-5
    shape=(state.ny,state.nx)
    state.setvar(np.sin(np.arange(state.nx))[None,:]*np.ones(shape)*.2,'ssh')
    model.taux_wind=np.ones((1,state.ny,state.nx+1))*.03
    model.tauy_wind=np.ones((1,state.ny+1,state.nx))*.015
    model.wind_taper_u=model.wind_taper_v=1.;model.wind_dt=86400.
    b=Basis(cfg,state,verbose=False);b.set_basis([0.,2.])
    class Obs:
        def scan_misfit_size(self):return 1
        def scan_misfit(self,t,fields):return jnp.concatenate([fields[k].ravel() for k in ('ssh','u','v')])
    times=[0.,86400.,172800.]
    residual=make_daily_residual(model,state,b,Obs(),times)
    forcing=jnp.zeros((3,b.natom))
    def parameters(x):return {name:jnp.ones(shape)*x[i] for i,name in enumerate(names)}
    loss=jax.jit(lambda x:jnp.sum(residual(forcing,parameters(x))**2))
    x=jnp.array([.05,-.03,.02,.10][:len(names)]);gradient=np.asarray(jax.jit(jax.grad(loss))(x))
    assert np.isfinite(gradient).all() and np.all(abs(gradient)>1e-6)
    for i in range(len(names)):
        delta=jnp.eye(len(names))[i]*.005
        fd=float((loss(x+delta)-loss(x-delta))/.01)
        np.testing.assert_allclose(gradient[i],fd,rtol=.04,atol=1e-5)
    # A transferred fixed field must produce the same trajectory as a learned
    # field at the same values, without any physical-parameter controls in q.
    state.params.update(parameters(x))
    fixed=make_daily_residual(model,state,b,Obs(),times)
    np.testing.assert_allclose(jax.jit(fixed)(forcing,{}),
                               jax.jit(residual)(forcing,parameters(x)),rtol=1e-6,atol=1e-7)


def test_spatial_bottom_drag_faces_and_energy():
    from models.model_qgsw.sw import SW
    from types import SimpleNamespace
    rate=jnp.arange(12,dtype=jnp.float32).reshape(4,3)*1.e-6+1.e-6
    core=SimpleNamespace(bottom_drag_coef=2.e-6)
    u=jnp.ones((1,1,5,3))*2.;v=jnp.ones((1,1,4,4))*3.
    du=jnp.zeros((1,1,3,3));dv=jnp.zeros((1,1,4,2))
    ru,rv=SW.add_bottom_drag(core,du,dv,u,v,rate)
    np.testing.assert_allclose(ru[0,0],-(rate[:-1]+rate[1:]),rtol=1e-6)
    np.testing.assert_allclose(rv[0,0],-1.5*(rate[:,:-1]+rate[:,1:]),rtol=1e-6)
    assert float(jnp.sum(ru*u[...,1:-1,:])+jnp.sum(rv*v[...,:,1:-1])) < 0.
