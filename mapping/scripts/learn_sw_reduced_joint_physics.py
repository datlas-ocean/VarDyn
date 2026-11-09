#!/usr/bin/env python3
"""Reduced MAP2 with H/g_prime/h_wind still optimized, initialized from MAP1."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
os.environ.setdefault('HDF5_USE_FILE_LOCKING','FALSE')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'mapping'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--statistics',type=Path)
    p.add_argument('--initial-physics-map',type=Path,required=True)
    p.add_argument('--fixed-physics',type=Path)
    p.add_argument('--preconditioner',choices=['prior','component_scale','local_atom'],default='prior')
    p.add_argument('--maxiter',type=int,default=3000)
    p.add_argument('--ftol',type=float,default=5.e-5)
    p.add_argument('--days',type=int)
    p.add_argument('--bounds',type=float,nargs=4)
    p.add_argument('--wavelengths',type=float,nargs='+')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--use-config-optimizer',action='store_true')
    p.add_argument('--restart-from',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--allow-sponge-change',action='store_true',
        help='Allow only sponge_coef to differ from the MAP1 source model')
    args=p.parse_args()
    if args.restart_from and not args.use_config_optimizer:p.error('Continuation requires --use-config-optimizer')
    if not args.statistics or args.fixed_physics or args.restart_from:
        p.error('Joint MAP2 requires statistics and initial physics, without fixed physics or full restart')
    import numpy as np
    import jax
    import jax.numpy as jnp
    from src import exp,state,mod,basis,obs,obsop,config_default as defaults
    from learning_local.sw_learning_empirical import map_only,MapOptions,validate_trajectory
    from learning_local.sw_learning_joint_forward import make_daily_residual
    from learning_local.sw_joint_control import JointControl
    from learning_local.sw_reduced_temporal import ReducedTemporalBasis,local_statistics,summarize_local_statistics
    from learning_local.sw_learning_windowed import atomic_npz
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    print('Device',jax.devices()[0] if args.prepare_only else jax.devices('gpu')[0],flush=True)
    cfg=exp.Exp(str(args.config.resolve()))
    physics_options=runpy.run_path(str(args.config.resolve()))['JOINT_PHYSICS']
    physical_names=set(cfg.MOD.name_params)
    if physical_names not in ({'c','h_wind'},{'H','g_prime','h_wind'},{'H','g_prime','h_wind','bottom_drag_coef'}):
        raise ValueError('Expected c/h_wind or H/g_prime/h_wind controls')
    if physical_names not in ({'H','g_prime','h_wind'}, {'H','g_prime','h_wind','bottom_drag_coef'}):raise ValueError('Expected H/g_prime/h_wind controls')
    if args.days:cfg.EXP.final_date=cfg.EXP.init_date+timedelta(days=args.days)
    if args.bounds:cfg.GRID.update(zip(('lon_min','lon_max','lat_min','lat_max'),args.bounds))
    if args.wavelengths:cfg.BASIS.wavelengths_km=args.wavelengths
    for name,subdir in [('path_save','trajectory'),('path_obs','observations'),('tmp_DA_path','scratch')]:
        cfg.EXP[name]=str(out/subdir);Path(cfg.EXP[name]).mkdir(exist_ok=True)
    for op in cfg.OBSOP.values():op.path_save=str(out/'operators')
    resolved=json.dumps(dict(cfg),default=str,sort_keys=True)
    source=args.initial_physics_map
    old=json.loads((source/'resolved_config.json').read_text())
    for block in ('GRID','MOD','BASIS','OBS'):
        original=old[block]
        if block=='MOD' and args.allow_sponge_change:
            original=dict(original,sponge_coef=json.loads(resolved)[block]['sponge_coef'])
        if original!=json.loads(resolved)[block]:raise ValueError('Physical initialization changed '+block)
    for key in ('init_date','final_date'):
        if old['EXP'][key]!=json.loads(resolved)['EXP'][key]:raise ValueError('Source dates changed')
    source_prep=json.loads((source/'preparation.json').read_text())
    if source_prep['physics_options']!=physics_options:raise ValueError('Physical basis options changed')
    if not json.loads((source/'complete.json').read_text())['diagnostics']['success']:
        raise ValueError('MAP1 must have completed successfully')
    sources=[Path(__file__),args.config,ROOT/'mapping/learning_local/sw_learning_joint_forward.py',
        ROOT/'mapping/learning_local/sw_joint_control.py',ROOT/'mapping/learning_local/sw_learning_empirical.py',
        ROOT/'mapping/src/basis.py',ROOT/'mapping/learning_local/sw_reduced_temporal.py',
        ROOT/'mapping/src/mod.py',ROOT/'mapping/src/basis.py',ROOT/'mapping/models/model_qgsw/sw.py',
        ROOT/'mapping/src/obsop.py']
    inputs={x.path for x in cfg.OBS.values()}|{v['file'] for v in cfg.MOD.bc_files.values()}
    inputs.update(x for x in (cfg.MOD.filec_aux,cfg.MOD.path_wind,cfg.GRID.name_init_mask) if x)
    provenance=dict(sources={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
        inputs={x:dict(size=Path(x).stat().st_size,mtime_ns=Path(x).stat().st_mtime_ns) for x in inputs},
        configuration=json.loads(resolved),physics_options=physics_options,
        source_sponge_coef=old['MOD']['sponge_coef'],
        allow_sponge_change=args.allow_sponge_change,
        preconditioner=args.preconditioner,floor_fraction=1.e-6)
    if args.use_config_optimizer:
        for x in (ROOT/'mapping/learning_local/sw_joint_optax.py',ROOT/'mapping/src/inv.py'):
            provenance['sources'][str(x)]=hashlib.sha256(x.read_bytes()).hexdigest()
        provenance['optimizer_dispatch']='config.INV'
    if args.restart_from:
        old=json.loads((args.restart_from/'resolved_config.json').read_text())
        for block in ('GRID','MOD','BASIS','OBS'):
            if old[block]!=json.loads(resolved)[block]:raise ValueError('Continuation changed '+block)
        for key in ('init_date','final_date'):
            if old['EXP'][key]!=json.loads(resolved)['EXP'][key]:raise ValueError('Continuation dates changed')
        previous=json.loads((args.restart_from/'preparation.json').read_text())
        if previous['physics_options']!=physics_options:raise ValueError('Physical basis changed')
        provenance['restart_from']=str(args.restart_from.resolve())
        provenance['restart_sha256']=hashlib.sha256((args.restart_from/'map_checkpoint.npz').read_bytes()).hexdigest()
    for name,path in [('statistics',args.statistics),('physics_checkpoint',source/'map_checkpoint.npz'),
                      ('physics_fields',source/'physical_parameters.npz')]:
        if path:provenance[name]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    file=out/'provenance.json'
    if file.exists() and json.loads(file.read_text())!=provenance:raise ValueError('Joint MAP provenance mismatch')
    file.write_text(json.dumps(provenance,indent=2));(out/'resolved_config.json').write_text(resolved)
    signature=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    grid=state.State(cfg,verbose=False);model=mod.Model(cfg,grid)
    # Respect an explicit CFL choice; retain the original bound for fixed-dt runs.
    cfl_limit=float(cfg.MOD.cfl) if cfg.MOD.cfl is not None else .05
    speed_ratio=cfg.MOD.c_control_max_ratio if 'c' in physical_names else 1.
    dt_limit=cfl_limit*float(np.min(np.minimum(grid.DX,grid.DY)/(model.c*speed_ratio)))
    if model.dt>dt_limit:raise ValueError(f'dt={model.dt} exceeds controlled-speed CFL envelope {dt_limit}')
    times=np.arange((cfg.EXP.final_date-cfg.EXP.init_date).days+1)*86400.
    dates=np.array([np.datetime64(cfg.EXP.init_date)+np.timedelta64(int(t),'s') for t in times])
    model.set_bc(dates,t_bc=times);model.init(grid)
    daily=basis.Basis(cfg,grid);daily.set_basis(times/86400.)
    reference_std=daily.std.copy();reference_tdec=daily.tdec.copy()
    physics={};fixed={};temporal=None
    if args.statistics:
        with np.load(args.statistics) as data:
            np.testing.assert_array_equal(data['records'],daily.records)
            if str(data['kernel'])!='compact_cosine_squared_partition_unity':raise ValueError('Unexpected temporal kernel')
            daily.update_statistics(data['std'],np.where(np.isfinite(data['tdec']),data['tdec'],reference_tdec),data['mean'])
            temporal=ReducedTemporalBasis(times/86400.,daily.std,daily.tdec,daily.mean,
                facnlt=cfg.BASIS.temporal_facnlt,window_width=data['window_width_days'])
        atomic_npz(out/'temporal_basis.npz',records=daily.records,**temporal.metadata())
    if not args.fixed_physics:
        for name in cfg.MOD.name_params:
            sigma=physics_options['sigma_'+name]
            cc=cfg.copy();cc.BASIS=exp.Config(defaults.BASIS_GAUSS2D|dict(super='BASIS_GAUSS2D',
                name_mod_var=name,sigma_D=physics_options['sigma_D_km'],facns=physics_options['facns'],
                sigma_Q=sigma,compute_velocities=False,use_state_background=False))
            bb=basis.Basis(cc,grid);_,std=bb.set_basis(np.array([0.]),return_q=True)
            physics[name]=(bb,std)
    control=JointControl(daily,temporal,physics,fixed)
    initial=control.initial()
    with np.load(source/'map_checkpoint.npz') as saved:
        physical_z=saved['z'][source_prep['forcing_controls']:].copy()
    if physical_z.size!=control.size-control.nforcing:raise ValueError('Physical coefficient count mismatch')
    initial[control.nforcing:]=physical_z
    initial_fields,initial_params=jax.jit(control.decode)(jnp.asarray(initial))
    zero_tolerance=1.e-6*max(1.,float(np.max(np.abs(daily.mean))))
    if float(jnp.max(jnp.abs(initial_fields)))>zero_tolerance:raise ValueError('MAP2 forcing must start at zero')
    with np.load(source/'physical_parameters.npz') as saved:
        np.testing.assert_array_equal(saved['lon'],grid.lon);np.testing.assert_array_equal(saved['lat'],grid.lat)
        for name,value in initial_params.items():
            np.testing.assert_allclose(np.asarray(value),saved['control_'+name],rtol=1.e-5,atol=1.e-6)
    atomic_npz(out/'initial_physics_controls.npz',**{k:np.asarray(v) for k,v in initial_params.items()})
    atomic_npz(out/'basis_metadata.npz',records=daily.records,std=daily.std,tdec=daily.tdec,mean=daily.mean)
    prep=dict(controls=control.size,forcing_controls=control.nforcing,
        physical_controls={k:v[0].nbasis for k,v in physics.items()},physics_fixed=bool(fixed),
        dt_seconds=float(model.dt),controlled_speed_dt_limit=dt_limit,
        initial_forcing='zero',initial_physics=str(source.resolve()),physical_controls_optimized=True,
        physical_prior_reference='unchanged from MAP1',ntime=daily.ntime,natom=daily.natom,physics_options=physics_options)
    (out/'preparation.json').write_text(json.dumps(prep,indent=2));print('PREPARATION',json.dumps(prep),flush=True)
    if args.prepare_only:return
    observed=obs.Obs(cfg,grid);operator=obsop.Obsop(cfg,grid,observed,model);operator.process_obs()
    residual=make_daily_residual(model,grid,daily,operator,times)
    def objective_residual(z):
        forcing,parameters=control.decode(z)
        return residual(forcing,parameters)
    initial=initial[None]
    if args.use_config_optimizer:
        from learning_local.sw_joint_optax import joint_optax_map
        if args.preconditioner!='component_scale':raise ValueError('Expected component_scale')
        fitted,diagnostics=joint_optax_map(objective_residual,initial,control.groups(args.preconditioner),
            out,signature,cfg.INV,restart_from=args.restart_from,resume=args.resume)
    else:
        fitted,diagnostics=map_only(objective_residual,[1.],None,[0.],initial,out,signature,
            MapOptions(maxiter=args.maxiter,ftol=args.ftol,preconditioner=args.preconditioner),args.resume,
            independent=True,preconditioner_groups=control.groups(args.preconditioner))
    analysis,params=jax.jit(control.decode)(jnp.asarray(fitted.ravel()))
    analysis=np.asarray(analysis);params={k:np.asarray(v) for k,v in params.items()}
    trajectory=jax.jit(make_daily_residual(model,grid,daily,operator,times,return_states=True))
    states={k:np.asarray(v) for k,v in trajectory(jnp.asarray(analysis),params).items()}
    land={k:model._land_mask_for_shape(grid,v.shape[1:]) for k,v in states.items()}
    diagnostics['trajectory_validation']=validate_trajectory(states,land)
    total_H=model._H_control_to_total_state(params.get('c',params.get('H')),
                                            params.get('g_prime'))
    total_g=model._g_prime_control_to_total_state(params.get('g_prime'))
    total_g=np.broadcast_to(total_g,total_H.shape).copy()
    total_c=np.sqrt(total_H*total_g)
    total_hw=model._h_wind_control_to_total_state(params['h_wind'])
    wet=daily.g['wet']
    for name,value in [('c',total_c),('h_wind',total_hw)]:
        if not np.isfinite(value[wet]).all() or np.any(value[wet]<=0):raise FloatingPointError('Invalid learned physical field')
    drag_fields = {}
    if 'bottom_drag_coef' in params:
        rate = model._coefficient_control_to_total_state('bottom_drag_coef', params['bottom_drag_coef'])
        if not np.isfinite(rate[wet]).all() or np.any(rate[wet] <= 0):
            raise FloatingPointError('Invalid learned bottom drag')
        drag_fields = dict(bottom_drag_coef=rate, bottom_drag_timescale_days=1. / (rate * 86400.),
                           reference_bottom_drag_coef=np.asarray(model.bottom_drag_coef_ref))
        diagnostics['bottom_drag_timescale_days_quantiles'] = np.quantile(
            drag_fields['bottom_drag_timescale_days'][wet], [0,.1,.5,.9,1]).tolist()
    atomic_npz(out/'physical_parameters.npz',**drag_fields,c=total_c,H=total_H,g_prime_field=total_g,h_wind=total_hw,
        **{'control_'+k:v for k,v in params.items()},reference_c=model.c,
        reference_h_wind=np.asarray(model.h_wind_ref_sw.T),lon=grid.lon,lat=grid.lat,
        g_prime=cfg.MOD.g_prime,dt_seconds=model.dt)
    local=local_statistics(analysis,reference_std,daily.tdec,floor_fraction=1.e-6,
                           facnlt=cfg.BASIS.temporal_facnlt,min_spacing_days=cfg.BASIS.temporal_min_spacing_days)
    atomic_npz(out/'local_statistics_after_map.npz',records=daily.records,**local)
    (out/'local_statistics_summary.json').write_text(json.dumps(summarize_local_statistics(local,daily.records),indent=2))
    diagnostics.update(controls=control.size,forcing_controls=control.nforcing,physics_fixed=bool(fixed),
        c_quantiles=np.quantile(total_c[wet],[0,.1,.5,.9,1]).tolist(),
        H_quantiles=np.quantile(total_H[wet],[0,.1,.5,.9,1]).tolist(),
        g_prime_quantiles=np.quantile(total_g[wet],[0,.1,.5,.9,1]).tolist(),
        h_wind_quantiles=np.quantile(total_hw[wet],[0,.1,.5,.9,1]).tolist())
    atomic_npz(out/'reconstruction.npz',analysis=analysis,std=daily.std,tdec=daily.tdec,mean=daily.mean,
        records=daily.records,dates_ns=dates.astype('datetime64[ns]').astype('int64'),
        diagnostics=json.dumps(diagnostics),provenance=signature,**{'state_'+k:v for k,v in states.items()})
    (out/'complete.json').write_text(json.dumps(dict(provenance=signature,diagnostics=diagnostics),indent=2))
    print('COMPLETE',json.dumps(diagnostics),flush=True)


if __name__=='__main__':main()
