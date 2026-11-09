#!/usr/bin/env python3
"""Assimilate GLORYS and learn daily SW-potential AR1 statistics by approximate EM."""
import argparse
from datetime import timedelta, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'mapping'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-date', type=datetime.fromisoformat)
    parser.add_argument('--e-step', action='store_true', help='One window posterior at prescribed common statistics')
    parser.add_argument('--map-only', action='store_true', help='One continuous MAP and empirical scale statistics; no posterior solves')
    parser.add_argument('--physical-prior', action='store_true', help='Equal physical forcing RMS across scales')
    parser.add_argument('--momentum-rms', type=float, help='Override configured momentum RMS; fallback 0.1 (m/s)/day')
    parser.add_argument('--height-rms', type=float, help='Override configured height RMS; fallback 1 m/day')
    parser.add_argument('--acf-max-lag', type=int, default=30)
    parser.add_argument('--std-floor-fraction', type=float, default=.05)
    parser.add_argument('--map-gtol', type=float, default=1.e-3)
    parser.add_argument('--map-ftol', type=float, default=1.e-9)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--compare-to', type=Path, help='Reference reconstruction for physical-field sensitivity metrics')
    parser.add_argument('--statistics', type=Path)
    parser.add_argument('--reduced-temporal', action='store_true',
                        help='Build compact local temporal wavelets from --statistics; start controls at zero')
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--initial-trajectory', type=Path)
    parser.add_argument('--initial-day', type=int, default=0)
    parser.add_argument('--map-maxcor', type=int, default=20)
    parser.add_argument('--map-preconditioner',choices=['prior','component_scale','local_atom'],default='prior')
    parser.add_argument('--cg-preconditioner', choices=['none', 'spectral', 'spectral-host'], default='none')
    parser.add_argument('--cg-spectral-rank', type=int, default=256)
    parser.add_argument('--prepare-only', action='store_true', help='Validate inputs, physics, basis; do not assimilate')
    parser.add_argument('--pilot-spectral-rank', type=int, default=64)
    parser.add_argument('--pilot-preconditioner', choices=['diagonal', 'spectral', 'spectral-host'], default='diagonal')
    parser.add_argument('--pilot-pcg', action='store_true', help='Audit diagonal PCG at the saved pilot MAP')
    parser.add_argument('--pilot-map-cg', action='store_true', help='Checkpoint a MAP, then audit posterior CG budgets')
    parser.add_argument('--profile', action='store_true', help='Measure forward, gradient and posterior kernels without EM')
    parser.add_argument('--profile-repeats', type=int, default=2)
    parser.add_argument('--profile-max-gib', type=float, default=20.)
    parser.add_argument('--profile-cg-maxiter', type=int, default=30)
    parser.add_argument('--days', type=int, help='Shorten the experiment for a pilot')
    parser.add_argument('--bounds', type=float, nargs=4, metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'), help='Pilot domain')
    parser.add_argument('--wavelengths', type=float, nargs='+', help='Pilot spatial scales in km')
    parser.add_argument('--outer-iterations', type=int, default=8)
    parser.add_argument('--pooled-iterations', type=int, default=2)
    parser.add_argument('--map-maxiter', type=int, default=100)
    parser.add_argument('--cg-maxiter', type=int, default=300, help='Posterior CG cap for EM')
    parser.add_argument('--cg-rtol', type=float, default=1.e-4, help='Posterior CG relative tolerance for EM')
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--std-factor', type=float, default=1., help='Independent initialization multiplier')
    parser.add_argument('--initial-tdec', type=float, help='Override configured initial correlation time in days')
    parser.add_argument('--holdout-days', type=int, nargs='*', default=[], help='Zero-based days excluded from fitting')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.days is not None and args.days < 1:
        parser.error('--days must be at least 1')
    if args.std_factor <= 0 or (args.initial_tdec is not None and args.initial_tdec <= 0):
        parser.error('Initial parameters must be positive')
    if sum((args.map_only, args.e_step, args.pilot_pcg, args.pilot_map_cg, args.profile)) > 1:
        parser.error('Choose only one computation mode')
    if args.reduced_temporal and (not args.map_only or not args.statistics or args.warm_start):
        parser.error('--reduced-temporal requires --map-only --statistics and forbids --warm-start')
    if args.map_preconditioner != 'prior' and not args.reduced_temporal:
        parser.error('Grouped MAP preconditioning requires --reduced-temporal')
    import jax
    import numpy as np
    import xarray as xr
    from src import exp, state, mod, basis, obs, obsop
    from src.sw_learning import learn, LearningOptions
    from src.sw_learning_forward import make_daily_residual
    if not args.prepare_only:
        print('Device:', jax.devices('gpu')[0], flush=True)
    cfg = exp.Exp(str(args.config.resolve()))
    if args.initial_tdec is None:
        args.initial_tdec = cfg.BASIS.initial_tdec_days
    if args.start_date:
        cfg.EXP.init_date = args.start_date
    if args.days:
        cfg.EXP.final_date = cfg.EXP.init_date+timedelta(days=args.days)
    if args.bounds:
        cfg.GRID.update(zip(('lon_min', 'lon_max', 'lat_min', 'lat_max'), args.bounds))
    if args.wavelengths:
        cfg.BASIS.wavelengths_km = args.wavelengths
    args.output.mkdir(parents=True, exist_ok=True)
    for key, subdir in (('path_save', 'trajectory'), ('tmp_DA_path', 'scratch'), ('path_obs', 'observations')):
        cfg.EXP[key] = str(args.output/subdir)
        Path(cfg.EXP[key]).mkdir(exist_ok=True)
    for operator in cfg.OBSOP.values():
        operator.path_save = str(args.output/'operators')
    cfg.BASIS.initial_std = {key: value*args.std_factor for key, value in cfg.BASIS.initial_std.items()}
    cfg.BASIS.initial_tdec_days = args.initial_tdec
    resolved = json.dumps(dict(cfg), default=str, sort_keys=True)
    source_paths = [args.config.resolve(), ROOT/'mapping/src/basis.py',
        ROOT/'mapping/src/sw_learning.py', ROOT/'mapping/src/sw_learning_moments.py',
        ROOT/'mapping/src/sw_learning_spectral.py', ROOT/'mapping/src/sw_learning_host_spectral.py',
        ROOT/'mapping/src/sw_learning_forward.py',
        ROOT/'mapping/src/covariance.py', ROOT/'mapping/src/mod.py',
        ROOT/'mapping/models/model_qgsw/sw.py', ROOT/'mapping/src/obsop.py']
    if args.map_only or args.physical_prior or cfg.BASIS.get('initial_physical_rms'):
        source_paths += [ROOT/'mapping/src/sw_learning_empirical.py', Path(__file__).resolve()]
    if args.reduced_temporal:
        source_paths += [ROOT/'mapping/src/sw_reduced_temporal.py', ROOT/'mapping/src/covariance.py']
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    inputs = set(item.path for item in cfg.OBS.values())
    inputs.update(item['file'] for item in cfg.MOD.bc_files.values())
    inputs.update(p for p in (cfg.MOD.filec_aux, cfg.MOD.path_wind, cfg.GRID.name_init_mask) if p)
    input_metadata = {p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) for p in inputs}
    for auxiliary in (args.statistics, args.warm_start, args.initial_trajectory, args.compare_to):
        if auxiliary is not None:
            input_metadata[str(auxiliary.resolve())] = dict(sha256=hashlib.sha256(auxiliary.read_bytes()).hexdigest())
    provenance = hashlib.sha256((resolved+json.dumps(args.holdout_days)+json.dumps(sources, sort_keys=True)
                                +json.dumps(input_metadata, sort_keys=True)+str(args.initial_day)).encode()).hexdigest()
    if args.map_only or args.physical_prior or cfg.BASIS.get('initial_physical_rms'):
        settings = {key: getattr(args, key) for key in ('map_only', 'physical_prior', 'momentum_rms',
            'height_rms', 'std_factor', 'acf_max_lag', 'std_floor_fraction')}
        if args.reduced_temporal:
            settings['reduced_temporal'] = True
        provenance = hashlib.sha256((provenance+json.dumps(settings, sort_keys=True)).encode()).hexdigest()
    resolved_path = args.output/'resolved_config.json'
    if resolved_path.exists() and resolved_path.read_text() != resolved:
        raise ValueError('Output directory contains a different resolved configuration')
    provenance_path = args.output/'provenance.json'
    provenance_record = dict(sources=sources, inputs=input_metadata)
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance_record:
        raise ValueError('Output directory contains different source code or input files')
    (args.output/'resolved_config.json').write_text(resolved)
    provenance_path.write_text(json.dumps(provenance_record, indent=2))
    grid = state.State(cfg)
    model = mod.Model(cfg, grid)
    if model.name_params:
        raise ValueError('Physical parameters must remain fixed')
    days = (cfg.EXP.final_date-cfg.EXP.init_date).total_seconds()/86400.
    if not days.is_integer():
        raise ValueError('Learning period must span whole days')
    times = np.arange(int(days)+1)*86400.
    dates = np.asarray([np.datetime64(cfg.EXP.init_date+timedelta(seconds=float(t))) for t in times])
    model.set_bc(dates, t_bc=times)
    model.init(grid)
    if args.initial_trajectory:
        with np.load(args.initial_trajectory) as saved:
            expected_date = np.datetime64(cfg.EXP.init_date, 'ns').astype('int64')
            if saved['dates_ns'][args.initial_day] != expected_date:
                raise ValueError('Initial analyzed state date mismatch')
            for name in grid.var:
                value = saved['state_'+name][args.initial_day]
                if value.shape != np.asarray(grid.var[name]).shape:
                    raise ValueError('Initial analyzed state grid mismatch')
                grid.setvar(value, name)
    daily = basis.Basis(cfg, grid)
    daily.set_basis(times/86400.)
    if args.physical_prior or cfg.BASIS.get('initial_physical_rms'):
        from src.sw_learning_empirical import physical_scale_prior
        amplitudes = dict(cfg.BASIS.get('initial_physical_rms') or dict(
            divergent=.1, rotational=.1, height=1., height_mean=1.))
        if args.momentum_rms is not None:
            amplitudes.update(divergent=args.momentum_rms, rotational=args.momentum_rms)
        if args.height_rms is not None:
            amplitudes.update(height=args.height_rms, height_mean=args.height_rms)
        amplitudes = {key: value*args.std_factor for key,value in amplitudes.items()}
        prior, physical_report = physical_scale_prior(daily, tdec=args.initial_tdec, amplitudes=amplitudes)
        daily.update_statistics(**prior)
        (args.output/'physical_prior.json').write_text(json.dumps(physical_report, indent=2))
    if args.statistics:
        with np.load(args.statistics) as saved:
            if not np.array_equal(saved['records'], daily.records):
                raise ValueError('Shared statistics use a different spatial basis')
            if args.reduced_temporal:
                if str(saved['kernel']) != 'compact_cosine_squared_partition_unity':
                    raise ValueError('Reduced basis requires matching window-ACF statistics')
                if int(saved['temporal_facnlt']) != cfg.BASIS.temporal_facnlt:
                    raise ValueError('Window density differs from ACF fit')
                window_width = saved['window_width_days'].copy()
                local_reference_std = saved['std_floor'].copy()/args.std_floor_fraction
            # Effective 1/e time may be unresolved on short pilots; it is not
            # used as an AR1 parameter in reduced mode.
            tau = saved['tdec']
            daily.update_statistics(saved['std'], np.where(np.isfinite(tau),tau,daily.tdec), saved['mean'])
    from src.sw_learning_windowed import atomic_npz
    atomic_npz(args.output/'basis_metadata.npz', records=daily.records, std=daily.std, tdec=daily.tdec, mean=daily.mean)
    temporal = None
    if args.reduced_temporal:
        from src.sw_reduced_temporal import ReducedTemporalBasis
        temporal = ReducedTemporalBasis(times/86400., daily.std, daily.tdec, daily.mean,
            cfg.BASIS.temporal_facnlt, cfg.BASIS.temporal_min_spacing_days, window_width=window_width)
        atomic_npz(args.output/'temporal_basis.npz', records=daily.records, **temporal.metadata())
    h = np.asarray(model.H0).squeeze().T
    report = dict(ntime=daily.ntime, natom=daily.natom, controls=daily.nbasis,
                  grid_shape=[grid.ny, grid.nx], wet_cells=int(daily.g['wet'].sum()),
                  matrix_nnz=int(daily.spatial_matrix.nnz),
                  atoms_per_component={name: int((daily.records[:, 0] == i).sum()) for i, name in enumerate(daily.components)},
                  dt_seconds=float(model.dt), prior=('prescribed statistics' if args.statistics else
                      'configured physical RMS' if args.physical_prior or cfg.BASIS.get('initial_physical_rms') else
                      'generic; no previous learned artifact'),
                  source_sigma_noise={key: item.sigma_noise for key, item in cfg.OBS.items()},
                  holdout_days=args.holdout_days, H_min=float(np.min(h)), H_max=float(np.max(h)))
    report['H_wet_quantiles'] = np.quantile(np.broadcast_to(h, daily.g['wet'].shape)[daily.g['wet']], [0., .01, .5, .99, 1.]).tolist()
    if temporal is not None:
        report.update(controls=temporal.nbasis, daily_coefficients=daily.nbasis,
                      temporal_basis='local compact cosine-squared', initial_forcing='zero',
                      temporal_facnlt=temporal.facnlt,
                      spacing_days_quantiles=np.quantile(temporal.spacing, [0,.1,.5,.9,1]).tolist())
    (args.output/'preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.prepare_only:
        return
    observed = obs.Obs(cfg, grid)
    operator = obsop.Obsop(cfg, grid, observed, model)
    operator.process_obs()
    selected = np.ones(len(times), bool)
    if any(day < 0 or day >= len(times) for day in args.holdout_days):
        raise ValueError('Holdout day outside experiment')
    selected[args.holdout_days] = False
    residual = make_daily_residual(model, grid, daily, operator, times, selected)
    if args.map_only:
        from src.sw_learning_empirical import MapOptions, map_only, estimate_statistics, compare_fields, validate_trajectory
        if args.initial_trajectory:
            raise ValueError('Annual MAP uses one continuous trajectory from its original initial state')
        initial = np.tile(daily.mean, (daily.ntime, 1)).astype('float32')
        if args.warm_start:
            with np.load(args.warm_start) as saved:
                if (not np.array_equal(saved['records'], daily.records)
                        or not np.array_equal(saved['dates_ns'], dates.astype('datetime64[ns]').astype('int64'))):
                    raise ValueError('Annual warm-start dates or basis mismatch')
                initial = saved['analysis'].copy()
        options = MapOptions(args.map_maxiter, args.map_maxcor, args.map_gtol,
                             args.map_ftol, args.checkpoint_every,args.map_preconditioner)
        if temporal is None:
            analysis, diagnostics = map_only(residual, daily.std, daily.tdec, daily.mean, initial,
                                            args.output, provenance, options, args.resume)
        else:
            groups=None
            if args.map_preconditioner=='component_scale':
                from src.sw_learning_empirical import scale_groups
                _,atom_groups=scale_groups(daily.records)
                groups=np.repeat(atom_groups,temporal.counts)
            elif args.map_preconditioner=='local_atom':
                groups=np.repeat(np.arange(daily.natom),temporal.counts)
            reduced, diagnostics = map_only(lambda b: residual(temporal.expand(b.ravel())),
                temporal.control_std, None, temporal.control_mean,
                np.zeros((1, temporal.nbasis), dtype='float32'),
                args.output, provenance, options, args.resume, independent=True,
                preconditioner_groups=groups)
            atomic_npz(args.output/'reduced_control.npz', control=reduced.ravel(), provenance=provenance)
            analysis = np.asarray(jax.jit(temporal.expand)(jax.numpy.asarray(reduced.ravel())))
            diagnostics.update(controls=temporal.nbasis, daily_coefficients=daily.nbasis,
                               initial_forcing='zero', local_temporal_statistics=True)
        if temporal is None:
            estimated, empirical_report = estimate_statistics(analysis, daily.records, daily.std, daily.tdec,
                args.acf_max_lag, args.std_floor_fraction)
        else:
            from src.sw_reduced_temporal import local_statistics, summarize_local_statistics
            local = local_statistics(analysis, local_reference_std, daily.tdec,
                                     args.acf_max_lag, args.std_floor_fraction,
                                     facnlt=cfg.BASIS.temporal_facnlt,
                                     min_spacing_days=cfg.BASIS.temporal_min_spacing_days)
            atomic_npz(args.output/'local_statistics_after_map.npz', records=daily.records, **local)
            estimated = local
            empirical_report = summarize_local_statistics(local, daily.records)
        trajectory = make_daily_residual(model, grid, daily, operator, times, return_states=True)
        states = jax.jit(trajectory)(jax.numpy.asarray(analysis))
        states = {name: np.asarray(value) for name, value in states.items()}
        land_masks = {name: model._land_mask_for_shape(grid, value.shape[1:]) for name, value in states.items()}
        diagnostics['trajectory_validation'] = validate_trajectory(states, land_masks)
        if args.holdout_days:
            validation = make_daily_residual(model, grid, daily, operator, times, ~selected)
            diagnostics['holdout_squared_residual'] = float(jax.numpy.sum(validation(jax.numpy.asarray(analysis))**2))
        if args.compare_to:
            with np.load(args.compare_to) as saved:
                if (not np.array_equal(saved['records'], daily.records)
                        or not np.array_equal(saved['dates_ns'], dates.astype('datetime64[ns]').astype('int64'))):
                    raise ValueError('Comparison dates or basis mismatch')
                comparison = compare_fields(daily, analysis, saved['analysis'])
                comparison['states'] = {}
                for name, value in states.items():
                    ref = saved['state_'+name]
                    land = land_masks[name]
                    wet = np.ones(value.shape[1:], bool) if land is None else ~np.asarray(land, bool)
                    if not np.isfinite(ref[:, wet]).all():
                        raise ValueError('Nonfinite ocean reference trajectory')
                    comparison['states'][name] = dict(rms_difference=float(np.sqrt(np.mean((value[:, wet]-ref[:, wet]).astype(float)**2))),
                        reference_rms=float(np.sqrt(np.mean(ref[:, wet].astype(float)**2))))
                comparison['reference_diagnostics'] = json.loads(str(saved['diagnostics']))
            (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2))
        atomic_npz(args.output/'reconstruction.npz', analysis=analysis, std=daily.std, tdec=daily.tdec,
            mean=daily.mean, records=daily.records, dates_ns=dates.astype('datetime64[ns]').astype('int64'),
            diagnostics=json.dumps(diagnostics), provenance=provenance,
            **{'state_'+name: value for name, value in states.items()})
        atomic_npz(args.output/'empirical_statistics.npz', **estimated, records=daily.records,
                   report=json.dumps(empirical_report), provenance=provenance)
        (args.output/'empirical_statistics.json').write_text(json.dumps(empirical_report, indent=2))
        (args.output/'complete.json').write_text(json.dumps(dict(provenance=provenance, diagnostics=diagnostics), indent=2))
        print(json.dumps(diagnostics), flush=True)
        return
    if args.e_step:
        from src.sw_learning import posterior_step
        options = LearningOptions(map_maxiter=args.map_maxiter, map_maxcor=args.map_maxcor,
            probes=args.probes, seed=args.seed, cg_maxiter=args.cg_maxiter, cg_rtol=args.cg_rtol,
            cg_preconditioner=args.cg_preconditioner, cg_spectral_rank=args.cg_spectral_rank)
        options.validate()
        initial = np.tile(daily.mean, (daily.ntime, 1))
        if args.warm_start:
            with np.load(args.warm_start) as saved:
                if not np.array_equal(saved['dates_ns'],dates.astype('datetime64[ns]').astype('int64')):
                    raise ValueError('Warm-start dates mismatch')
                if not np.array_equal(saved['records'],daily.records):
                    raise ValueError('Warm-start basis mismatch')
                initial = saved['analysis'].copy()
        from dataclasses import asdict
        map_signature = provenance
        analysis, variance, lag, diagnostics = posterior_step(residual,daily.std,daily.tdec,daily.mean,
            initial,options,args.seed,args.output/'map_latest.npz',map_signature)
        trajectory = make_daily_residual(model,grid,daily,operator,times,return_states=True)
        states = jax.jit(trajectory)(jax.numpy.asarray(analysis))
        atomic_npz(args.output/'posterior.npz', analysis=analysis,variance=variance,lag_covariance=lag,
            std=daily.std,tdec=daily.tdec,mean=daily.mean,records=daily.records,
            dates_ns=dates.astype('datetime64[ns]').astype('int64'),
            diagnostics=json.dumps(diagnostics),provenance=provenance,
            **{'state_'+name: np.asarray(value) for name,value in states.items()})
        print(json.dumps(diagnostics),flush=True)
        return
    if args.pilot_pcg:
        from src.sw_learning_cg_audit import audit
        audit(residual, daily, args.output, args.probes, args.seed, provenance, args.pilot_preconditioner, args.pilot_spectral_rank)
        return
    if args.pilot_map_cg:
        from src.sw_learning_pilot import run_pilot
        run_pilot(residual, daily, args.output, args.map_maxiter, args.probes, args.seed, provenance)
        return
    if args.profile:
        from src.sw_learning_profile import profile_problem
        profile_problem(residual, daily, args.output, args.profile_repeats,
                        args.profile_max_gib, args.profile_cg_maxiter)
        return
    validation = make_daily_residual(model, grid, daily, operator, times, ~selected) if args.holdout_days else None
    options = LearningOptions(outer_iterations=args.outer_iterations, pooled_iterations=args.pooled_iterations,
        map_maxiter=args.map_maxiter, probes=args.probes, seed=args.seed,
        cg_maxiter=args.cg_maxiter, cg_rtol=args.cg_rtol, map_maxcor=args.map_maxcor,
        cg_preconditioner=args.cg_preconditioner,cg_spectral_rank=args.cg_spectral_rank)
    result = learn(residual, daily, args.output, options, validation, args.resume, provenance)
    data = {key: (('time', 'atom') if result[key].ndim == 2 else ('atom',), result[key])
            for key in ('analysis', 'variance', 'std', 'tdec', 'mean')}
    for index, name in enumerate(('component', 'wavelength_km', 'radius_km', 'center_x', 'center_y', 'basin', 'support', 'direct_support')):
        data[name] = (('atom',), daily.records[:, index])
    xr.Dataset(data, coords=dict(time=dates, atom=np.arange(daily.natom)), attrs=dict(
        schema='sw-daily-learning-v1', layout_signature=daily.layout_signature,
        potential_units='m2/s/day', height_units='m interface/day', tdec_units='day',
        std_definition='stationary coefficient standard deviation at daily nodes')).to_netcdf(args.output/'learning.nc')


if __name__ == '__main__':
    main()
