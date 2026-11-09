"""Single-trajectory MAP calibration, without posterior covariance sampling.

Statistics describe reconstructed coefficients, not an uncertainty-corrected EM
estimate. Physical scale amplitudes include atom overlap on the actual wet grid.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .basis import DailyAR1Covariance
from .sw_learning_windowed import atomic_npz


def scale_groups(records):
    """Pool locations by component/wavelength; basin means remain separate."""
    keys = np.array([(r[0], r[1], r[5] if r[0] == 3 else 0) for r in records])
    return np.unique(keys, axis=0, return_inverse=True)


def physical_scale_prior(basis, momentum_rms=.1, height_rms=1., tdec=2., amplitudes=None):
    """Equal expected area-weighted field RMS per component and wavelength.

    For momentum RMS is over both wet face grids (not the vector magnitude).
    For height it is over wet cells; a basin-mean atom uses its own basin area.
    Independent coefficients imply E[||A a||_W²] = sum_j std_j² ||A_j||_W².
    A single coefficient std per group accounts for derivatives AND overlap.
    """
    if not all(np.isfinite(x) and x > 0 for x in (momentum_rms, height_rms, tdec)):
        raise ValueError('Physical amplitudes and tdec must be finite and positive')
    targets = dict(divergent=momentum_rms, rotational=momentum_rms, height=height_rms, height_mean=height_rms)
    if amplitudes is not None:
        if set(amplitudes) != set(targets):
            raise ValueError('Physical prior must specify all four forcing components')
        targets.update(amplitudes)
    if not all(np.isfinite(x) and x > 0 for x in targets.values()):
        raise ValueError('Physical component amplitudes must be finite and positive')
    g = basis.g
    weights = np.concatenate(((g['au']*g['wu']).ravel(),
                              (g['av']*g['wv']).ravel(), (g['area']*g['wet']).ravel()))
    matrix = basis.spatial_matrix.astype(np.float64)
    energy = np.asarray(matrix.power(2).T @ weights).ravel()
    keys, groups = scale_groups(basis.records)
    std = np.empty(basis.natom)
    report = []
    momentum_area = float(np.sum(g['au']*g['wu'])+np.sum(g['av']*g['wv']))
    for index, (component, wavelength, basin) in enumerate(keys):
        ids = groups == index
        area = momentum_area if component < 2 else float(np.sum(g['area']*(
            (g['basins'] == basin) if component == 3 else g['wet'])))
        response = np.sqrt(energy[ids].sum()/area) if area > 0 else 0.
        if not np.isfinite(response) or response <= 0:
            raise ValueError('Empty physical response in a forcing scale')
        target = targets[('divergent','rotational','height','height_mean')[int(component)]]
        std[ids] = target/response
        report.append(dict(component=int(component), wavelength_km=float(wavelength),
                           basin=int(basin), atoms=int(ids.sum()), coefficient_std=float(std[ids][0]),
                           physical_rms=float(target), response=float(response)))
    return dict(std=std, tdec=np.full(basis.natom, tdec), mean=np.zeros(basis.natom)), report


def fit_decay(acf, pairs, minimum=.25, maximum=120.):
    """Weighted exponential fit over consecutive positive lags, without modes."""
    acf, pairs = np.asarray(acf, float), np.asarray(pairs, float)
    if (acf.ndim != 1 or pairs.shape != acf.shape or len(acf) < 3
            or not np.isfinite(acf).all() or not np.isfinite(pairs).all()
            or np.any(pairs <= 0) or not 0 < minimum < maximum):
        raise ValueError('Invalid ACF fit inputs')
    end = next((i for i in range(1, len(acf)) if acf[i] <= 0), len(acf))
    lags = np.arange(1, end)
    if len(lags) < 2:
        return minimum, dict(status='unresolved_short_decay', used_lags=lags.tolist(), rmse=None)
    w = pairs[lags]/pairs[lags].sum()
    objective = lambda log_tau: float(np.sum(w*(np.exp(-lags/np.exp(log_tau))-acf[lags])**2))
    fit = minimize_scalar(objective, bounds=(np.log(minimum), np.log(maximum)), method='bounded')
    tau = float(np.exp(fit.x))
    status = 'bound' if tau < minimum*1.01 or tau > maximum/1.01 else 'fitted'
    return tau, dict(status=status, used_lags=lags.tolist(), rmse=float(np.sqrt(fit.fun)))


def estimate_statistics(analysis, records, reference_std, reference_tdec, max_lag=30,
                        std_floor_fraction=.05, tdec_bounds=(.25, 120.)):
    """Temporal mean per atom; pooled temporal variance/ACF per scale.

    Per-atom centering avoids mistaking fixed spatial mean differences for long
    temporal persistence. The group mean and raw variance are also reported.
    A documented floor protects poorly observed groups from MAP-only collapse.
    """
    a = np.asarray(analysis)
    if (a.ndim != 2 or a.shape[1] != len(records) or a.shape[0] < 4
            or not np.isfinite(a).all() or max_lag < 2 or not 0 < std_floor_fraction <= 1):
        raise ValueError('Need finite coefficients, >=4 dates, and >=2 ACF lags')
    ntime, natom = a.shape
    reference_std = np.broadcast_to(reference_std, (natom,))
    reference_tdec = np.broadcast_to(reference_tdec, (natom,))
    if (not np.isfinite(reference_std).all() or not np.isfinite(reference_tdec).all()
            or np.any(reference_std <= 0) or np.any(reference_tdec <= 0)):
        raise ValueError('Invalid reference prior')
    keys, groups = scale_groups(records)
    mean = a.mean(axis=0, dtype=np.float64)
    std, tdec = np.empty(natom), np.empty(natom)
    report = []
    nlags = min(max_lag, ntime//3)
    if nlags < 2:
        nlags = 2
    for group, (component, wavelength, basin) in enumerate(keys):
        ids = groups == group
        centered = a[:, ids].astype(np.float64)-mean[ids]
        covariance = np.array([np.mean(centered[:ntime-k]*centered[k:]) for k in range(nlags+1)])
        raw_std = float(np.sqrt(max(covariance[0], 0.)))
        floor = float(std_floor_fraction*np.sqrt(np.mean(reference_std[ids]**2)))
        pooled_std = max(raw_std, floor)
        if raw_std <= np.finfo(float).eps*max(1., float(np.max(abs(mean[ids])))):
            acf = np.zeros(nlags+1); acf[0] = 1.
            tau = float(np.median(reference_tdec[ids]))
            fit = dict(status='zero_variance_prior_retained', used_lags=[], rmse=None)
        else:
            acf = covariance/covariance[0]
            tau, fit = fit_decay(acf, ntime-np.arange(nlags+1), *tdec_bounds)
        std[ids], tdec[ids] = pooled_std, tau
        report.append(dict(component=int(component), wavelength_km=float(wavelength), basin=int(basin),
            atoms=int(ids.sum()), mean=float(np.mean(mean[ids])), raw_std=raw_std, std=pooled_std,
            std_floor=floor, floor_active=raw_std < floor, tdec_days=tau,
            acf=acf.tolist(), lags_days=list(range(nlags+1)), **fit))
    return dict(std=std, tdec=tdec, mean=mean), report


def compare_fields(basis, analysis, reference):
    """Physical differences, including interference of overlapping atoms."""
    if analysis.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError('Invalid reference reconstruction')
    g = basis.g
    weights = ((g['au']*g['wu']).ravel(), (g['av']*g['wv']).ravel(),
               (g['area']*g['wet']).ravel())
    energies = np.zeros((3, 3))
    for start in range(0, len(analysis), 8):
        a = np.asarray(basis.spatial_matrix @ analysis[start:start+8].T, float)
        b = np.asarray(basis.spatial_matrix @ reference[start:start+8].T, float)
        for index, (left, right) in enumerate(zip(basis.offsets[:-1], basis.offsets[1:])):
            for column, values in enumerate((a[left:right], b[left:right], a[left:right]-b[left:right])):
                energies[index, column] += np.sum(weights[index][:, None]*values**2)
    report = {}
    for index, name in enumerate(('u', 'v', 'height')):
        rms = np.sqrt(energies[index]/(weights[index].sum()*len(analysis)))
        report[name] = dict(rms=float(rms[0]), reference_rms=float(rms[1]), rms_difference=float(rms[2]),
            relative_difference=float(rms[2]/rms[1]) if rms[1] > 0 else None)
    return report


def validate_trajectory(states, land_masks):
    """Land NaNs are intentional; only nonfinite prognostic ocean cells fail."""
    report = {}
    for name, values in states.items():
        values = np.asarray(values)
        land = land_masks[name]
        wet = np.ones(values.shape[1:], bool) if land is None else ~np.asarray(land, bool)
        if wet.shape != values.shape[1:]:
            raise ValueError(f'Trajectory mask shape mismatch: {name}')
        bad = ~np.isfinite(values[:, wet])
        report[name] = dict(nonfinite_wet=int(bad.sum()),
                           nonfinite_land=int((~np.isfinite(values[:, ~wet])).sum()))
        if bad.any():
            first = int(np.flatnonzero(np.any(bad, axis=1))[0])
            raise FloatingPointError(f'Nonfinite ocean trajectory: {name}, day index {first}')
    return report


@dataclass
class MapOptions:
    maxiter: int = 3000
    maxcor: int = 20
    gtol: float = 1.e-3
    ftol: float = 1.e-9
    checkpoint_every: int = 25
    preconditioner: str = 'prior'


def gradient_group_scaling(gradient, groups, lower=.1, upper=10.):
    """Fixed D for z=z_origin+D*y; D² balances initial group gradient RMS.

    This is a gradient-based heuristic, not an estimated Hessian. Keep a
    bounded positive transform so the physical objective/prior is unchanged.
    """
    g=np.asarray(gradient,float).ravel(); groups=np.asarray(groups)
    if groups.shape != g.shape or not np.isfinite(g).all():
        raise ValueError('Invalid preconditioner gradient/groups')
    _,ids=np.unique(groups,return_inverse=True)
    rms=np.sqrt(np.bincount(ids,weights=g*g)/np.bincount(ids))
    active=rms[rms>0]
    reference=float(np.median(active)) if len(active) else 1.
    factor=np.clip(np.sqrt(reference/np.maximum(rms,reference*1.e-12)),lower,upper)
    return factor[ids], dict(groups=len(rms),reference_gradient_rms=reference,
        scale_quantiles=np.quantile(factor,[0,.1,.5,.9,1]).tolist(),
        clipped_groups=int(np.sum((factor==lower)|(factor==upper))),
        formula='D=sqrt(median_positive_group_RMS/group_RMS), clipped to [0.1,10]; z=z_origin+D*y')


def map_only(residual, std, tdec, mean, initial, output, signature, options=None, resume=False,
             independent=False, preconditioner_groups=None):
    """MAP with periodic atomic checkpoints; no linearization, CG, or sampling."""
    options = options or MapOptions()
    if (options.maxiter < 1 or options.maxcor < 1 or options.checkpoint_every < 1
            or options.gtol <= 0 or options.ftol <= 0):
        raise ValueError('Invalid MAP options')
    if options.preconditioner not in ('prior','component_scale','local_atom'):
        raise ValueError('Unknown MAP preconditioner')
    if options.preconditioner != 'prior' and (not independent or preconditioner_groups is None):
        raise ValueError('Grouped preconditioning requires independent controls and groups')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    signature_data = dict(problem=signature, options=asdict(options))
    if independent:
        signature_data['covariance'] = 'independent_temporal_wavelets'
    signature = json.dumps(signature_data, sort_keys=True)
    shape = initial.shape
    if independent:
        from .covariance import DiagonalCovariance
        covariance = DiagonalCovariance(jnp.asarray(std, dtype=jnp.float32))
    else:
        covariance = DailyAR1Covariance(std, tdec, shape[0])
    center = jnp.tile(jnp.asarray(mean, dtype=jnp.float32), shape[0])
    def cost(z):
        r = residual((center+covariance.sqrt(z)).reshape(shape)).ravel()
        return .5*(jnp.vdot(r, r)+jnp.vdot(z, z))
    evaluate = jax.jit(jax.value_and_grad(cost))
    checkpoint = output/'map_checkpoint.npz'
    z0 = np.asarray(covariance.inverse_sqrt(jnp.asarray(initial, dtype=center.dtype).ravel()-center), float)
    history, offset = [], 0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError('MAP checkpoint exists; use --resume')
        with np.load(checkpoint) as saved:
            if str(saved['signature']) != signature:
                raise ValueError('MAP checkpoint signature mismatch')
            z0 = saved['z'].astype(float)
            history = json.loads(str(saved['history']))
            offset = int(saved['iterations'])
            if bool(saved['success']):
                return np.asarray(center+covariance.sqrt(jnp.asarray(z0, center.dtype))).reshape(shape), json.loads(str(saved['diagnostics']))
    started, latest, count, evaluations = time.perf_counter(), {}, 0, 0
    origin=np.zeros_like(z0); transform=np.ones_like(z0)
    preconditioner_report=dict(strategy=options.preconditioner,setup_evaluations=0,setup_seconds=0.)
    if options.preconditioner != 'prior':
        path=output/'map_preconditioner.npz'
        if resume and path.exists():
            with np.load(path) as saved:
                if str(saved['signature']) != signature:
                    raise ValueError('Preconditioner signature mismatch')
                transform=saved['scale'].copy();origin=saved['origin'].copy()
                preconditioner_report=json.loads(str(saved['report']))
        else:
            origin=np.asarray(covariance.inverse_sqrt(jnp.asarray(initial, dtype=center.dtype).ravel()-center),float)
            value,gradient=evaluate(jnp.asarray(origin,dtype=center.dtype))
            if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
                raise FloatingPointError('Nonfinite initial gradient for preconditioner')
            transform,details=gradient_group_scaling(np.asarray(gradient),preconditioner_groups)
            preconditioner_report.update(details,setup_evaluations=1,setup_seconds=time.perf_counter()-started)
            atomic_npz(path,scale=transform,origin=origin,signature=signature,report=json.dumps(preconditioner_report))
            evaluations=1
        if transform.shape != z0.shape or np.any(transform<=0) or not np.isfinite(transform).all():
            raise ValueError('Invalid preconditioner transform')
        print('MAP preconditioner',json.dumps(preconditioner_report),flush=True)
    canonical_initial=z0.copy()
    z0=(z0-origin)/transform
    accepted_z, accepted_gradient = z0.copy(), None
    accepted_cost = None
    rejected = 0
    def fun(z):
        nonlocal evaluations, rejected, accepted_gradient, accepted_cost
        canonical=origin+transform*z
        value, gradient = evaluate(jnp.asarray(canonical, dtype=center.dtype))
        value, gradient = float(value), np.asarray(gradient, float)
        canonical_gradient_inf=float(np.max(np.abs(gradient)))
        gradient=gradient*transform
        evaluations += 1
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            rejected += 1
            failure = dict(evaluation=evaluations, iteration=offset+count,
                           cost_finite=bool(np.isfinite(value)),
                           gradient_nonfinite=int((~np.isfinite(gradient)).sum()))
            failures = output/'rejected_trials'; failures.mkdir(exist_ok=True)
            atomic_npz(failures/f'trial_{time.time_ns()}.npz', z=canonical,
                       last_accepted=origin+transform*accepted_z, diagnostics=json.dumps(failure))
            print('MAP rejected trial', json.dumps(failure), flush=True)
            if accepted_gradient is None:
                raise FloatingPointError('Nonfinite initial MAP cost/gradient; diagnostic trial saved')
            # Finite rejection barrier: infinity makes SciPy's interpolation
            # jump back to zero and can falsely satisfy ftol. The barrier is
            # above the accepted cost, with positive slope along the trial
            # direction, so the line search brackets and shortens that step.
            # It is NOT the physical objective; callback/final guards forbid
            # accepting it. Keep its scale moderate to avoid vanishing steps.
            delta = np.asarray(z)-accepted_z
            norm2 = float(np.vdot(delta, delta))
            if not np.isfinite(norm2) or norm2 <= 0:
                raise FloatingPointError('Invalid MAP trial displacement; diagnostic saved')
            scale = max(1., abs(accepted_cost), abs(float(np.vdot(accepted_gradient, delta))))
            return accepted_cost+scale, (2*scale/norm2)*delta
        if accepted_gradient is None:
            accepted_gradient = gradient.copy()
            accepted_cost = value
        latest.update(cost=value, gradient_inf=canonical_gradient_inf,
                      optimizer_gradient_inf=float(np.max(np.abs(gradient))))
        latest['valid_evaluation'] = evaluations
        fun.last_gradient = gradient
        return value, gradient
    def save(z, success, diagnostics):
        atomic_npz(checkpoint, z=np.asarray(origin+transform*z, np.float32), signature=signature,
                   success=success, iterations=offset+count, history=json.dumps(history),
                   diagnostics=json.dumps(diagnostics))
    def callback(z):
        nonlocal count, accepted_z, accepted_gradient, accepted_cost
        if latest.get('valid_evaluation') != evaluations:
            raise FloatingPointError('Optimizer accepted an invalid MAP trial; checkpoint retained')
        accepted_z, accepted_gradient = np.asarray(z).copy(), fun.last_gradient.copy()
        accepted_cost = latest['cost']
        count += 1
        row = dict(iteration=offset+count, seconds=time.perf_counter()-started,
                   evaluations=evaluations, rejected_trials=rejected, **latest)
        history.append(row)
        if count == 1 or count % 10 == 0:
            print('MAP', json.dumps(row), flush=True)
        if count % options.checkpoint_every == 0:
            save(z, False, row)
        if latest['gradient_inf'] <= options.gtol:
            raise StopIteration
    initial_cost, _ = fun(z0)
    initial_prior_cost = float(.5*np.vdot(canonical_initial, canonical_initial))
    fit = minimize(fun, z0, jac=True, method='L-BFGS-B', callback=callback,
                   options=dict(maxiter=options.maxiter, maxcor=options.maxcor,
                                gtol=options.gtol*float(transform.min()), ftol=options.ftol, maxls=40))
    if fit.success and latest.get('valid_evaluation') != evaluations:
        raise FloatingPointError('Optimizer returned an invalid MAP trial; checkpoint retained')
    canonical_gradient=np.asarray(fit.jac)/transform
    gradient_converged=bool(np.max(np.abs(canonical_gradient)) <= options.gtol)
    if gradient_converged and fit.status==99:
        fit.success=True;fit.message='Canonical prior-coordinate gradient tolerance reached'
    z = jnp.asarray(origin+transform*fit.x, dtype=center.dtype)
    prior_cost = float(.5*jnp.vdot(z, z))
    diagnostics = dict(success=bool(fit.success), message=str(fit.message), iterations=offset+int(fit.nit),
        evaluations=evaluations, rejected_trials=rejected, seconds=time.perf_counter()-started, initial_cost=initial_cost,
        initial_prior_cost=initial_prior_cost, initial_observation_cost=initial_cost-initial_prior_cost,
        cost=float(fit.fun), gradient_inf=float(np.max(np.abs(canonical_gradient))),
        optimizer_gradient_inf=float(np.max(np.abs(fit.jac))),preconditioner=preconditioner_report,
        gradient_converged=gradient_converged,
        prior_cost=prior_cost, observation_cost=float(fit.fun)-prior_cost,
        precision='float32 SW and cost; cost stopping does not certify gradient convergence',
        posterior_covariance=False)
    save(fit.x, bool(fit.success), diagnostics)
    (output/'map_history.json').write_text(json.dumps(history, indent=2))
    (output/'map_result.json').write_text(json.dumps(diagnostics, indent=2))
    if not fit.success:
        raise RuntimeError(f'MAP incomplete; checkpoint retained: {fit.message}')
    return np.asarray(center+covariance.sqrt(z)).reshape(shape), diagnostics
