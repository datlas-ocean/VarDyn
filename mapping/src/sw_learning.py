"""Approximate EM for nonlinear observations of daily SW potential controls.

The E step uses a MAP/Laplace approximation and randomized Gauss--Newton
posterior samples. The M step includes posterior variance AND lag covariance.
Windowed fits share annual parameters through pooled posterior moments.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .basis import DailyAR1Covariance


@dataclass
class LearningOptions:
    outer_iterations: int = 8
    pooled_iterations: int = 2
    map_maxiter: int = 100
    map_maxcor: int = 20
    cg_preconditioner: str = "none"
    cg_spectral_rank: int = 256
    map_gtol: float = 1.e-3
    probes: int = 16
    cg_rtol: float = 1.e-4
    cg_maxiter: int = 300
    tdec_min: float = 1.
    tdec_max: float = 60.
    spatial_penalty: float = 10.
    regional_penalty: float = 1.
    tolerance: float = .02
    seed: int = 0

    def validate(self):
        if (self.cg_preconditioner not in ("none", "spectral", "spectral-host")
                or self.cg_spectral_rank < 1 or self.map_maxcor < 1 or self.outer_iterations < 1 or self.pooled_iterations < 0 or self.probes < 2
                or self.cg_maxiter < 1 or self.map_maxiter < 1 or self.cg_rtol <= 0
                or not 0 < self.tdec_min < self.tdec_max
                or self.spatial_penalty < 0 or self.regional_penalty < 0):
            raise ValueError('Invalid learning options')


def parameter_groups(records):
    """Component/scale pooled parameters, with separate basin means."""
    keys = [(int(r[0]), float(r[1]), int(r[5]) if int(r[0]) == 3 else 0) for r in records]
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    return groups


def spatial_edges(records):
    """Four nearest neighbors at the same scale/component/basin."""
    pairs = set()
    keys = records[:, [0, 1, 5]]
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if len(ids) < 2 or records[ids[0], 0] == 3:
            continue
        dist, neighbors = cKDTree(records[ids, 3:5]).query(records[ids, 3:5], k=min(5, len(ids)))
        for row, (ds, js) in enumerate(zip(dist, neighbors)):
            for d, j in zip(np.atleast_1d(ds), np.atleast_1d(js)):
                if d > 0 and d <= records[ids[row], 1]*1000:
                    pairs.add(tuple(sorted((int(ids[row]), int(ids[j])))))
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def posterior_step(residual, std, tdec, mean, initial, options, seed, map_checkpoint=None, map_signature=""):
    """Return physical MAP and marginal/lag covariance estimated from samples.

    Each sample solves (I + J'J) z = eta + J'xi in whitened coordinates.
    Its covariance is (I + J'J)^-1. Failed CG solves abort rather than silently
    supplying biased moments to EM. Residual must already include R^-1/2.
    """
    shape = initial.shape
    covariance = DailyAR1Covariance(std, tdec, shape[0])
    center = jnp.tile(jnp.asarray(mean), shape[0])
    def white_residual(z):
        return residual((center+covariance.sqrt(z)).reshape(shape)).ravel()
    def cost(z):
        r = white_residual(z)
        return .5*(jnp.vdot(r, r)+jnp.vdot(z, z))
    evaluate = jax.jit(jax.value_and_grad(cost))
    def scipy_cost(z):
        value, grad = evaluate(jnp.asarray(z, dtype=center.dtype))
        value, grad = float(value), np.asarray(grad, dtype=float)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('Nonfinite assimilation cost/gradient')
        return value, grad
    z0 = np.asarray(covariance.inverse_sqrt(jnp.asarray(initial).ravel()-center), float)
    fit = None
    if map_checkpoint is not None and Path(map_checkpoint).exists():
        with np.load(map_checkpoint) as saved:
            if str(saved['signature']) != map_signature:
                raise ValueError('MAP checkpoint mismatch')
            z0 = saved['z'].astype(float)
            if bool(saved['success']):
                info = json.loads(str(saved['diagnostics']))
                fit = SimpleNamespace(x=z0, fun=info['cost'], jac=np.array([info['gradient_inf']]),
                                      nit=info['iterations'], message=info['message'])
    if fit is None:
        count = [0]
        def progress(z):
            count[0] += 1
            if count[0] % 25 == 0:
                print(f'MAP iteration {count[0]}', flush=True)
        fit = minimize(scipy_cost, z0, jac=True, method='L-BFGS-B', callback=progress,
                       options=dict(maxiter=options.map_maxiter, gtol=options.map_gtol, ftol=1.e-10,
                                    maxcor=options.map_maxcor, maxls=30))
        if map_checkpoint is not None:
            path = Path(map_checkpoint)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, z=fit.x, signature=map_signature, success=fit.success,
                    diagnostics=json.dumps(dict(cost=float(fit.fun), gradient_inf=float(np.max(np.abs(fit.jac))),
                                                iterations=int(fit.nit), message=str(fit.message))))
            path.with_suffix('.tmp').replace(path)
        if not fit.success:
            raise RuntimeError(f'MAP did not converge; checkpoint saved if enabled: {fit.message}')
    z = jnp.asarray(fit.x, dtype=center.dtype)
    r, tangent = jax.linearize(white_residual, z)
    transpose = jax.linear_transpose(tangent, jnp.zeros_like(z))
    def hessian(v):
        return v+transpose(tangent(v))[0]
    precondition, precondition_report = None, None
    spectral_factors = None
    if options.cg_preconditioner == 'spectral':
        from .sw_learning_spectral import spectral_apply, spectral_factors as build
        Q, values, precondition_report = build(jax.jit(hessian), z, rank=options.cg_spectral_rank, seed=seed+99129)
        spectral_factors = (Q, values)
        # Q and values are dynamic inputs: do not capture a 2+ GiB Lanczos
        # basis as an XLA executable constant.
        def solve_with_spectral(b, Q, values):
            return jax.scipy.sparse.linalg.cg(
                hessian, b, M=lambda v: spectral_apply(Q, values, v),
                tol=options.cg_rtol, maxiter=options.cg_maxiter)[0]
        solve = jax.jit(solve_with_spectral)
    else:
        if options.cg_preconditioner == 'spectral-host':
            from .sw_learning_host_spectral import host_spectral_inverse as build
            precondition, precondition_report = build(jax.jit(hessian), z, rank=options.cg_spectral_rank, seed=seed+99129)
        solve = jax.jit(lambda b: jax.scipy.sparse.linalg.cg(hessian, b, M=precondition, tol=options.cg_rtol, maxiter=options.cg_maxiter)[0])
    rng = np.random.default_rng(seed)
    variance, lag = np.zeros(shape), np.zeros((shape[0]-1, shape[1]))
    cg_errors = []
    for _ in range(options.probes):
        eta = jnp.asarray(rng.standard_normal(z.shape), dtype=z.dtype)
        xi = jnp.asarray(rng.standard_normal(r.shape), dtype=z.dtype)
        rhs = eta+transpose(xi)[0]
        sample = solve(rhs, *spectral_factors) if spectral_factors is not None else solve(rhs)
        error = float(jnp.linalg.norm(hessian(sample)-rhs)/jnp.maximum(jnp.linalg.norm(rhs), 1.e-30))
        if not np.isfinite(error) or error > max(5*options.cg_rtol, 1.e-5):
            raise RuntimeError(f'Posterior CG failed: relative residual {error:g}')
        cg_errors.append(error)
        physical = np.asarray(covariance.sqrt(sample)).reshape(shape)
        variance += physical**2/options.probes
        lag += physical[1:]*physical[:-1]/options.probes
    analysis = np.asarray(center+covariance.sqrt(z)).reshape(shape)
    return analysis, variance, lag, dict(map_cost=float(fit.fun), map_iterations=int(fit.nit),
        map_gradient_inf=float(np.max(np.abs(fit.jac))), map_message=str(fit.message),
        preconditioner=precondition_report, cg_max_residual=max(cg_errors), probes=options.probes)


def maximize_statistics(analysis, variance, lag, std, tdec, groups, edges, anchor, options):
    """Expected complete-data AR1 likelihood plus smooth log-parameter penalty."""
    from .sw_learning_moments import window_moments
    return maximize_window_statistics(window_moments(analysis, variance, lag), std, tdec, groups, edges, anchor, options)


def maximize_window_statistics(moments, std, tdec, groups, edges, anchor, options):
    """One M-step for all windows; penalties are applied once, no bridge pairs."""
    from .sw_learning_moments import expected_ar1, common_mean
    natom = len(std)
    groups = np.asarray(groups, int)
    ngroup = groups.max()+1
    index, edge = jnp.asarray(groups), jnp.asarray(edges)
    anchor = jnp.asarray(anchor)
    initial = np.array([[np.mean(np.log(v[groups == g])) for g in range(ngroup)] for v in (std, tdec)])
    def objective(flat):
        log_std, log_tau = flat.reshape(2, ngroup)[:, index]
        value = expected_ar1(moments, log_std, log_tau)
        local = jnp.stack((log_std, log_tau))
        if edges.size:
            value += .5*options.spatial_penalty*jnp.sum((local[:, edge[:, 0]]-local[:, edge[:, 1]])**2)
        if ngroup == natom:
            value += .5*options.regional_penalty*jnp.sum((local-anchor)**2)
        return value
    evaluate = jax.jit(jax.value_and_grad(objective))
    def wrapper(x):
        # Parameter-only optimization: double precision avoids float32 line
        # search stagnation. This context does not change SW model precision.
        with jax.experimental.enable_x64():
            value, gradient = evaluate(jnp.asarray(x, dtype=jnp.float64))
            return float(value), np.asarray(gradient, float)
    bounds = [(float(v-8), float(v+8)) for v in initial[0]]+[(np.log(options.tdec_min), np.log(options.tdec_max))]*ngroup
    result = minimize(wrapper, initial.ravel(), jac=True, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=500, ftol=1.e-9, gtol=1.e-4, maxls=40))
    if not result.success:
        raise RuntimeError(f'Hyperparameter fit failed: {result.message}')
    log_std, log_tau = result.x.reshape(2, ngroup)[:, groups]
    rho = np.exp(-np.exp(-log_tau))
    mean = moments['reference']+common_mean(moments, rho)
    return np.exp(log_std), np.exp(log_tau), mean, float(result.fun)


def learn(residual, basis, output, options=None, validation_residual=None, resume=False, provenance=''):
    """Run pooled then local EM, checkpointing every complete iteration.

    A final E step makes the saved trajectory consistent with the final prior.
    Restart checks the layout, experiment provenance, and numerical options.
    """
    options = options or LearningOptions()
    options.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'learning_latest.npz'
    signature = json.dumps(dict(layout=basis.layout_signature, provenance=provenance, options=asdict(options)), sort_keys=True)
    groups = parameter_groups(basis.records)
    edges = spatial_edges(basis.records)
    std, tdec, mean = basis.std.copy(), basis.tdec.copy(), basis.mean.copy()
    analysis = np.tile(mean, (basis.ntime, 1))
    anchor = np.log(np.stack((std, tdec)))
    history, start = [], 0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f'Use resume or a new output directory: {checkpoint}')
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['signature']) != signature:
                raise ValueError('Learning checkpoint configuration/layout mismatch')
            std, tdec, mean, analysis, anchor = [saved[k].copy() for k in ('std', 'tdec', 'mean', 'analysis', 'anchor')]
            history = json.loads(str(saved['history']))
            start = len(history)
            if history and history[-1]['phase'] == 'local' and history[-1]['parameter_change'] < options.tolerance:
                start = options.outer_iterations
    for iteration in range(start, options.outer_iterations):
        print(f'EM iteration {iteration+1}/{options.outer_iterations}', flush=True)
        analysis, variance, lag, diagnostics = posterior_step(residual, std, tdec, mean, analysis, options, options.seed+iteration)
        pooled = iteration < options.pooled_iterations
        new_std, new_tdec, new_mean, objective = maximize_statistics(analysis, variance, lag, std, tdec,
            groups if pooled else np.arange(basis.natom), np.empty((0, 2), int) if pooled else edges, anchor, options)
        change = max(float(np.max(np.abs(np.log(new_std/std)))), float(np.max(np.abs(np.log(new_tdec/tdec)))),
                     float(np.max(np.abs(new_mean-mean)/std)))
        std, tdec, mean = new_std, new_tdec, new_mean
        if pooled:
            anchor = np.log(np.stack((std, tdec)))
        diagnostics.update(iteration=iteration+1, phase='pooled' if pooled else 'local', parameter_change=change,
                           expected_objective=objective)
        if validation_residual is not None:
            diagnostics['validation_squared_residual'] = float(jnp.sum(validation_residual(jnp.asarray(analysis))**2))
        history.append(diagnostics)
        with (output/'learning_latest.tmp').open('wb') as stream:
            np.savez_compressed(stream, std=std, tdec=tdec, mean=mean, analysis=analysis, anchor=anchor,
                                signature=signature, history=json.dumps(history))
        (output/'learning_latest.tmp').replace(checkpoint)
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(diagnostics), flush=True)
        if not pooled and change < options.tolerance:
            break
    analysis, variance, lag, diagnostics = posterior_step(residual, std, tdec, mean, analysis, options, options.seed+100000)
    if validation_residual is not None:
        diagnostics['validation_squared_residual'] = float(jnp.sum(validation_residual(jnp.asarray(analysis))**2))
    basis.update_statistics(std, tdec, mean)
    np.savez_compressed(output/'posterior.npz', analysis=analysis, variance=variance, lag_covariance=lag,
        std=std, tdec=tdec, mean=mean, records=basis.records, start_day=basis.start,
        origin_day=basis.origin, signature=signature)
    (output/'result.json').write_text(json.dumps(dict(final_posterior=diagnostics,
        converged=bool(history and history[-1]['phase'] == 'local' and history[-1]['parameter_change'] < options.tolerance),
        iterations=len(history), normalization='unit-peak scalar potentials; support-RMS height',
        approximation='MAP plus sampled Gauss-Newton covariance; approximate EM'), indent=2))
    return dict(analysis=analysis, variance=variance, lag_covariance=lag, std=std, tdec=tdec, mean=mean, history=history)
