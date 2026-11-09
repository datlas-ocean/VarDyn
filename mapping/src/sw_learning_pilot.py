"""Checkpointed full-basis MAP pilot and posterior CG budget audit."""
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from .basis import DailyAR1Covariance


def run_pilot(residual, basis, output, maxiter=1000, probes=4, seed=0, provenance=''):
    output = Path(output)
    signature = provenance + basis.layout_signature
    covariance = DailyAR1Covariance(basis.std, basis.tdec, basis.ntime)
    center = jnp.tile(jnp.asarray(basis.mean), basis.ntime)
    def white(z):
        return residual((center + covariance.sqrt(z)).reshape(basis.ntime, basis.natom)).ravel()
    def cost(z):
        r = white(z)
        return .5*(jnp.vdot(r, r)+jnp.vdot(z, z))
    evaluate = jax.jit(jax.value_and_grad(cost))
    start = time.perf_counter()
    history = []
    latest = {}
    def fun(z):
        value, gradient = evaluate(jnp.asarray(z, center.dtype))
        value, gradient = float(value), np.asarray(gradient, float)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite MAP evaluation')
        latest.update(cost=value, gradient_inf=float(np.max(np.abs(gradient))))
        return value, gradient
    checkpoint = output/'pilot_map.npz'
    z0 = np.zeros(center.shape, dtype=float)
    converged = False
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved['signature']) != signature:
                raise ValueError('Pilot checkpoint provenance mismatch')
            z0 = saved['z'].astype(float)
            converged = bool(saved['converged'])
    def save(z, converged):
        with (output/'pilot_map.tmp').open('wb') as f:
            np.savez(f, z=np.asarray(z, np.float32), signature=signature, converged=converged)
        (output/'pilot_map.tmp').replace(checkpoint)
    def callback(z):
        history.append(dict(iteration=len(history)+1, seconds=time.perf_counter()-start, **latest))
        if len(history) % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
            (output/'pilot_map_history.json').write_text(json.dumps(history, indent=2))
        if len(history) % 50 == 0:
            save(z, False)
    if not converged:
        initial_cost, initial_gradient = fun(z0)
        fit = minimize(fun, z0, jac=True, method='L-BFGS-B', callback=callback,
                       options=dict(maxiter=maxiter, gtol=1.e-3, ftol=1.e-10, maxcor=20, maxls=30))
        save(fit.x, bool(fit.success))
        report = dict(success=bool(fit.success), message=str(fit.message), iterations=int(fit.nit),
                      evaluations=int(fit.nfev), maxcor=20, initial_cost=initial_cost, cost=float(fit.fun),
                      initial_gradient_inf=float(np.max(np.abs(initial_gradient))),
                      gradient_inf=float(np.max(np.abs(fit.jac))), seconds=time.perf_counter()-start)
        (output/'pilot_map_result.json').write_text(json.dumps(report, indent=2))
        (output/'pilot_map_history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(report), flush=True)
        if not fit.success:
            raise RuntimeError('MAP incomplete; checkpoint saved, rerun to continue before posterior audit')
        z0 = fit.x
    z = jnp.asarray(z0, center.dtype)
    r, tangent = jax.linearize(white, z)
    transpose = jax.linear_transpose(tangent, jnp.zeros_like(z))
    hessian = jax.jit(lambda v: v+transpose(tangent(v))[0])
    # Same random RHS is used at every budget, preserving the posterior law.
    def cg(b, budget):
        bb = jnp.vdot(b, b)
        def cond(state):
            i, x, r, p, rr = state
            return (i < budget) & (rr > 1.e-8*bb)
        def step(state):
            i, x, r, p, rr = state
            hp = hessian(p)
            alpha = rr/jnp.vdot(p, hp)
            x, r = x+alpha*p, r-alpha*hp
            new_rr = jnp.vdot(r, r)
            return i+1, x, r, r+(new_rr/rr)*p, new_rr
        count, x, _, _, rr = jax.lax.while_loop(cond, step,
            (jnp.array(0), jnp.zeros_like(b), b, b, bb))
        return x, count, jnp.sqrt(rr/bb)
    solve = jax.jit(cg)
    compiled = None
    rng = np.random.default_rng(seed)
    results = []
    for probe in range(probes):
        rhs = jnp.asarray(rng.standard_normal(z.shape), z.dtype)+transpose(jnp.asarray(rng.standard_normal(r.shape), z.dtype))[0]
        for budget in (300, 600, 1200, 2400):
            if compiled is None:
                compiled = solve.lower(rhs, jnp.array(budget)).compile()
            began = time.perf_counter()
            sample, count, recursive = compiled(rhs, jnp.array(budget))
            sample.block_until_ready()
            seconds = time.perf_counter()-began
            error = float(jnp.linalg.norm(hessian(sample)-rhs)/jnp.linalg.norm(rhs))
            row = dict(probe=probe, budget=budget, iterations=int(count), seconds=seconds,
                       recursive_relative_residual=float(recursive), true_relative_residual=error,
                       accepted=bool(np.isfinite(error) and error <= 5.e-4))
            results.append(row)
            (output/'pilot_cg.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
            if row['accepted']:
                break
        else:
            raise RuntimeError('Posterior solve failed even at 2400 iterations; preconditioning required')
    return results
