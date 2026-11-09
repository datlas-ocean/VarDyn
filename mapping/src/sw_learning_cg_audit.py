"""Posterior PCG audit using a positive stochastic Gauss--Newton diagonal."""
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .basis import DailyAR1Covariance


def audit(residual, basis, output, probes=4, seed=0, provenance='', method='diagonal', rank=64):
    output = Path(output)
    label = method+str(rank) if method.startswith('spectral') else method
    with np.load(output/'pilot_map.npz') as saved:
        if str(saved['signature']) != provenance+basis.layout_signature or not bool(saved['converged']):
            raise ValueError('A compatible completed MAP is required')
        z = jnp.asarray(saved['z'])
    covariance = DailyAR1Covariance(basis.std, basis.tdec, basis.ntime)
    center = jnp.tile(jnp.asarray(basis.mean), basis.ntime)
    def white(z):
        return residual((center+covariance.sqrt(z)).reshape(basis.ntime, basis.natom)).ravel()
    r, tangent = jax.linearize(white, z)
    transpose = jax.linear_transpose(tangent, jnp.zeros_like(z))
    jt = jax.jit(lambda x: transpose(x)[0])
    hessian = jax.jit(lambda x: x+transpose(tangent(x))[0])
    began = time.perf_counter()
    if method.startswith('spectral'):
        if method == 'spectral-host':
            from .sw_learning_host_spectral import host_spectral_inverse as build
        else:
            from .sw_learning_spectral import spectral_inverse as build
        precondition, report = build(hessian, z, rank=rank, seed=seed+99129)
        report.update(method=method, seconds=time.perf_counter()-began)
    else:
        # E[(J^T xi)^2] is diag(J^T J); each estimate is nonnegative.
        rng_diagonal = np.random.default_rng(seed+99129)
        diagonal = jnp.ones_like(z)
        began = time.perf_counter()
        for i in range(64):
            x = jnp.asarray(rng_diagonal.standard_normal(r.shape), z.dtype)
            diagonal = diagonal+jt(x)**2/64
        diagonal.block_until_ready()
        report = dict(diagonal_probes=64, diagonal_seconds=time.perf_counter()-began,
                      diagonal_quantiles=np.quantile(np.asarray(diagonal), [0,.01,.5,.99,1]).tolist())
        precondition = lambda v: v/diagonal
        report['method'] = method
    print(json.dumps(report), flush=True)
    (output/('pilot_preconditioner_'+label+'.json')).write_text(json.dumps(report, indent=2))
    def advance(b, state, budget):
        norm = jnp.vdot(b,b)
        def cond(s):
            i,x,r,p,rz,rr=s
            return (i < budget) & (rr > 1.e-8*norm)
        def step(s):
            i,x,r,p,rz,rr=s
            hp=hessian(p)
            alpha=rz/jnp.vdot(p,hp)
            x,r=x+alpha*p,r-alpha*hp
            w=precondition(r)
            new_rz=jnp.vdot(r,w)
            return i+1,x,r,w+(new_rz/rz)*p,new_rz,jnp.vdot(r,r)
        return jax.lax.while_loop(cond,step,state)
    kernel=jax.jit(advance)
    rng=np.random.default_rng(seed)
    results=[]
    compiled=None
    for probe in range(probes):
        rhs=jnp.asarray(rng.standard_normal(z.shape),z.dtype)+jt(jnp.asarray(rng.standard_normal(r.shape),z.dtype))
        w=precondition(rhs)
        state=(jnp.array(0),jnp.zeros_like(z),rhs,w,jnp.vdot(rhs,w),jnp.vdot(rhs,rhs))
        elapsed=0.
        for budget in (300,600,1200,2400):
            if compiled is None:
                compiled=kernel.lower(rhs,state,jnp.array(budget)).compile()
            began=time.perf_counter()
            state=compiled(rhs,state,jnp.array(budget))
            state[1].block_until_ready()
            elapsed+=time.perf_counter()-began
            error=float(jnp.linalg.norm(hessian(state[1])-rhs)/jnp.linalg.norm(rhs))
            recursive=float(jnp.sqrt(state[-1]/jnp.vdot(rhs,rhs)))
            row=dict(probe=probe,budget=budget,iterations=int(state[0]),seconds=elapsed,
                     true_relative_residual=error,recursive_relative_residual=recursive,
                     accepted=bool(np.isfinite(error) and error<=5.e-4))
            results.append(row)
            (output/('pilot_pcg_'+label+'.json')).write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
            if row['accepted']:
                break
            # If recursive convergence is false in true residual, restart from
            # the true residual; do not accept a misleading internal criterion.
            if recursive<=1.e-4:
                true_r=rhs-hessian(state[1]); w=precondition(true_r)
                state=(state[0],state[1],true_r,w,jnp.vdot(true_r,w),jnp.vdot(true_r,true_r))
        else:
            raise RuntimeError('Preconditioned posterior CG has not converged')
    return results
