"""Device-resident MAP continuation using the standard decoupled optimizer."""
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .inv import minimize_optax_decoupled
from .sw_learning_empirical import gradient_group_scaling
from .sw_learning_windowed import atomic_npz


def joint_optax_map(residual, initial, groups, output, signature, config,
                    restart_from=None, resume=False):
    if config.minimizer!='optax-decoupled' or not config.jit_cost_and_grad or config.cost_and_grad_schedule!='scan':
        raise ValueError('This driver requires optax-decoupled, JIT and scan')
    output=Path(output);started=time.perf_counter();history=[];offset=0
    def cost(z):
        r=residual(z).ravel()
        return .5*(jnp.vdot(r,r)+jnp.vdot(z,z))
    canonical_evaluate=jax.jit(jax.value_and_grad(cost))
    checkpoint=output/'map_checkpoint.npz'
    source=output if resume else (Path(restart_from) if restart_from else None)
    if source is not None:
        with np.load(source/'map_checkpoint.npz') as data:
            z=data['z'].copy()
            if resume:
                if str(data['signature'])!=signature:raise ValueError('Checkpoint provenance mismatch')
                history=json.loads(str(data['history']));offset=int(data['iterations'])
        with np.load(source/'map_preconditioner.npz') as data:
            scale=data['scale'].copy();origin=data['origin'].copy()
    else:
        z=np.asarray(initial).ravel();origin=z.copy()
        value,gradient=canonical_evaluate(jnp.asarray(z))
        if groups is None:
            scale=np.ones_like(z)
        else:
            scale,_=gradient_group_scaling(np.asarray(gradient),groups)
    if z.shape!=initial.ravel().shape or scale.shape!=z.shape or not np.isfinite(z).all():
        raise ValueError('Incompatible continuation control')
    if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid preconditioner')
    atomic_npz(output/'map_preconditioner.npz',scale=scale,origin=origin,signature=signature)
    origin=jnp.asarray(origin,dtype=jnp.float32);scale=jnp.asarray(scale,dtype=jnp.float32)
    y=(jnp.asarray(z,dtype=jnp.float32)-origin)/scale
    @jax.jit
    def evaluate(y):
        value,gradient=canonical_evaluate(origin+scale*y)
        return value,gradient*scale
    value,gradient=evaluate(y);initial_cost=float(value)
    if not np.isfinite(initial_cost) or not bool(jnp.all(jnp.isfinite(gradient))):
        raise FloatingPointError('Nonfinite continuation state')
    if restart_from and not resume:
        old=json.loads((Path(restart_from)/'map_result.json').read_text())
        if not np.isclose(initial_cost,old['cost'],rtol=2.e-6,atol=1.):
            raise ValueError(f'Continuation cost mismatch: {initial_cost} vs {old["cost"]}')
    initial_prior=float(.5*jnp.vdot(origin+scale*y,origin+scale*y))
    print('OPTAX INITIAL',json.dumps(dict(cost=initial_cost,config=dict(config),restart_from=str(restart_from)),default=str),flush=True)
    attempts=[];completed=0;evals=1
    def save(params,success,diagnostics):
        atomic_npz(checkpoint,z=np.asarray(origin+scale*params),signature=signature,
            success=success,iterations=offset+completed,history=json.dumps(history),diagnostics=json.dumps(diagnostics))
        (output/'map_history.json').write_text(json.dumps(history,indent=2))
    save(y,False,dict(cost=initial_cost))
    maxiter=int(config.maxiter)
    result=None
    for retry in range(int(config.max_retries)+1):
        base=completed
        def callback(iteration,params,cost,grad,details):
            nonlocal completed
            completed=base+iteration
            row=dict(iteration=offset+completed,cost=cost,seconds=time.perf_counter()-started,
                gradient_inf=float(jnp.max(jnp.abs(grad/scale))),retry=retry,**details)
            history.append(row)
            if completed==1 or completed%10==0:print('MAP',json.dumps(row),flush=True)
            if completed%10==0:save(params,False,row)
        result=minimize_optax_decoupled(evaluate,y,maxiter=maxiter-completed,
            history_size=int(config.lbfgs_history_size),gtol=config.gtol,ftol=config.ftol,
            convergence_patience=int(config.convergence_nit),minimum_iterations=int(config.minimum_iterations),
            gradient_max_norm=config.gradient_max_norm,iteration_callback=callback,
            initial_value=value,initial_gradient=gradient)
        attempts.append(dict(status=result.status,iterations=result.iterations_completed,evaluations=result.function_evaluations))
        evals+=result.function_evaluations-1;y=result.control
        value,gradient=evaluate(y);evals+=1
        save(y,result.converged,dict(attempts=attempts,cost=float(value)))
        if result.converged or completed>=maxiter or result.status!='line search failed':break
        print('OPTAX RETRY',retry+1,'resetting L-BFGS history at last accepted state',flush=True)
    z=origin+scale*y;prior=float(.5*jnp.vdot(z,z))
    diagnostics=dict(success=result.converged,message=result.status,iterations=offset+completed,
        evaluations=evals,seconds=time.perf_counter()-started,initial_cost=initial_cost,
        initial_prior_cost=initial_prior,initial_observation_cost=initial_cost-initial_prior,
        cost=float(value),prior_cost=prior,observation_cost=float(value)-prior,
        gradient_inf=float(jnp.max(jnp.abs(gradient/scale))),gradient_converged=False,
        minimizer=config.minimizer,convergence_nit=config.convergence_nit,ftol=config.ftol,
        gtol=config.gtol,preconditioner='prior' if groups is None else 'component_scale',attempts=attempts,
        restart_from=str(restart_from),posterior_covariance=False,
        precision='float32 model, controls and objective',optimizer_history_restarted=True)
    save(y,result.converged,diagnostics)
    (output/'map_result.json').write_text(json.dumps(diagnostics,indent=2))
    if not result.converged:raise RuntimeError('MAP incomplete: '+result.status)
    return np.asarray(z).reshape(initial.shape),diagnostics
