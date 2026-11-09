"""Bounded, synchronized GPU measurements for annual SW learning sizing.

No parameter fitting: kernels are evaluated at the generic prior mean. Compiled
memory estimates are checked before executing large cost/gradient/CG kernels.
"""
import json
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def profile_problem(residual, basis, output, repeats=3, max_gib=20., cg_maxiter=30):
    if repeats < 1 or max_gib <= 0 or cg_maxiter < 1:
        raise ValueError('Positive profiling limits required')
    output = Path(output)
    device = jax.devices('gpu')[0]
    budget = int(max_gib*2**30)
    report = dict(device=str(device), precision=str(jnp.asarray(0.).dtype),
                  ntime=basis.ntime, natom=basis.natom, controls=basis.nbasis,
                  matrix_nnz=int(basis.spatial_matrix.nnz),
                  matrix_host_bytes=sum(a.nbytes for a in (basis.spatial_matrix.data, basis.spatial_matrix.indices, basis.spatial_matrix.indptr)),
                  budget_bytes=budget, linearization='generic prior mean, not optimized MAP', stages={})
    def snapshot():
        return dict(host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                    gpu={key: int(value) for key, value in (device.memory_stats() or {}).items()})
    def write(stage, **values):
        report['stages'][stage] = dict(**values, **snapshot())
        tmp = output/'profile.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(output/'profile.json')
        print('PROFILE', stage, json.dumps(values), flush=True)
    def compile_checked(name, function, *args):
        started = time.perf_counter()
        executable = jax.jit(function).lower(*args).compile()
        mem = executable.memory_analysis()
        memory = {k: int(getattr(mem, k, 0)) for k in (
            'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
            'alias_size_in_bytes', 'generated_code_size_in_bytes')}
        stats = device.memory_stats() or {}
        # Existing arrays include arguments and captured constants. Outputs and
        # workspace are additional allocations; this is intentionally conservative.
        estimate = stats.get('bytes_in_use', 0)+memory['output_size_in_bytes']+memory['temp_size_in_bytes']
        effective_budget = min(budget, int(.9*stats.get('bytes_limit', budget/.9)))
        write(name+'_compile', seconds=time.perf_counter()-started, memory=memory,
              estimated_live_plus_workspace_bytes=int(estimate), effective_budget_bytes=effective_budget,
              allowed=estimate <= effective_budget)
        if estimate > effective_budget:
            report['stopped_before'] = name
            write('stopped', reason='compiled memory estimate exceeds profiling budget')
            return None
        return executable
    def timed(name, executable, *args):
        started = time.perf_counter()
        result = executable(*args)
        jax.block_until_ready(result)
        first = time.perf_counter()-started
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = executable(*args)
            jax.block_until_ready(result)
            durations.append(time.perf_counter()-started)
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(result))
        write(name, first_seconds=first, repeat_seconds=durations, median_seconds=float(np.median(durations)), finite=finite)
        if not finite:
            raise FloatingPointError(f'Nonfinite {name} result')
        return result
    covariance = basis.covariance_operator
    center = jnp.tile(jnp.asarray(basis.mean), basis.ntime)
    z = jnp.zeros(basis.nbasis, dtype=center.dtype)
    def white_residual(x):
        return residual((center+covariance.sqrt(x)).reshape(basis.ntime, basis.natom))
    def cost(x):
        r = white_residual(x)
        return .5*(jnp.vdot(x, x)+jnp.vdot(r, r))
    write('setup', **{key: int(value) for key, value in dict(
        control_float32_bytes=4*basis.nbasis,
        control_float64_bytes=8*basis.nbasis,
        posterior_moments_host_bytes=8*basis.natom*(2*basis.ntime-1),
        scipy_lbfgs_workspace_lower_bound_bytes=8*basis.nbasis*(2*5+5)).items()})
    forward = compile_checked('forward', white_residual, z)
    if forward is None:
        return report
    r = timed('forward', forward, z)
    report['observation_slots'] = r.size
    report['residual_float32_bytes'] = 4*r.size
    gradient = compile_checked('cost_gradient', jax.value_and_grad(cost), z)
    if gradient is None:
        return report
    value, grad = timed('cost_gradient', gradient, z)
    report['initial_cost'] = float(value)
    report['initial_gradient_inf'] = float(jnp.max(jnp.abs(grad)))
    # This is the same non-jitted linearization entry point as posterior_step.
    started = time.perf_counter()
    r, tangent = jax.linearize(white_residual, z)
    jax.block_until_ready(r)
    transpose = jax.linear_transpose(tangent, jnp.zeros_like(z))
    write('linearize', seconds=time.perf_counter()-started)
    def hessian(v):
        return v+transpose(tangent(v))[0]
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(z.shape), dtype=z.dtype)
    hv = compile_checked('hessian_vector', hessian, v)
    if hv is None:
        return report
    hv_v = timed('hessian_vector', hv, v)
    w = jnp.asarray(rng.standard_normal(z.shape), dtype=z.dtype)
    hv_w = hv(w)
    lhs, right = jnp.vdot(w, hv_v), jnp.vdot(v, hv_w)
    write('hessian_checks',
          symmetry_relative_error=float(abs(lhs-right)/jnp.maximum(jnp.maximum(abs(lhs), abs(right)), 1.)),
          rayleigh_quotient=float(jnp.vdot(v, hv_v)/jnp.vdot(v, v)))
    xi = jnp.asarray(rng.standard_normal(r.shape), dtype=z.dtype)
    started = time.perf_counter()
    rhs = v+transpose(xi)[0]
    rhs.block_until_ready()
    write('posterior_rhs', seconds=time.perf_counter()-started)
    def cg(b):
        # Explicit counter for sizing; same unpreconditioned CG as production.
        zero = jnp.zeros_like(b)
        squared = jnp.vdot(b, b)
        def condition(carry):
            i, x, r, p, rr, history = carry
            return (i < cg_maxiter) & (rr > 1.e-8*squared)
        def step(carry):
            i, x, r, p, rr, history = carry
            hp = hessian(p)
            alpha = rr/jnp.vdot(p, hp)
            x, r = x+alpha*p, r-alpha*hp
            new_rr = jnp.vdot(r, r)
            history = history.at[i+1].set(jnp.sqrt(new_rr/squared))
            return i+1, x, r, r+(new_rr/rr)*p, new_rr, history
        history = jnp.zeros(cg_maxiter+1, dtype=b.dtype).at[0].set(1.)
        i, x, r, p, rr, history = lax.while_loop(condition, step, (0, zero, b, b, squared, history))
        return i, x, jnp.sqrt(rr/squared), history
    cg_kernel = compile_checked('cg', cg, rhs)
    if cg_kernel is None:
        return report
    count, sample, relative, history = timed('cg', cg_kernel, rhs)
    true_relative = float(jnp.linalg.norm(hv(sample)-rhs)/jnp.linalg.norm(rhs))
    report['cg_iterations'] = int(count)
    report['cg_recursive_relative_residual'] = float(relative)
    report['cg_relative_residual'] = true_relative
    report['cg_converged'] = bool(true_relative <= 5.e-4)
    report['cg_residual_history'] = np.asarray(history)[:int(count)+1].tolist()
    write('complete', cg_iterations=int(count), cg_relative_residual=true_relative,
          cg_recursive_relative_residual=float(relative))
    return report
