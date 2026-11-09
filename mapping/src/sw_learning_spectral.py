"""Positive low-rank spectral inverse for a matrix-free SPD precision."""
import jax
import jax.numpy as jnp
import numpy as np


def spectral_factors(hessian, template, rank=64, seed=1837):
    """Build GPU-resident Lanczos factors for a positive spectral inverse.

    ``Q`` and ``values`` are deliberately returned as arrays rather than closed
    over by a jitted PCG kernel.  For the annual-window controls the former
    makes XLA embed a multi-GiB constant in the executable.  Passing them as
    dynamic device arguments keeps both the factors and PCG on the GPU.
    """

    if rank < 1:
        raise ValueError('Spectral rank must be positive')
    rank = min(rank, template.size)
    stats = template.device.memory_stats()
    # Rotation temporarily holds two full subspace arrays. Leave headroom
    # for the already compiled forward/adjoint and CG work vectors.
    required = (2*rank+12)*template.size*template.dtype.itemsize
    if stats and required > stats.get('bytes_limit', float('inf'))-stats.get('bytes_in_use', 0):
        raise MemoryError(f'Spectral rank {rank} needs about {required/2**30:.2f} GiB additional device memory; '
                          'reduce rank or use a structured/host-stored preconditioner')
    Q = jnp.zeros((rank, template.size), dtype=template.dtype)
    q = jnp.asarray(np.random.default_rng(seed).standard_normal(template.shape), template.dtype)
    q = q/jnp.linalg.norm(q)
    small = np.zeros((rank, rank))
    dot = lambda a,b: jnp.matmul(a,b,precision=jax.lax.Precision.HIGHEST)
    @jax.jit
    def orthogonalize(Q, w):
        coefficients = dot(Q,w)
        w = w-dot(Q.T,coefficients)
        w = w-dot(Q.T,dot(Q,w))
        return coefficients, w, jnp.linalg.norm(w)
    for i in range(rank):
        Q = Q.at[i].set(q)
        coefficients,w,beta = orthogonalize(Q,hessian(q))
        small[:,i] = np.asarray(coefficients)
        if float(beta) < 1.e-10 or i == rank-1:
            Q = Q[:i+1]
            small = small[:i+1,:i+1]
            break
        q = w/beta
    small = np.triu(small)+np.triu(small,1).T
    values, vectors = np.linalg.eigh(small)
    values = jnp.asarray(np.maximum(values,1.),template.dtype)
    Q = dot(jnp.asarray(vectors.T,template.dtype),Q)
    return Q, values, dict(rank=int(Q.shape[0]), storage='gpu',
                            ritz_min=float(values.min()),ritz_max=float(values.max()),
                            orthogonality_max=float(jnp.max(jnp.abs(dot(Q,Q.T)-jnp.eye(Q.shape[0])))))


def spectral_apply(Q, values, v):
    """Apply factors returned by :func:`spectral_factors` to ``v``."""
    dot = lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
    coeff = dot(Q, v)
    perp = v-dot(Q.T, coeff)
    return perp-dot(Q.T, dot(Q, perp))+dot(Q.T, coeff/values)


def spectral_inverse(hessian, template, rank=64, seed=1837):
    """Return a callable spectral inverse for small standalone uses.

    Large PCG solves should use :func:`spectral_factors` and pass the returned
    arrays as dynamic arguments to their jitted kernel.
    """
    Q, values, report = spectral_factors(hessian, template, rank, seed)
    return jax.jit(lambda v: spectral_apply(Q, values, v)), report
