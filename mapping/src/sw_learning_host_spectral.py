"""Host-stored spectral preconditioner for controls too large for GPU storage."""
import jax
import jax.numpy as jnp
import numpy as np
import psutil
import os


def host_spectral_inverse(hessian, template, rank=256, seed=1837):
    """Build a Lanczos subspace in RAM; expose a pure CPU callback to PCG.

    Only vectors (not the full subspace) cross the device boundary on each
    application. This function is for posterior solves, not differentiation.
    The posterior precision and random RHS remain unchanged.
    """
    if rank < 1:
        raise ValueError('Rank must be positive')
    rank = min(rank, template.size)
    dtype = np.dtype(template.dtype)
    required = (rank+16)*template.size*dtype.itemsize
    if required > .8*psutil.virtual_memory().available:
        raise MemoryError(f'Host spectral basis needs at least {required/2**30:.2f} GiB RAM')
    Q = np.zeros((rank, template.size), dtype=dtype)
    q = np.random.default_rng(seed).standard_normal(template.shape).astype(dtype)
    q /= np.linalg.norm(q)
    small = np.zeros((rank, rank))
    for i in range(rank):
        Q[i] = q
        w = np.array(hessian(jnp.asarray(q)), copy=True)
        rows = Q[:i+1]
        coefficients = rows@w
        small[:i+1, i] = coefficients
        w -= rows.T@coefficients
        w -= rows.T@(rows@w)
        beta = float(np.linalg.norm(w))
        if beta < 1.e-10 or i == rank-1:
            Q = Q[:i+1]
            small = small[:i+1,:i+1]
            break
        q = w/beta
    small = np.triu(small)+np.triu(small,1).T
    values, vectors = np.linalg.eigh(small)
    values = np.maximum(values,1.)
    orthogonality = float(np.max(np.abs(Q@Q.T-np.eye(len(Q)))))
    def apply_host(value):
        v = np.asarray(value, dtype=dtype)
        coeff = Q@v
        perp = v-Q.T@coeff
        low = (vectors@((vectors.T@coeff)/values)).astype(dtype)
        return np.asarray(perp-Q.T@(Q@perp)+Q.T@low, dtype=dtype)
    shape = jax.ShapeDtypeStruct(template.shape, template.dtype)
    def inverse(v):
        return jax.pure_callback(apply_host, shape, v, vmap_method='sequential')
    return inverse, dict(rank=len(Q), storage='host', host_bytes=Q.nbytes,
                        host_blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'), ritz_min=float(values.min()), ritz_max=float(values.max()),
                        orthogonality_max=orthogonality)
