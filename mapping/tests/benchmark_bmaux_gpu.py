"""Benchmark BMaux on a GPU with a small synthetic auxiliary field.

Run with ``CUDA_VISIBLE_DEVICES=0 PYTHONPATH=mapping python
mapping/tests/benchmark_bmaux_gpu.py``. Set ``BASIS_BENCH_LEGACY=1`` to
measure the previous public adjoint wrapper on the same basis operators.
Set ``BASIS_BENCH_TIME_MEMORY=1`` to isolate resident memory for the
temporal representation, and ``BASIS_BENCH_LEGACY_TIME=1`` to restore the
previous CSR matrices and one-hot time selector for that comparison.
"""

import datetime
import gc
import json
import os
import statistics
import tempfile
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.experimental import sparse as jsparse

from src.basis import (Basis_bmaux, _clear_adstate_names,
                       _ensure_adstate_names, _sum_adstate_names, mywindow)


def make_basis(aux_path):
    lon, lat = np.meshgrid(np.linspace(0., 5., 48),
                           np.linspace(40., 44., 36))
    state = SimpleNamespace(
        ny=lat.shape[0], nx=lon.shape[1], lon=lon, lat=lat,
        lon_min=lon.min(), lon_max=lon.max(),
        lat_min=lat.min(), lat_max=lat.max(),
        lon_unit='0_360', mask=None,
        f=np.full_like(lon, 1.e-4),
        DX=np.full_like(lon, 1.e4), DY=np.full_like(lon, 1.e4),
    )
    config = SimpleNamespace(
        BASIS=SimpleNamespace(
            file_aux=aux_path, flux=False, facns=1., facnlt=2.,
            npsp=3.5, facpsp=1.5, lmin=200., lmax=600.,
            tdecmin=2.5, tdecmax=40., factdec=1., facQ=1.,
            l_largescale=500., facQ_largescale=1., name_mod_var='ssh',
            path_background=None, var_background=None, norm_time=True,
            c_grid_var=None, compute_velocities=False,
            name_mod_u='u', name_mod_v='v', file_depth=None,
            file_facQaux=None,
        ),
        EXP=SimpleNamespace(
            init_date=datetime.datetime(2020, 1, 1),
            tmp_DA_path=tempfile.gettempdir(),
        ),
    )
    basis = Basis_bmaux(config, state)
    basis.set_basis(np.arange(12., dtype=float))
    return basis


def make_aux(path):
    frequencies = np.linspace(.001, .006, 6)
    latitudes = np.linspace(38., 46., 9)
    longitudes = np.linspace(-2., 7., 10)
    shape = (frequencies.size, latitudes.size, longitudes.size)
    xr.Dataset(
        {'Std': (('f', 'lat', 'lon'), np.full(shape, .01)),
         'Tdec': (('f', 'lat', 'lon'), np.full(shape, 5.))},
        coords={'f': frequencies, 'lat': latitudes, 'lon': longitudes},
    ).to_netcdf(path)


def timed(operation):
    jax.block_until_ready(operation())
    samples = []
    for _ in range(30):
        start = time.perf_counter()
        jax.block_until_ready(operation())
        samples.append((time.perf_counter() - start) * 1000.)
    return statistics.median(samples)


def old_temporal_representation(basis, times):
    """Recreate the former BMaux CSR time matrices and identity selector."""
    matrices = []
    for band in range(basis.nf):
        width = basis.iff_wavebounds[band + 1] - basis.iff_wavebounds[band]
        dense = np.zeros((len(times), width))
        offset = 0
        for time_index in range(basis.enst[band].shape[1]):
            for point in range(basis.NP[band]):
                centre = basis.enst[band][point, time_index]
                if np.isfinite(centre):
                    for row, t in enumerate(times):
                        dt = t - centre
                        if abs(dt) <= basis.tdec[band][point]:
                            dense[row, offset:offset + 2 * basis.ntheta] = (
                                mywindow(dt / basis.tdec[band][point])
                                / basis.norm_fact[band][point])
                offset += 2 * basis.ntheta
        matrices.append(jsparse.csr_fromdense(jnp.asarray(dense).T))
    selector = jnp.eye(len(times))
    jax.block_until_ready((matrices, selector))
    return matrices, selector


def main():
    with tempfile.TemporaryDirectory() as directory:
        aux_path = os.path.join(directory, 'bmaux.nc')
        make_aux(aux_path)
        basis = make_basis(aux_path)

        if os.environ.get('BASIS_BENCH_TIME_MEMORY'):
            legacy_time = bool(os.environ.get('BASIS_BENCH_LEGACY_TIME'))
            if legacy_time:
                basis.enst_device = None
                gc.collect()
            before = jax.devices()[0].memory_stats() or {}
            matrices, selector = (old_temporal_representation(
                basis, np.arange(12., dtype=float)) if legacy_time
                else ([], None))
            after = jax.devices()[0].memory_stats() or {}
            representation_bytes = sum(
                matrix.data.nbytes + matrix.indices.nbytes
                + matrix.indptr.nbytes for matrix in matrices)
            if selector is not None:
                representation_bytes += selector.nbytes
            print(json.dumps({
                'device': str(jax.devices()[0]),
                'temporal_representation': 'old_csr_and_eye' if legacy_time
                else 'direct_weights',
                'nphys': basis.nphys, 'nbasis': basis.nbasis,
                'ntime': 12,
                'before_bytes_in_use': before.get('bytes_in_use'),
                'after_bytes_in_use': after.get('bytes_in_use'),
                'delta_bytes_in_use': after.get('bytes_in_use', 0)
                - before.get('bytes_in_use', 0),
                'peak_bytes_in_use': after.get('peak_bytes_in_use'),
                'temporal_array_payload_bytes': representation_bytes,
            }, indent=2))
            return

        control = jnp.ones(basis.nbasis, dtype=jnp.float32)
        adjoint = jnp.ones(basis.shape_phys, dtype=jnp.float32)
        legacy = bool(os.environ.get('BASIS_BENCH_LEGACY'))

        def old_transpose():
            ad_state = {'ssh': adjoint}
            _ensure_adstate_names(basis, ad_state, jnp.zeros(basis.shape_phys))
            ad_field = _sum_adstate_names(basis, ad_state)
            result = basis._operg_reduced_jit(5., ad_field)
            if not basis.multi_mode:
                _clear_adstate_names(basis, ad_state)
            return result, ad_state['ssh']

        def new_transpose():
            ad_state = {'ssh': adjoint}
            result = basis.operg_transpose(5., ad_state)
            return result, ad_state['ssh']

        forward = lambda: basis.operg(5., control)
        transpose = old_transpose if legacy else new_transpose
        forward_ms = timed(forward)
        transpose_ms = timed(transpose)
        forward_kernel_ms = timed(lambda: basis._operg_jit(5., control))
        transpose_kernel_ms = timed(
            lambda: basis._operg_reduced_jit(5., adjoint))
        stats = jax.devices()[0].memory_stats() or {}
        result, cleared = transpose()
        lhs = jnp.vdot(forward(), adjoint)
        rhs = jnp.vdot(control, result)
        print(json.dumps({
            'device': str(jax.devices()[0]),
            'implementation': 'legacy_wrapper' if legacy else 'current',
            'nphys': basis.nphys, 'nbasis': basis.nbasis,
            'spatial_nnz': sum(int(matrix.nse) for matrix in basis.Gx),
            'forward_median_ms': forward_ms,
            'transpose_median_ms': transpose_ms,
            'forward_kernel_median_ms': forward_kernel_ms,
            'transpose_kernel_median_ms': transpose_kernel_ms,
            'bytes_in_use': stats.get('bytes_in_use'),
            'peak_bytes_in_use': stats.get('peak_bytes_in_use'),
            'transpose_norm': float(jnp.linalg.norm(result)),
            'adjoint_relative_error': float(
                jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(lhs), 1.e-12)),
            'cleared_max_abs': float(jnp.max(jnp.abs(cleared))),
        }, indent=2))


if __name__ == '__main__':
    main()
