"""Numerical checks and an optional GPU benchmark for reduced bases.

Run the benchmark with ``python mapping/tests/test_basis_sparse_temporal.py``.
It reports JAX allocator statistics for the selected CUDA device.
"""

import json
import os
import time
import gc
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from src.basis import (Basis_gauss3d, Basis_bm, Basis_bmaux, Basis_hbc,
                       _clear_adstate_names, _ensure_adstate_names,
                       _sum_adstate_names, mywindow)


def _basis(shape=(20, 30), ntime=12):
    ny, nx = shape
    lon, lat = np.meshgrid(np.linspace(0., 5., nx),
                           np.linspace(40., 44., ny))
    state = SimpleNamespace(
        ny=ny, nx=nx, lon=lon, lat=lat,
        lon_min=lon.min(), lon_max=lon.max(),
        lat_min=lat.min(), lat_max=lat.max(),
        f=np.full_like(lon, 1.e-4),
        DX=np.full_like(lon, 1.e4), DY=np.full_like(lon, 1.e4),
        mask=None, lon_unit='0_360',
    )
    config = SimpleNamespace(BASIS=SimpleNamespace(
        flux=False, facns=2., facnlt=2., sigma_D=120., sigma_T=3.,
        sigma_Q=.01, facQ=1., file_facQaux=None, name_var_facQaux=None,
        normalize_fact=False, name_mod_var='ssh', time_spinup=None,
        flag_variable_Q=False, path_sad=None, name_var_sad=None,
        path_background=None, var_background=None, c_grid_var=None,
        compute_velocities=False, name_mod_u='u', name_mod_v='v',
    ))
    basis = Basis_gauss3d(config, state)
    basis.set_basis(np.arange(ntime, dtype=float))
    return basis


def _reference_weights(basis, t):
    dt = t - basis.ENST
    weights = np.where(np.abs(dt) < basis.sigma_T,
                       mywindow(dt / basis.sigma_T), 0.)
    return np.repeat(weights, basis.Nx)


def test_gauss3d_sparse_space_and_direct_time_match_reference():
    basis = _basis()
    rng = np.random.default_rng(2)
    control = rng.normal(size=basis.nbasis).astype(np.float32)
    adjoint = rng.normal(size=basis.shape_phys).astype(np.float32)
    spatial = np.asarray(basis.Gauss_xy.todense())
    for t in (0., 1.5, 5., 11.):
        weights = _reference_weights(basis, t)
        reduced_space = (weights * control).reshape(-1, basis.Nx).sum(axis=0)
        expected = (spatial @ reduced_space).reshape(basis.shape_phys)
        actual = basis.operg(t, jnp.asarray(control))
        np.testing.assert_allclose(actual, expected, rtol=2.e-5, atol=2.e-5)

        expected_adjoint = (weights.reshape(-1, basis.Nx)
                            * (spatial.T @ adjoint.ravel())[None, :]).ravel()
        actual_adjoint = basis.operg_transpose(
            t, {'ssh': jnp.asarray(adjoint)})
        np.testing.assert_allclose(actual_adjoint, expected_adjoint,
                                   rtol=2.e-5, atol=2.e-5)
        np.testing.assert_allclose(
            np.vdot(actual, adjoint), np.vdot(control, actual_adjoint),
            rtol=2.e-5, atol=2.e-5)


def test_gauss3d_adjoint_state_is_cleared_and_zero_is_reused():
    basis = _basis()
    field = jnp.ones(basis.shape_phys)
    state = {'ssh': field}
    expected = basis.operg_transpose(1., state)
    first_zero = state['ssh']
    np.testing.assert_allclose(first_zero, 0.)

    state['ssh'] = field
    np.testing.assert_allclose(basis.operg_transpose(1., state), expected)
    assert state['ssh'] is first_zero

    state['ssh'] = None
    np.testing.assert_allclose(basis.operg_transpose(1., state), 0.)
    np.testing.assert_allclose(state['ssh'], 0.)

    @jax.jit
    def traced_transpose(adjoint):
        traced_state = {'ssh': adjoint}
        result = basis.operg_transpose(1., traced_state)
        return result, traced_state['ssh']

    traced_result, traced_zero = traced_transpose(field)
    np.testing.assert_allclose(traced_result, expected)
    np.testing.assert_allclose(traced_zero, 0.)


def test_bm_temporal_weights_match_reference():
    basis = Basis_bm.__new__(Basis_bm)
    basis.enst_device = [jnp.asarray([-1., 1., 3.])]
    basis.tdec = [2.]
    basis.norm_fact = [1.3]
    basis.norm_time = True
    basis.flux = False
    basis.Nx = [4]
    expected = np.repeat(
        np.where(np.abs(1.5 - np.array([-1., 1., 3.])) <= 2.,
                 mywindow((1.5 - np.array([-1., 1., 3.])) / 2.), 0.) / 1.3,
        4)
    np.testing.assert_allclose(basis.get_Gt_value(1.5, 0), expected,
                               rtol=1.e-6)


def test_bmaux_temporal_weights_match_reference():
    basis = Basis_bmaux.__new__(Basis_bmaux)
    centres = np.array([[0., 2., np.nan], [1., 3., 5.]])
    basis.enst_device = [jnp.asarray(centres)]
    basis.tdec = [np.array([1.5, 2.])]
    basis.norm_fact = [np.array([1.2, 1.4])]
    basis.ntheta = 2
    basis.flux = False
    expected = []
    for it in range(centres.shape[1]):
        for p in range(centres.shape[0]):
            dt = 2.5 - centres[p, it]
            weight = (mywindow(dt / basis.tdec[0][p])
                      / basis.norm_fact[0][p]
                      if np.isfinite(dt) and abs(dt) <= basis.tdec[0][p]
                      else 0.)
            expected.extend([weight] * (2 * basis.ntheta))
    np.testing.assert_allclose(basis.get_Gt_value(2.5, 0), expected,
                               rtol=1.e-6)


def test_hbc_temporal_weights_match_reference():
    basis = Basis_hbc.__new__(Basis_hbc)
    basis.enst_bc_device = jnp.asarray([-2., 0., 2., 4.])
    basis.T_bc = 2.
    t = 1.
    dt = t - np.array([-2., 0., 2., 4.])
    expected = np.where(np.abs(dt) < basis.T_bc,
                        mywindow(dt / basis.T_bc), 0.)
    np.testing.assert_allclose(basis.get_bc_t_gauss_value(t), expected,
                               rtol=1.e-6)


def test_hbc_adjoint_reuses_device_zero():
    basis = Basis_hbc.__new__(Basis_hbc)
    basis.name_params = ['hbcx']
    basis.nphys = 6
    basis.slice_params_phys = {'hbcS': slice(0, 3), 'hbcN': slice(3, 6)}
    basis._operg_reduced_jit = lambda t, values: jnp.asarray(values)
    field = jnp.arange(6., dtype=jnp.float32).reshape(1, 2, 1, 1, 3)
    state = {'hbcx': field}
    np.testing.assert_allclose(basis.operg_transpose(0., state), np.arange(6.))
    zero = state['hbcx']
    np.testing.assert_allclose(zero, 0.)
    state['hbcx'] = field
    basis.operg_transpose(0., state)
    assert state['hbcx'] is zero


def test_missing_adjoint_zero_is_constructed_only_when_needed():
    target = SimpleNamespace(name_mod_var=['a', 'b'])
    calls = []

    def make_zero():
        calls.append(1)
        return np.zeros(3)

    state = {'a': np.ones(3), 'b': np.ones(3)}
    _ensure_adstate_names(target, state, make_zero)
    assert not calls
    state['b'] = None
    _ensure_adstate_names(target, state, make_zero)
    assert len(calls) == 1
    np.testing.assert_allclose(state['b'], 0.)


def _benchmark():
    device = jax.devices()[0]
    basis = _basis(shape=(70, 90), ntime=32)
    control = jnp.ones(basis.nbasis, dtype=jnp.float32)
    adjoint = jnp.ones(basis.shape_phys, dtype=jnp.float32)
    legacy = bool(os.environ.get('BASIS_BENCH_LEGACY'))
    if legacy:
        # Recreate the previous dense-to-CSR construction and one-hot time
        # selection. Run this mode in a separate process for allocator peaks.
        dense = jnp.asarray(basis.Gauss_xy.todense())
        space = jsparse.CSR.fromdense(dense)
        space_t = jsparse.CSR.fromdense(dense.T)
        temporal = np.stack([_reference_weights(basis, t)
                             for t in range(32)], axis=1)
        gt = jsparse.CSR.fromdense(jnp.asarray(temporal))
        eye = jnp.eye(32)
        basis.Gauss_xy = basis.Gauss_xy_T = None
        gc.collect()

        @jax.jit
        def forward(t, x):
            index = jnp.where(jnp.arange(32) == t, size=1)[0][0]
            weights = gt @ eye[index]
            return (space @ (weights * x).reshape(-1, basis.Nx).sum(0))

        @jax.jit
        def transpose(t, y):
            index = jnp.where(jnp.arange(32) == t, size=1)[0][0]
            weights = gt @ eye[index]
            return (weights.reshape(-1, basis.Nx)
                    * (space_t @ y.ravel())[None, :]).ravel()

        def legacy_public_transpose():
            ad_state = {'ssh': adjoint}
            _ensure_adstate_names(basis, ad_state, jnp.zeros(basis.shape_phys))
            ad_params = _sum_adstate_names(basis, ad_state)
            result = transpose(5, ad_params)
            if not basis.multi_mode:
                _clear_adstate_names(basis, ad_state)
            return result

        forward_op = lambda: forward(5, control)
        transpose_op = legacy_public_transpose
        kernel_forward_op = forward_op
        kernel_transpose_op = lambda: transpose(5, adjoint)
    else:
        forward_op = lambda: basis.operg(5., control)
        transpose_op = lambda: basis.operg_transpose(
            5., {'ssh': adjoint})
        kernel_forward_op = lambda: basis._operg_jit(5., control)
        kernel_transpose_op = lambda: basis._operg_reduced_jit(5., adjoint)

    def timed(operation):
        operation().block_until_ready()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            operation().block_until_ready()
            samples.append((time.perf_counter() - start) * 1000.)
        return float(np.median(samples))

    forward_ms = timed(forward_op)
    transpose_ms = timed(transpose_op)
    kernel_forward_ms = timed(kernel_forward_op)
    kernel_transpose_ms = timed(kernel_transpose_op)
    stats = device.memory_stats() or {}
    # Evaluate precision only after recording the allocator peak: the dense
    # reference is deliberately outside the memory benchmark.
    if legacy:
        spatial_reference = np.asarray(dense)
    else:
        spatial_reference = np.asarray(basis.Gauss_xy.todense())
    weights = _reference_weights(basis, 5.)
    expected_forward = spatial_reference @ weights.reshape(-1, basis.Nx).sum(0)
    expected_transpose = (weights.reshape(-1, basis.Nx)
                          * (spatial_reference.T @ np.ones(basis.nphys))[None, :]).ravel()
    forward_error = float(np.max(np.abs(np.asarray(forward_op()).ravel() - expected_forward)))
    transpose_error = float(np.max(np.abs(np.asarray(transpose_op()) - expected_transpose)))
    report = {
        'device': str(device), 'implementation': 'legacy' if legacy else 'new',
        'nphys': basis.nphys, 'nbasis': basis.nbasis,
        'spatial_nnz': int(space.nse if legacy else basis.Gauss_xy.nse),
        'forward_median_ms': forward_ms,
        'transpose_median_ms': transpose_ms,
        'forward_kernel_median_ms': kernel_forward_ms,
        'transpose_kernel_median_ms': kernel_transpose_ms,
        'bytes_in_use': stats.get('bytes_in_use'),
        'peak_bytes_in_use': stats.get('peak_bytes_in_use'),
        'forward_max_abs_error': forward_error,
        'transpose_max_abs_error': transpose_error,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('Set CUDA_VISIBLE_DEVICES to a GPU index.')
    _benchmark()
