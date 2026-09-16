from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from src.basis import Basis_gauss2d


def _make_basis(time_spinup):
    config = SimpleNamespace(BASIS=SimpleNamespace(
        facns=2.,
        sigma_D=300.,
        sigma_Q=0.01,
        facQ=1.,
        time_spinup=time_spinup,
        name_mod_var='parameter',
        flag_variable_Q=False,
        path_sad=None,
        name_var_sad={'lon': '', 'lat': '', 'var': ''},
        path_background=None,
        var_background=None,
        c_grid_var=None,
        compute_velocities=False,
        name_mod_u='u',
        name_mod_v='v',
    ))
    lon, lat = np.meshgrid(np.linspace(0., 1., 3), np.linspace(40., 41., 2))
    state = SimpleNamespace(
        ny=2,
        nx=3,
        lon=lon,
        lat=lat,
        lon_min=lon.min(),
        lon_max=lon.max(),
        lat_min=lat.min(),
        lat_max=lat.max(),
        f=np.ones_like(lon),
        DX=np.ones_like(lon),
        DY=np.ones_like(lon),
        mask=None,
        lon_unit='0_360',
    )
    basis = Basis_gauss2d(config, state)
    basis.set_basis(np.asarray([0., 1., 2.]))
    return basis


def test_gauss2d_time_spinup_smoothly_ramps_forward_projection():
    basis = _make_basis(time_spinup=2.)
    control = jnp.ones(basis.nbasis)

    at_start = basis.operg(0., control)
    halfway = basis.operg(1., control)
    after_spinup = basis.operg(2., control)

    np.testing.assert_allclose(at_start, 0., atol=1.e-7)
    np.testing.assert_allclose(halfway, 0.5 * after_spinup, rtol=1.e-6)


def test_gauss2d_spinup_forward_and_transpose_remain_adjoint():
    basis = _make_basis(time_spinup=2.)
    control = jnp.arange(basis.nbasis, dtype=float) + 1.
    physical_adjoint = jnp.arange(basis.nphys, dtype=float).reshape(basis.shape_phys)

    physical = basis.operg(0.7, control)
    reduced_adjoint = basis.operg_transpose(
        0.7, {'parameter': physical_adjoint.copy()})

    np.testing.assert_allclose(
        jnp.vdot(physical, physical_adjoint),
        jnp.vdot(control, reduced_adjoint),
        rtol=1.e-6,
        atol=1.e-6,
    )


def test_gauss2d_without_spinup_is_time_independent():
    basis = _make_basis(time_spinup=None)
    control = jnp.ones(basis.nbasis)

    np.testing.assert_allclose(
        basis.operg(0., control), basis.operg(10., control), rtol=1.e-6)
