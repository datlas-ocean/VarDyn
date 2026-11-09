"""Regression checks for open meridional boundaries and physical controls."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from src import config_default as defaults
from src.exp import Config
from src.mod import Model_qgsw
from src.state import State
from src.inv import Inv_4Dvar
from src import basis as bases
import hashlib


def make_model(directory, **options):
    config = Config()
    config.EXP = Config(defaults.EXP | dict(path_save=directory, tmp_DA_path=directory,
        flag_plot=0, init_date=datetime(2024, 1, 1), final_date=datetime(2024, 1, 1, 1)))
    config.GRID = Config(defaults.GRID_GEO | dict(super='GRID_GEO', lon_min=-.5, lon_max=.5,
        lat_min=-.4, lat_max=.4, dlon=.1, dlat=.1, name_init_mask=None))
    config.MOD = Config(defaults.MOD_QGSW | dict(super='MOD_QGSW', H=100., g_prime=.04,
        dtmodel=60., cfl=None, height_representation='interface_displacement',
        dist_sponge_bc=0., h_adv_scheme='linear_upwind3', mom_adv_scheme='upwind3') | options)
    state = State(config, verbose=False)
    model = Model_qgsw(config, state)
    return model, state, config


class SWReviewTests(unittest.TestCase):
    def test_y_characteristics_signs_masks_and_adjoint(self):
        with TemporaryDirectory() as directory:
            model, _, _ = make_model(directory, open_boundary_y='characteristic')
            core = model.model
            nx, ny = core.nx, core.ny
            v = jnp.zeros((1, 1, nx, ny+1), dtype=core.dtype)
            h = jnp.ones((1, 1, nx, ny), dtype=core.dtype) * .5 * core.area
            result = core._apply_y_characteristic(v, h, v, jnp.zeros_like(h), core.H)
            physical = np.asarray(result / core.dy_vgrid)
            expected = float(jnp.sqrt(jnp.asarray(core.g_prime).ravel()[0] / 100.) * .5)
            np.testing.assert_allclose(physical[..., 0], -expected, rtol=2e-6)
            np.testing.assert_allclose(physical[..., -1], expected, rtol=2e-6)
            np.testing.assert_array_equal(core.masks.v[..., 0], True)
            np.testing.assert_array_equal(core.masks.v[..., -1], True)
            np.testing.assert_array_equal(core.masks.u[..., 0, :], False)
            transform = lambda height: core._apply_y_characteristic(v, height, v, jnp.zeros_like(h), core.H)
            direction = jnp.ones_like(h)
            weight = jnp.ones_like(v).at[..., 0].set(2.)
            _, tangent = jax.jvp(transform, (h,), (direction,))
            _, transpose = jax.vjp(transform, h)
            np.testing.assert_allclose(jnp.vdot(tangent, weight),
                                       jnp.vdot(direction, transpose(weight)[0]), rtol=2e-6)
            land = jnp.ones((nx, ny)).at[2, 0].set(0.).at[3, 2].set(0.)
            masks = type(core.masks)(land, open_y=True)
            self.assertFalse(bool(masks.v[0, 0, 2, 0]))
            self.assertFalse(bool(masks.v[0, 0, 3, 2]))
            self.assertFalse(bool(masks.v[0, 0, 3, 3]))

    def test_boundary_flux_budget_for_x_y_and_both(self):
        with TemporaryDirectory() as directory:
            model, _, _ = make_model(directory)
            core = model.model
            nx, ny = core.nx, core.ny
            q = jnp.ones((1, 1, nx, ny))
            u = jnp.ones((1, 1, nx+1, ny)) * .2
            v = jnp.ones((1, 1, nx, ny+1)) * .3
            rng = np.random.default_rng(9)
            fx = jnp.asarray(rng.normal(size=(1, 1, nx-1, ny)))
            fy = jnp.asarray(rng.normal(size=(1, 1, nx, ny-1)))
            for open_x, open_y in ((False, False), (True, False), (False, True), (True, True)):
                core.open_boundary_x = 'characteristic' if open_x else 'closed'
                core.open_boundary_y = 'characteristic' if open_y else 'closed'
                core.masks = type(core.masks)(jnp.ones((nx, ny)), open_x=open_x, open_y=open_y)
                result = core._divergence_with_open_boundaries(fx, fy, u, v, q)
                # Unequal prescribed inflow/outflow gives a known total budget.
                uu = u.at[..., -1, :].set(.4)
                vv = v.at[..., :, -1].set(.6)
                result = core._divergence_with_open_boundaries(fx, fy, uu, vv, q)
                area = np.broadcast_to(np.asarray(core.area), (nx, ny))
                expected = (float(np.sum(.4*area[-1] - .2*area[0])) if open_x else 0.)
                expected += float(np.sum(.6*area[:, -1] - .3*area[:, 0])) if open_y else 0.
                np.testing.assert_allclose(np.sum(result), expected, rtol=2e-6, atol=2.)

    def test_g_prime_map_defines_closure_and_observation_units(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'parameters.nc'
            lon, lat = np.linspace(-1., 1., 15), np.linspace(-1., 1., 13)
            field = .035 + .001 * np.broadcast_to(lon, (lat.size, lon.size))
            xr.Dataset({'gp': (('lat', 'lon'), field)}, coords={'lon': lon, 'lat': lat}).to_netcdf(path)
            model, state, _ = make_model(directory, g_prime=None, file_params_aux=str(path),
                name_var_params_aux={'lon':'lon', 'lat':'lat', 'g_prime':'gp'})
            expected = .035 + .001 * state.lon
            np.testing.assert_allclose(np.broadcast_to(model.g_prime[0].T, expected.shape), expected, rtol=2e-5)
            np.testing.assert_allclose(model.c**2, 100.*expected, rtol=2e-5)
            np.testing.assert_allclose(model.ssh_observation_scale, expected/9.81, rtol=2e-5)

    def test_log_controls_preserve_depth_bounds_and_reference_values(self):
        model = Model_qgsw.__new__(Model_qgsw)
        model.dtype = jnp.float32
        model.H0 = jnp.array([[[100.]]]); model.H_floor = jnp.array([[[20.]]])
        model.H_max_bound = 150.; model.phase_speed_squared_floor = None
        model.h_wind_ref_sw = jnp.array([[80.]])
        model.h_wind_floor_sw = jnp.array([[10.]]); model.h_wind_max_bound_sw = 120.
        model.diff_coef_ref = 200.; model.g_prime = jnp.array([[[.04]]])
        for value in (-2., 0., 1.):
            control = jnp.array([[value]])
            np.testing.assert_allclose(model._H_control_to_total(control), min(150., 20.+80.*np.exp(value)), rtol=2e-6)
            np.testing.assert_allclose(model._h_wind_control_to_total_sw(control), min(120., 10.+70.*np.exp(value)), rtol=2e-6)
            np.testing.assert_allclose(model._coefficient_control_to_total_sw('diff_coef', control), 200.*np.exp(value), rtol=2e-6)
            np.testing.assert_allclose(model._g_prime_control_to_total(control), .04*np.exp(value), rtol=2e-6)

    def test_open_y_and_xy_steps_preserve_constant_passive_tracer(self):
        with TemporaryDirectory() as directory:
            for x_mode in ('closed', 'characteristic'):
                model, _, _ = make_model(directory, open_boundary_x=x_mode,
                                          open_boundary_y='characteristic')
                core = model.model
                u = jnp.zeros((1, 1, core.nx+1, core.ny), dtype=core.dtype)
                v = jnp.zeros((1, 1, core.nx, core.ny+1), dtype=core.dtype)
                h = jnp.ones((1, 1, core.nx, core.ny), dtype=core.dtype) * .01
                plain = core.step(u, v, h, nstep=2)
                coupled = core.step_with_tracer(u, v, h, jnp.ones_like(h)*7., nstep=2)
                for field, traced in zip(plain, coupled[:3]):
                    self.assertTrue(bool(jnp.isfinite(field).all()))
                    np.testing.assert_allclose(traced, field, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(coupled[3], 7., rtol=2e-6, atol=2e-6)

    def test_calibrated_basis_preserves_temporal_variance_and_adjoint(self):
        from copy import deepcopy
        from test_daily_sw_basis import small_grid_and_config
        with TemporaryDirectory() as directory:
            state, _ = small_grid_and_config()
            geometry = bases._sw_forcing_geometry(state)
            matrix, records, shapes = bases._build_sw_forcing_atoms(
                geometry, np.array([200.]),
                dict(momentum=np.array([90.]), height=np.array([70.])))
            state.params = dict(zip(('u', 'v', 'ssh'), (jnp.zeros(shape) for shape in shapes)))
            path = Path(directory)
            bases.scipy.sparse.save_npz(path/'atoms.npz', matrix)
            count = len(records)
            variables = dict(component=(('atom',), records[:, 0]),
                slow_std=(('atom',), np.full(count, .01)), wave_std=(('atom',), np.zeros(count)),
                slow_halfwidth_days=(('atom',), np.linspace(2., 8., count)),
                wave_halfwidth_days=(('atom',), np.full(count, 8.)),
                wave_period_days=(('atom',), np.zeros(count)),
                bias_u=(('y', 'xu'), np.zeros(shapes[0])),
                bias_v=(('yv', 'x'), np.zeros(shapes[1])), bias_h=(('y', 'x'), np.zeros(shapes[2])))
            xr.Dataset(variables, attrs=dict(schema='sw-local-atoms-v1',
                grid_hash=bases._sw_forcing_grid_hash(geometry), matrix_file='atoms.npz',
                matrix_sha256=hashlib.sha256((path/'atoms.npz').read_bytes()).hexdigest())).to_netcdf(path/'calibration.nc')
            config = Config(BASIS=Config(deepcopy(defaults.BASIS_SW_FORCING) | dict(
                super='BASIS_SW_FORCING', calibration_file=str(path/'calibration.nc'))))
            basis = bases.Basis(config, state, verbose=False)
            _, std = basis.set_basis(np.arange(0., 12.01, .5), return_q=True)
            control = jnp.asarray(np.random.default_rng(12).normal(size=basis.nbasis))
            weights = {name: jnp.ones(shape) for name, shape in zip(basis.name_mod_var, shapes)}
            for time in (0., 3.75, 12.):
                variance = np.bincount(np.asarray(basis.atom),
                    weights=(std*np.asarray(basis._temporal(time)))**2, minlength=count)
                np.testing.assert_allclose(variance, .01**2, rtol=2e-5)
                expected = sum(jnp.vdot(field, weights[name]) for name, field in zip(
                    basis.name_mod_var, basis.project(time, control)))
                np.testing.assert_allclose(jnp.vdot(control, basis.operg_transpose(time, dict(weights))),
                                           expected, rtol=2e-5, atol=1e-5)

    def test_removed_block_minimizer_is_rejected(self):
        config = Config(INV=Config(minimizer='block-coordinate'))
        with self.assertRaisesRegex(ValueError, 'Unknown 4DVar minimizer'):
            Inv_4Dvar(config=config)


if __name__ == '__main__':
    unittest.main()
