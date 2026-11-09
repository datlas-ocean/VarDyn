"""Regression coverage for the three issues found in the Dev-SW review."""
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from src import basis as bases
from src import config_default as defaults
from src.exp import Config
from test_sw_review_changes import make_model


def write_calibration(state, directory, reference):
    geometry = bases._sw_forcing_geometry(state)
    matrix, records, shapes = bases._build_sw_forcing_atoms(
        geometry, np.array([200.]),
        dict(momentum=np.array([90.]), height=np.array([70.])))
    path = Path(directory)
    path.mkdir(exist_ok=True)
    bases.scipy.sparse.save_npz(path/'atoms.npz', matrix)
    count = len(records)
    variables = dict(component=(('atom',), records[:, 0]),
        slow_std=(('atom',), np.full(count, .01)), wave_std=(('atom',), np.zeros(count)),
        slow_halfwidth_days=(('atom',), np.full(count, 4.)),
        wave_halfwidth_days=(('atom',), np.full(count, 8.)),
        wave_period_days=(('atom',), np.zeros(count)),
        bias_u=(('y', 'xu'), np.zeros(shapes[0])),
        bias_v=(('yv', 'x'), np.zeros(shapes[1])), bias_h=(('y', 'x'), np.zeros(shapes[2])))
    xr.Dataset(variables, attrs=dict(schema='sw-local-atoms-v1', reference_g_prime=reference,
        grid_hash=bases._sw_forcing_grid_hash(geometry), matrix_file='atoms.npz',
        matrix_sha256=hashlib.sha256((path/'atoms.npz').read_bytes()).hexdigest())).to_netcdf(path/'calibration.nc')
    return path/'calibration.nc'


class SWReviewFixTests(unittest.TestCase):
    def test_wind_depth_uses_joint_H_gravity_control_in_both_forward_paths(self):
        with TemporaryDirectory() as directory:
            model, state, _ = make_model(directory, name_params=['H', 'g_prime'],
                                          phase_speed_squared_floor=1.)
            nx, ny = model.model.nx, model.model.ny
            u = jnp.zeros((1, 1, nx+1, ny), dtype=model.dtype)
            v = jnp.zeros((1, 1, nx, ny+1), dtype=model.dtype)
            h = jnp.zeros((1, 1, nx, ny), dtype=model.dtype)
            H = jnp.zeros((1, nx, ny), dtype=model.dtype)
            gp = jnp.ones_like(H)*np.log(2.)
            Fu, Fv, Fh = jnp.zeros((ny, nx+1)), jnp.zeros((ny+1, nx)), jnp.zeros((ny, nx))
            captured = []
            def step(uu, vv, hh, **kwargs):
                captured.append(kwargs)
                return uu, vv, hh
            with patch.object(model, 'model_step', step):
                model.jstep_core(0., u, v, h, H, gp, Fu, Fv, Fh, Fu, Fv, Fh)
            def step_tracer(uu, vv, hh, cc, **kwargs):
                captured.append(kwargs)
                return uu, vv, hh, cc
            tracer = jnp.zeros((1, ny, nx))
            with patch.object(model.model, 'step_with_tracer', step_tracer):
                model.jstep_core_trac(0., u, v, h, tracer, H, gp, Fu, Fv, Fh, tracer,
                                     Fu, Fv, Fh, tracer)
            for arguments in captured:
                depth = model.model.H + arguments['H']
                np.testing.assert_allclose(depth, 50., rtol=2e-6)
                np.testing.assert_allclose(arguments['h_wind'], depth[0], rtol=2e-6)
            derivative = jax.grad(lambda alpha: jnp.mean(
                model._wind_depth_to_model(H, None, jnp.ones_like(H)*alpha)))(jnp.log(2.))
            np.testing.assert_allclose(derivative, -50., rtol=2e-6)

    def test_calibration_uses_resolved_diagnosed_and_mapped_gravity(self):
        with TemporaryDirectory() as directory:
            maps = Path(directory)/'maps.nc'
            lon, lat = np.linspace(-1., 1., 15), np.linspace(-1., 1., 13)
            xr.Dataset({'gp': (('lat', 'lon'), np.full((len(lat), len(lon)), .04))},
                       coords={'lon':lon, 'lat':lat}).to_netcdf(maps)
            mapped = dict(file_params_aux=str(maps),
                          name_var_params_aux={'lon':'lon', 'lat':'lat', 'g_prime':'gp'})
            for index, options in enumerate((dict(g_prime=None, c0=2.),
                                             dict(g_prime=None, **mapped),
                                             dict(g_prime=.06, **mapped))):
                model, state, config = make_model(directory, **options)
                calibration = write_calibration(state, Path(directory)/str(index), .04)
                config.BASIS = Config(deepcopy(defaults.BASIS_SW_FORCING) | dict(
                    super='BASIS_SW_FORCING', calibration_file=str(calibration)))
                # Cloned states retain the reference used by observation operators.
                copied = state.copy()
                np.testing.assert_allclose(copied.qgsw_reference_g_prime, .04, rtol=2e-6)
                basis = bases.Basis(config, copied, verbose=False)
                self.assertIsInstance(basis, bases.CalibratedSWForcing)
                with xr.open_dataset(calibration) as source:
                    changed = source.load()
                changed.attrs['reference_g_prime'] = .07
                changed.to_netcdf(calibration)
                with self.assertRaisesRegex(ValueError, 'different reference g_prime'):
                    bases.Basis(config, copied, verbose=False)

    def test_tracer_inflow_and_outflow_on_all_faces_and_schemes(self):
        with TemporaryDirectory() as directory:
            model, _, _ = make_model(directory, open_boundary_x='characteristic',
                                      open_boundary_y='characteristic')
            core = model.model
            nx, ny = core.nx, core.ny
            zero = jnp.zeros((1, 2, nx, ny), dtype=core.dtype)
            exterior = jnp.ones_like(zero).at[:, 0].set(2.).at[:, 1].set(5.)
            rate = 1e-5
            for scheme in ('rusanov1', 'linear_upwind3', 'linear_upwind5', 'weno'):
                core.tracer_adv_scheme = scheme
                for axis, edge, incoming_sign in ((0, 0, 1), (0, -1, -1),
                                                   (1, 0, 1), (1, -1, -1)):
                    u = jnp.zeros((1, 1, nx+1, ny))
                    v = jnp.zeros((1, 1, nx, ny+1))
                    if axis == 0:
                        u = u.at[..., edge, :].set(incoming_sign*rate)
                    else:
                        v = v.at[..., :, edge].set(incoming_sign*rate)
                    tendency = core.advection_tracer(u, v, zero, exterior*core.area)/core.area
                    expected = jnp.zeros_like(zero)
                    if axis == 0:
                        expected = expected.at[..., edge, :].set(rate*exterior[..., edge, :])
                    else:
                        expected = expected.at[..., :, edge].set(rate*exterior[..., :, edge])
                    np.testing.assert_allclose(tendency, expected, rtol=2e-6, atol=1e-10)
                    # Exterior data cannot inject tracer against an outgoing velocity.
                    outflow = core.advection_tracer(-u, -v, zero, exterior*core.area)
                    np.testing.assert_allclose(outflow, 0., atol=1e-10)
                    # Missing exterior data retains the original extrapolation behavior.
                    np.testing.assert_allclose(core.advection_tracer(u, v, zero), 0., atol=1e-10)

    def test_prescribed_tracer_enters_without_sponge_and_has_boundary_gradient(self):
        with TemporaryDirectory() as directory:
            model, _, _ = make_model(directory, open_boundary_y='characteristic')
            core = model.model
            nx, ny = core.nx, core.ny
            u = jnp.zeros((1, 1, nx+1, ny), dtype=core.dtype)
            v = jnp.ones((1, 1, nx, ny+1), dtype=core.dtype)*.02
            h = jnp.zeros((1, 1, nx, ny), dtype=core.dtype)
            def advance(boundary):
                return core.step_with_tracer(u, v, h, jnp.zeros_like(h),
                    nstep=1, u_b=u, v_b=v, h_b=h, c_b=jnp.ones_like(h)*boundary)[3]
            result = jax.jit(advance)(7.)
            self.assertGreater(float(jnp.sum(result[..., 0])), 0.)
            self.assertTrue(bool(jnp.isfinite(result).all()))
            sensitivity = jax.jit(jax.grad(lambda boundary: jnp.sum(advance(boundary))))(7.)
            self.assertGreater(float(sensitivity), 0.)


if __name__ == '__main__':
    unittest.main()
