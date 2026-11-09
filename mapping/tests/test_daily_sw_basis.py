"""Daily SW forcing keeps its temporal prior and adjoint after relocation."""
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src import config_default as defaults
from src.basis import (
    Basis, DailyAR1Covariance, DailySWForcing,
    _build_sw_forcing_atoms, _sw_forcing_geometry,
)
from src.exp import Config


def small_grid_and_config(physical_rms=None):
    ny, nx = 7, 8
    x, y = np.meshgrid(np.arange(nx) * 25000., np.arange(ny) * 25000.)
    state = SimpleNamespace(nx=nx, ny=ny, X=x, Y=y,
                            DX=x * 0 + 25000, DY=y * 0 + 25000, mask=None)
    options = deepcopy(defaults.BASIS_SW_DAILY) | dict(
        super='BASIS_SW_DAILY', wavelengths_km=[200.],
        initial_physical_rms=physical_rms)
    config = Config(BASIS=Config(options), EXP=Config(init_date=datetime(2020, 1, 1)))
    return state, config


class DailySWBasisTests(unittest.TestCase):
    def test_ar1_covariance_matches_exponential_and_adjoint(self):
        operator = DailyAR1Covariance([2., .4], [3., 10.], 5)
        root = np.asarray(jax.vmap(operator.sqrt)(jnp.eye(10))).T
        covariance = root @ root.T
        times = np.arange(5)
        for k, (sigma, tau) in enumerate(((2., 3.), (.4, 10.))):
            np.testing.assert_allclose(
                covariance[k::2, k::2],
                sigma**2 * np.exp(-abs(times[:, None] - times) / tau), rtol=2e-6)
        np.testing.assert_array_equal(covariance[::2, 1::2], 0.)
        vector = jnp.arange(10, dtype=float) / 10
        np.testing.assert_allclose(operator.inverse_sqrt(operator.sqrt(vector)), vector, atol=5e-7)
        np.testing.assert_allclose(operator.sqrt_transpose(vector), root.T @ vector, atol=5e-7)
        np.testing.assert_allclose(operator.inverse(vector), np.linalg.solve(covariance, vector),
                                   rtol=2e-5, atol=2e-5)

    def test_factory_interpolation_and_projection_adjoint(self):
        state, config = small_grid_and_config()
        basis = Basis(config, state, verbose=False)
        self.assertIsInstance(basis, DailySWForcing)
        basis.set_basis([0., 3.])
        signature, size = basis.layout_signature, basis.nbasis
        rng = np.random.default_rng(7)
        control = jnp.asarray(rng.normal(size=size))
        np.testing.assert_allclose(
            basis.coefficients(1.25, control),
            .75 * np.asarray(control).reshape(4, -1)[1]
            + .25 * np.asarray(control).reshape(4, -1)[2], atol=1e-6)
        for time in (-1., 4.):
            np.testing.assert_array_equal(basis.coefficients(time, control), 0.)
        fields = dict(zip(basis.name_mod_var,
                          (jnp.asarray(rng.normal(size=shape)) for shape in basis.shapes)))
        left = sum(jnp.vdot(field, fields[name])
                   for field, name in zip(basis.project(1.25, control), basis.name_mod_var))
        right = jnp.vdot(control, basis.operg_transpose(1.25, fields))
        np.testing.assert_allclose(left, right, atol=2e-5)
        basis.update_statistics(basis.std * 2, basis.tdec * 3)
        self.assertEqual(basis.nbasis, size)
        self.assertEqual(basis.layout_signature, signature)

    def test_rotational_atoms_have_zero_flux_divergence(self):
        state, _ = small_grid_and_config()
        geometry = _sw_forcing_geometry(state)
        matrix, records, shapes = _build_sw_forcing_atoms(
            geometry, np.array([200.]),
            dict(momentum=np.array([90.]), height=np.array([70.])),
            normalization='potential')
        u_stop, v_stop = np.cumsum([np.prod(shape) for shape in shapes])[:2]
        selected = records[:, 0] == 1
        self.assertTrue(selected.any())
        for atom in np.flatnonzero(selected):
            vector = matrix[:, atom].toarray().ravel()
            u = vector[:u_stop].reshape(shapes[0]) * geometry['fy']
            v = vector[u_stop:v_stop].reshape(shapes[1]) * geometry['fx']
            divergence = np.diff(u, axis=1) + np.diff(v, axis=0)
            np.testing.assert_allclose(divergence, 0., atol=2e-6 * max(
                np.max(abs(u)), np.max(abs(v))))

    def test_compensated_height_atoms_preserve_each_disconnected_basin_mass(self):
        state, _ = small_grid_and_config()
        state.mask = np.zeros((state.ny, state.nx), dtype=bool)
        state.mask[:, 3:5] = True
        geometry = _sw_forcing_geometry(state)
        self.assertEqual(geometry['nbasin'], 2)
        matrix, records, shapes = _build_sw_forcing_atoms(
            geometry, np.array([200.]),
            dict(momentum=np.array([90.]), height=np.array([70.])),
            normalization='potential')
        height_start = sum(np.prod(shape) for shape in shapes[:2])
        self.assertTrue((records[:, 0] == 2).any())
        self.assertEqual(np.count_nonzero(records[:, 0] == 3), 2)
        for atom in np.flatnonzero(records[:, 0] == 2):
            height = matrix[height_start:, atom].toarray().reshape(shapes[2])
            for basin in (1, 2):
                mass = height * geometry['area'] * (geometry['basins'] == basin)
                self.assertLessEqual(abs(mass.sum()), 2e-6 * np.sum(abs(mass)) + 1e-8)

    def test_initial_physical_rms_includes_masked_atom_response(self):
        targets = dict(divergent=.0304, rotational=.0198, height=13.5, height_mean=1.23)
        state, config = small_grid_and_config(targets)
        basis = Basis(config, state, verbose=False)
        geometry = basis.g
        weights = np.concatenate(((geometry['au'] * geometry['wu']).ravel(),
                                  (geometry['av'] * geometry['wv']).ravel(),
                                  (geometry['area'] * geometry['wet']).ravel()))
        for component, name in enumerate(basis.components):
            selected = basis.records[:, 0] == component
            self.assertTrue(selected.any())
            variance = np.asarray(basis.spatial_matrix[:, selected].power(2)
                                  @ basis.std[selected]**2).ravel()
            area = (np.sum(geometry['au'] * geometry['wu'])
                    + np.sum(geometry['av'] * geometry['wv'])) if component < 2 else np.sum(
                        geometry['area'] * geometry['wet'])
            np.testing.assert_allclose(np.sqrt(weights @ variance / area), targets[name], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
