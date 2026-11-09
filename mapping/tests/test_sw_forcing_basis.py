"""Coastal, temporal, transpose and restart invariants for SW forcing controls."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from src import config_default as defaults
from src.basis import Basis
from src.exp import Config


def problem(**overrides):
    ny, nx = 12, 16
    x, y = np.meshgrid(np.arange(nx)*25000., np.arange(ny)*25000.)
    wet = np.ones((ny, nx), bool)
    wet[4:7, 6:9] = False
    state = SimpleNamespace(nx=nx, ny=ny, X=x, Y=y, DX=np.full_like(x, 25000.),
                            DY=np.full_like(y, 25000.), mask=~wet,
                            params={"u": jnp.zeros((ny, nx+1)),
                                    "v": jnp.zeros((ny+1, nx)), "ssh": jnp.zeros((ny, nx))})
    options = deepcopy(defaults.BASIS_SW_FORCING)
    options.update(super="BASIS_SW_FORCING", spatial_radius_km=125., height_radius_km=200.,
                   slow_halfwidth_days=4., wave_envelope_halfwidth_days=6., wave_period_days=(3., 5.))
    options.update(overrides)
    config = Config(BASIS=Config(options))
    basis = Basis(config, state, verbose=False)
    basis.set_basis(np.arange(0, 12.01, .25), return_q=True, State=state)
    return basis, state, config


class SWForcingTests(unittest.TestCase):
    def test_transpose_and_jit_at_all_temporal_families(self):
        basis, state, _ = problem()
        rng = np.random.default_rng(22)
        direction = jnp.asarray(rng.normal(size=basis.nbasis), dtype=jnp.float32)
        weights = {name: jnp.asarray(rng.normal(size=v.shape), dtype=jnp.float32)
                   for name, v in state.params.items()}
        for time in (0., 1.125, 5.75, 12., 13.):
            forward = jax.jit(lambda z: sum(jnp.vdot(a, weights[n]) for a, n in
                                           zip(basis.project(time, z), basis.name_mod_var)))
            gradient = jax.jit(lambda: basis.operg_transpose(time, dict(weights)))()
            np.testing.assert_allclose(forward(direction), jnp.vdot(gradient, direction), rtol=2e-5, atol=2e-4)
            np.testing.assert_allclose(jax.grad(forward)(direction), gradient, rtol=2e-5, atol=2e-5)

    def test_no_coastal_flux_height_mass_and_rotational_divergence(self):
        basis, _, _ = problem()
        rng = np.random.default_rng(3)
        for component, (matrices, size) in basis.spatial.items():
            coefficients = jnp.asarray(rng.normal(size=size), dtype=jnp.float32)
            u, v, h = [np.asarray(m @ coefficients).reshape(shape)
                       for m, shape in zip(matrices, basis.shapes)]
            np.testing.assert_array_equal(u[~basis.wet_u], 0)
            np.testing.assert_array_equal(v[~basis.wet_v], 0)
            np.testing.assert_array_equal(h[~basis.wet], 0)
            if component == "rotational":
                divergence = (np.diff(u*basis.face_y, axis=1)+np.diff(v*basis.face_x, axis=0))/basis.area
                self.assertLess(abs(divergence).max(), 1e-10)
            if component == "height":
                residual = abs(np.sum(h*basis.area)) / np.sum(abs(h)*basis.area)
                self.assertLess(residual, 1e-6)
        self.assertNotIn("height_mean", basis.spatial)

    def test_bias_is_separate_from_zero_mean_slow_and_wave_fields(self):
        basis, _, _ = problem()
        t = jnp.linspace(0, 12, 2049)
        for family in ("slow", "wave"):
            values = np.asarray(jax.vmap(basis.temporal[family].at)(t))
            mean = np.trapezoid(values, np.asarray(t), axis=0)/12
            np.testing.assert_allclose(mean, 0, atol=2e-4)
            gram = np.trapezoid(values[:, :, None]*values[:, None, :], np.asarray(t), axis=0)/12
            np.testing.assert_allclose(gram, np.eye(values.shape[1]), atol=3e-4)
        np.testing.assert_array_equal(basis.temporal["bias"].at(0), basis.temporal["bias"].at(12))

    def test_mean_height_control_is_explicit_and_never_oscillatory(self):
        slow = deepcopy(defaults.BASIS_SW_FORCING["slow_std"])
        slow["height_mean"] = .1
        basis, _, _ = problem(slow_std=slow)
        matrices, size = basis.spatial["height_mean"]
        self.assertEqual(size, basis.nbasin)
        u, v, h = [np.asarray(a @ jnp.ones(size)).reshape(s)
                   for a, s in zip(matrices, basis.shapes)]
        np.testing.assert_array_equal(u, 0)
        np.testing.assert_array_equal(v, 0)
        np.testing.assert_array_equal(h, basis.wet.astype(float))
        blocks = [b for b in basis.blocks if b["component"] == "height_mean"]
        self.assertEqual([b["family"] for b in blocks], ["slow"])
        self.assertEqual(blocks[0]["std"], .1)
        wave = deepcopy(defaults.BASIS_SW_FORCING["wave_std"])
        wave["height_mean"] = .1
        with self.assertRaisesRegex(ValueError, "Wave forcing of the basin mean is disabled"):
            problem(wave_std=wave)

    def test_multi_basis_is_additive(self):
        basis, state, config = problem()
        one = Config(deepcopy(dict(config.BASIS)))
        config.BASIS = Config(first=one, second=Config(deepcopy(dict(one))))
        multi = Basis(config, state, verbose=False)
        multi.set_basis(np.arange(0, 12.01, .25), return_q=True, State=state)
        x = jnp.ones(basis.nbasis)
        fields = dict(state.params)
        multi.operg(1.25, jnp.concatenate((x, x)), State=fields)
        for name, value in zip(basis.name_mod_var, basis.project(1.25, x)):
            np.testing.assert_allclose(fields[name], 2*value, rtol=2e-6, atol=2e-6)

    def test_rejects_unresolved_wave_cadence(self):
        basis, state, _ = problem()
        with self.assertRaisesRegex(ValueError, "8 basis updates"):
            basis.set_basis(np.arange(13.), return_q=True, State=state)


if __name__ == "__main__":
    unittest.main()
