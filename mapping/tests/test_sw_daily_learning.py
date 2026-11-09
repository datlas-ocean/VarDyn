"""Daily covariance, posterior moments, EM, and SW forward regression checks."""
import pytest
pytest.importorskip("learning_local.sw_learning", reason="Optional local SW learning sources are absent")

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src import config_default as defaults
from src.basis import Basis
from src.exp import Config
from src.basis import DailyAR1Covariance
from learning_local.sw_learning import LearningOptions, posterior_step, maximize_statistics, learn


class DailyLearningTests(unittest.TestCase):
    def test_ar1_dense_covariance_and_transpose(self):
        op = DailyAR1Covariance([2., .4], [3., 10.], 5)
        L = np.asarray(jax.vmap(op.sqrt)(jnp.eye(10))).T
        q = L@L.T
        t = np.arange(5)
        for k, (sigma, tau) in enumerate(((2., 3.), (.4, 10.))):
            np.testing.assert_allclose(q[k::2, k::2], sigma**2*np.exp(-abs(t[:, None]-t)/tau), rtol=2e-6)
        x = jnp.arange(10, dtype=float)/10
        np.testing.assert_allclose(op.inverse_sqrt(op.sqrt(x)), x, atol=5e-7)
        np.testing.assert_allclose(op.sqrt_transpose(x), L.T@x, atol=5e-7)
        np.testing.assert_allclose(op.inverse(x), np.linalg.solve(q, x), rtol=2e-5, atol=2e-5)

    def test_linear_posterior_matches_dense_conditioning(self):
        shape = (6, 1)
        y = jnp.array([.1, .4, -.2, .8, 0., -.1]).reshape(shape)
        sigma_r = .8
        residual = lambda a: (a-y)/sigma_r
        options = LearningOptions(probes=512, map_gtol=1.e-5, cg_rtol=1.e-6)
        a, p, lag, _ = posterior_step(residual, np.array([1.3]), np.array([3.]), np.zeros(1), np.zeros(shape), options, 2)
        t = np.arange(6)
        q = 1.3**2*np.exp(-abs(t[:, None]-t)/3.)
        exact = np.linalg.inv(np.linalg.inv(q)+np.eye(6)/sigma_r**2)
        mu = exact@np.asarray(y).ravel()/sigma_r**2
        np.testing.assert_allclose(a.ravel(), mu, atol=2e-5)
        np.testing.assert_allclose(p.ravel(), np.diag(exact), rtol=.2)
        np.testing.assert_allclose(lag.ravel(), np.diag(exact, 1), atol=.06)

    def test_m_step_preserves_prior_without_observations(self):
        # Posterior equals prior; dropping P would collapse std to zero.
        n = 40
        std, tau = np.array([2.]), np.array([6.])
        a = np.full((n, 1), .3)
        p = np.full_like(a, 4.)
        lag = np.full((n-1, 1), 4*np.exp(-1/6))
        options = LearningOptions(regional_penalty=0.)
        s, t, mean, _ = maximize_statistics(a, p, lag, std, tau, np.array([0]), np.empty((0, 2), int), np.log([std, tau]), options)
        np.testing.assert_allclose(s, std, rtol=1e-4)
        np.testing.assert_allclose(t, tau, rtol=1e-4)
        np.testing.assert_allclose(mean, [.3], atol=1e-6)

    def test_daily_basis_interpolation_and_layout(self):
        ny, nx = 7, 8
        x, y = np.meshgrid(np.arange(nx)*25000., np.arange(ny)*25000.)
        s = SimpleNamespace(nx=nx, ny=ny, X=x, Y=y, DX=x*0+25000, DY=y*0+25000, mask=None)
        cfg = Config(BASIS=Config(deepcopy(defaults.BASIS_SW_DAILY)|dict(super='BASIS_SW_DAILY', wavelengths_km=[200.])), EXP=Config(init_date=datetime(2020, 1, 1)))
        b = Basis(cfg, s, verbose=False)
        b.set_basis([0., 3.])
        signature, size = b.layout_signature, b.nbasis
        rng = np.random.default_rng(7)
        control = jnp.asarray(rng.normal(size=b.nbasis))
        np.testing.assert_allclose(b.coefficients(1.25, control), .75*np.asarray(control).reshape(4, -1)[1]+.25*np.asarray(control).reshape(4, -1)[2], atol=1e-6)
        fields = dict(zip(b.name_mod_var, (jnp.asarray(rng.normal(size=sh)) for sh in b.shapes)))
        left = sum(jnp.vdot(a, fields[name]) for a, name in zip(b.project(1.25, control), b.name_mod_var))
        right = jnp.vdot(control, b.operg_transpose(1.25, fields))
        np.testing.assert_allclose(left, right, atol=2e-5)
        b.update_statistics(b.std*2, b.tdec*3)
        self.assertEqual(b.nbasis, size)
        self.assertEqual(b.layout_signature, signature)

    def test_em_recovers_synthetic_ar1_and_checkpoint_guard(self):
        rng = np.random.default_rng(42)
        n, m = 240, 3
        truth = np.zeros((n, m)); truth[0] = rng.normal(size=m)*2
        rho = np.exp(-1/5)
        for i in range(1, n):
            truth[i] = rho*truth[i-1]+2*np.sqrt(1-rho**2)*rng.normal(size=m)
        y = jnp.asarray(truth+.1*rng.normal(size=truth.shape))
        records = np.zeros((m, 8)); records[:, 1] = 200.; records[:, 3] = np.arange(m)*50000
        b = SimpleNamespace(std=np.ones(m), tdec=np.full(m, 2.), mean=np.zeros(m), ntime=n, natom=m,
            records=records, layout_signature='test', start=0., origin=0., update_statistics=lambda *args: None)
        options = LearningOptions(outer_iterations=3, pooled_iterations=1, probes=8, regional_penalty=2., map_maxiter=150, map_gtol=1e-3)
        with tempfile.TemporaryDirectory() as folder:
            result = learn(lambda a: (a-y)/.1, b, folder, options)
            self.assertTrue(np.all((result['std'] > 1.5) & (result['std'] < 2.5)))
            self.assertTrue(np.all((result['tdec'] > 2.5) & (result['tdec'] < 8.)))
            with self.assertRaises(FileExistsError):
                learn(lambda a: (a-y)/.1, b, folder, options)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                learn(lambda a: (a-y)/.1, b, folder, options, resume=True, provenance='changed')
            resumed = learn(lambda a: (a-y)/.1, b, folder, options, resume=True)
            np.testing.assert_array_equal(resumed['std'], result['std'])
            np.testing.assert_allclose(resumed['analysis'], result['analysis'], atol=1.e-6)

    def test_sw_forward_gradient_and_continuity(self):
        from datetime import timedelta
        from src.mod import Model_qgsw
        from src.state import State
        from learning_local.sw_learning_forward import make_daily_residual
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config()
            cfg.EXP = Config(defaults.EXP | dict(path_save=directory, tmp_DA_path=directory,
                flag_plot=0, init_date=datetime(2024, 1, 1), final_date=datetime(2024, 1, 3)))
            cfg.GRID = Config(defaults.GRID_GEO | dict(super='GRID_GEO', lon_min=-2., lon_max=2.,
                lat_min=-2., lat_max=2., dlon=.5, dlat=.5, name_init_mask=None))
            cfg.MOD = Config(defaults.MOD_QGSW | dict(super='MOD_QGSW', H=250., g_prime=.025,
                dtmodel=1800., cfl=None, height_representation='interface_displacement',
                dist_sponge_bc=0., h_adv_scheme='linear_upwind5', mom_adv_scheme='upwind5'))
            cfg.BASIS = Config(deepcopy(defaults.BASIS_SW_DAILY)|dict(super='BASIS_SW_DAILY', wavelengths_km=[300.]))
            state = State(cfg, verbose=False)
            model = Model_qgsw(cfg, state)
            model.init(state)
            b = Basis(cfg, state, verbose=False)
            b.set_basis([0., 2.])
            class Observations:
                def scan_misfit_size(self): return state.nx*state.ny
                def scan_misfit(self, t, fields): return fields['ssh'].ravel()
            residual = make_daily_residual(model, state, b, Observations(), [0., 86400., 172800.])
            # Basin-mean height control increases the domain-mean height across
            # day boundaries; resetting the model each day fails this check.
            a = np.zeros((3, b.natom), 'float32')
            a[:, b.records[:, 0] == 3] = .01
            values = np.asarray(jax.jit(residual)(jnp.asarray(a))).reshape(3, state.ny, state.nx)
            self.assertGreater(np.mean(values[2]), 1.5*np.mean(values[1]))
            direction = jnp.asarray(np.random.default_rng(4).normal(size=a.shape).astype('float32'))*.001
            f = jax.jit(lambda control: .5*jnp.sum(residual(control)**2))
            grad = jax.jit(jax.grad(f))(jnp.asarray(a))
            epsilon = .05
            fd = (f(a+epsilon*direction)-f(a-epsilon*direction))/(2*epsilon)
            np.testing.assert_allclose(fd, jnp.vdot(grad, direction), rtol=.02, atol=1.e-5)
            # A held-out day contributes neither residual nor precision rows.
            held = make_daily_residual(model, state, b, Observations(), [0., 86400., 172800.], [False, True, False])
            np.testing.assert_allclose(jax.jit(held)(jnp.asarray(a)), values[1].ravel(), rtol=1.e-5)

    def test_existing_4dvar_accepts_daily_covariance(self):
        from unittest.mock import patch
        import test_sw_forcing_qgsw as integration
        def daily(cfg, state, **kwargs):
            cfg.BASIS = Config(deepcopy(defaults.BASIS_SW_DAILY) | dict(super='BASIS_SW_DAILY', wavelengths_km=[200.]))
            return Basis(cfg, state, **kwargs)
        with patch.object(integration, 'Basis', daily):
            integration.SWForcingQGSWTests().test_existing_python_and_scan_adjoint_and_finite_difference()


if __name__ == '__main__':
    unittest.main()
