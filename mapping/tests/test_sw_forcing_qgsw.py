"""SSH-only 4DVar integration test of the public SW forcing basis."""
from copy import deepcopy
from datetime import datetime, timedelta
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

from src import config_default as defaults
from src.basis import Basis
from src.tools import DiagonalCovariance
from src.exp import Config
from src.inv import Variational
from src.mod import Model_qgsw
from src.state import State


class SSHOnly:
    def __init__(self, state, scale):
        self.scale = scale
        y, x = np.indices(state.var["ssh"].shape)
        self.target = jnp.asarray(.002*np.cos(x/3)*np.cos(y/3))

    def is_obs_time(self, t):
        return True

    def scan_misfit_size(self):
        return self.target.size

    def scan_misfit(self, t, fields):
        return (self.scale*fields["ssh"]-self.target).ravel()

    def misfit(self, t, state):
        return self.scan_misfit(t, state.var)

    def scan_adj(self, t, adjoint, state, misfit):
        # VarDyn observation adjoints include the observation precision.
        adjoint["ssh"] += self.scale*misfit.reshape(self.target.shape)/.002**2
        return adjoint

    def adj(self, t, adjoint, state, misfit):
        adjoint.var = self.scan_adj(t, adjoint.var, state.var, misfit)


class SWForcingQGSWTests(unittest.TestCase):
    def test_existing_python_and_scan_adjoint_and_finite_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config()
            cfg.EXP = Config(defaults.EXP | dict(path_save=directory, tmp_DA_path=directory,
                flag_plot=0, init_date=datetime(2024, 1, 1), final_date=datetime(2024, 1, 1, 6)))
            cfg.GRID = Config(defaults.GRID_GEO | dict(super="GRID_GEO", lon_min=-.5, lon_max=.5,
                lat_min=-.5, lat_max=.5, dlon=.1, dlat=.1, name_init_mask=None))
            cfg.MOD = Config(defaults.MOD_QGSW | dict(super="MOD_QGSW", H=100., g_prime=.04,
                dtmodel=300., cfl=None, height_representation="interface_displacement",
                dist_sponge_bc=0., h_adv_scheme="linear_upwind5", mom_adv_scheme="upwind5"))
            cfg.BASIS = Config(deepcopy(defaults.BASIS_SW_FORCING) | dict(super="BASIS_SW_FORCING",
                spatial_radius_km=50., height_radius_km=100., slow_halfwidth_days=.2,
                wave_envelope_halfwidth_days=.2, wave_period_days=(.25,)))
            cfg.INV = Config(defaults.INV_4DVAR | dict(prec=True, timestep_checkpoint=timedelta(minutes=30),
                save_minimization=False, compute_test=False, print_time=False,
                plot_state_during_minimization=False, cost_float64=False,
                jit_cost_and_grad=False, cost_and_grad_schedule="python"))
            state = State(cfg, verbose=False)
            model = Model_qgsw(cfg, state)
            model.init(state)
            state.to_device()
            checkpoints = np.arange(0, len(model.T), 6)
            basis = Basis(cfg, state, verbose=False)
            background, _ = basis.set_basis(model.T[checkpoints]/86400., return_q=True, State=state)
            obs = SSHOnly(state, .04/9.81)
            def evaluator(schedule):
                cfg.INV.cost_and_grad_schedule = schedule
                cfg.INV.jit_cost_and_grad = schedule == "scan"
                return Variational(config=cfg, M=model, H=obs, State=state,
                    B=basis.covariance_operator, R=DiagonalCovariance(.002), Basis=basis,
                    Xb=background, checkpoints=checkpoints, freq_it_plot=100, print_time=False)
            plain = evaluator("python")
            scan = evaluator("scan")
            x = jnp.zeros(basis.nbasis)
            jp, gp = plain.cost_and_grad(x)
            js, gs = scan.cost_and_grad(x)
            np.testing.assert_allclose(js, jp, rtol=2e-5)
            np.testing.assert_allclose(gs, gp, rtol=2e-3, atol=2e-4)
            direction = gs/jnp.linalg.norm(gs)
            epsilon = .02
            finite_difference = (scan.cost_and_grad(x+epsilon*direction)[0]
                                 -scan.cost_and_grad(x-epsilon*direction)[0])/(2*epsilon)
            np.testing.assert_allclose(finite_difference, jnp.vdot(gs, direction), rtol=.02, atol=.001)
            self.assertLess(float(scan.cost_and_grad(x-.01*direction)[0]), float(js))


if __name__ == "__main__":
    unittest.main()
