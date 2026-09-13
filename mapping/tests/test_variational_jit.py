from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from src.inv import Cov, Variational


class _State:
    def __init__(self, var=None, params=None):
        self.var = var or {"x": jnp.array([1.0, 2.0], dtype=jnp.float32)}
        self.params = params or {"x": jnp.zeros(2, dtype=jnp.float32)}

    def copy(self, free=False):
        if free:
            return _State(
                {name: jnp.zeros_like(value) for name, value in self.var.items()},
                {
                    name: jnp.zeros_like(value)
                    for name, value in self.params.items()
                },
            )
        return _State(dict(self.var), dict(self.params))


class _LinearModel:
    dt = 1
    T = np.array([0.0, 1.0])

    def step(self, State, nstep=1, t=0):
        State.var["x"] = State.var["x"] + nstep * State.params["x"]

    def step_adj(self, adState, State, nstep=1, t=0):
        adState.params["x"] = (
            adState.params["x"] + nstep * adState.var["x"]
        )


class _LinearObsop:
    def is_obs_time(self, t):
        return True

    def misfit(self, t, State):
        return State.var["x"]

    def adj(self, t, adState, State, misfit):
        adState.var["x"] = adState.var["x"] + misfit

    def scan_misfit_size(self):
        return 2

    def scan_misfit(self, t, State_var):
        return State_var["x"]

    def scan_adj(self, t, adState_var, State_var, misfit):
        result = dict(adState_var)
        result["x"] = result["x"] + misfit
        return result


class _LinearBasis:
    nbasis = 1

    def operg(self, t, X, State=None):
        State["x"] = jnp.full_like(State["x"], X[0])

    def operg_transpose(self, t, adState):
        result = jnp.sum(adState["x"], keepdims=True)
        adState["x"] = jnp.zeros_like(adState["x"])
        return result


def _config(schedule="python"):
    return SimpleNamespace(
        EXP=SimpleNamespace(tmp_DA_path="/tmp"),
        INV=SimpleNamespace(
            jit_cost_and_grad=True,
            cost_and_grad_schedule=schedule,
            cost_float64=False,
            timestep_checkpoint=SimpleNamespace(total_seconds=lambda: 1),
            prec=True,
            save_minimization=False,
            freq_it_plot=10,
            print_time=False,
            plot_state_during_minimization=False,
            compute_test=False,
        ),
    )


def test_compiled_and_legacy_cost_gradient_are_equivalent():
    variational = Variational(
        config=_config(),
        M=_LinearModel(),
        H=_LinearObsop(),
        State=_State(),
        R=Cov(1.0),
        B=Cov(jnp.ones(1, dtype=jnp.float32)),
        Basis=_LinearBasis(),
        Xb=jnp.zeros(1, dtype=jnp.float32),
        checkpoints=np.array([0, 1]),
        freq_it_plot=10,
        print_time=False,
    )
    control = jax.device_put(jnp.array([0.25], dtype=jnp.float32))

    expected_cost, expected_gradient = variational._cost_and_grad_impl(control)
    compiled_cost, compiled_gradient = variational.cost_and_grad(control)
    jax.block_until_ready((compiled_cost, compiled_gradient))

    np.testing.assert_allclose(compiled_cost, expected_cost, rtol=1e-6)
    np.testing.assert_allclose(
        compiled_gradient,
        expected_gradient,
        rtol=1e-6,
    )
    assert variational.it_plot == 1


def test_scan_and_python_schedules_are_equivalent():
    kwargs = dict(
        M=_LinearModel(),
        H=_LinearObsop(),
        State=_State(),
        R=Cov(1.0),
        B=Cov(jnp.ones(1, dtype=jnp.float32)),
        Basis=_LinearBasis(),
        Xb=jnp.zeros(1, dtype=jnp.float32),
        checkpoints=np.array([0, 1]),
        freq_it_plot=10,
        print_time=False,
    )
    python_schedule = Variational(config=_config("python"), **kwargs)
    scan_schedule = Variational(config=_config("scan"), **kwargs)
    control = jax.device_put(jnp.array([0.25], dtype=jnp.float32))

    python_cost, python_gradient = python_schedule.cost_and_grad(control)
    scan_cost, scan_gradient = scan_schedule.cost_and_grad(control)
    jax.block_until_ready(
        (python_cost, python_gradient, scan_cost, scan_gradient)
    )

    np.testing.assert_allclose(scan_cost, python_cost, rtol=1e-6)
    np.testing.assert_allclose(scan_gradient, python_gradient, rtol=1e-6)
