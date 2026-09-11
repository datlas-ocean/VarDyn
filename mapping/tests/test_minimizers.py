import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.inv import minimize_optax_decoupled


def _quadratic(control):
    return 0.5 * jnp.vdot(control, control), control


def test_optax_decoupled_converges_with_shared_relative_gradient_criterion():
    initial = jnp.array([1.0, 2.0], dtype=jnp.float32)

    result = minimize_optax_decoupled(
        _quadratic,
        initial,
        maxiter=20,
        history_size=3,
        relative_gradient_tolerance=0.6,
        convergence_patience=2,
        minimum_iterations=2,
    )

    assert result.converged
    assert result.status == "convergence criterion reached"
    assert result.iterations_completed == 2
    assert result.function_evaluations == 3
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result.control)),
        np.zeros(2),
        atol=1e-6,
    )


def test_optax_decoupled_runs_to_maxiter_without_a_criterion():
    result = minimize_optax_decoupled(
        _quadratic,
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        maxiter=3,
        history_size=3,
    )

    assert not result.converged
    assert result.status == "maximum iterations reached"
    assert result.iterations_completed == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maxiter", 0),
        ("history_size", 0),
        ("convergence_patience", 0),
        ("minimum_iterations", 0),
    ],
)
def test_optax_decoupled_rejects_invalid_configuration(field, value):
    options = dict(
        maxiter=3,
        history_size=3,
        convergence_patience=1,
        minimum_iterations=1,
    )
    options[field] = value

    with pytest.raises(ValueError):
        minimize_optax_decoupled(
            _quadratic,
            jnp.array([1.0], dtype=jnp.float32),
            **options,
        )
