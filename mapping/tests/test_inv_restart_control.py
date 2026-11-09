import numpy as np

from src.inv import Cov, _analysis_to_minimizer_control


def test_saved_analysis_is_converted_to_unpreconditioned_increment():
    background = np.array([2., -3.])
    analysis = np.array([5., 1.])

    control = _analysis_to_minimizer_control(
        analysis, background, Cov(np.array([10., 20.])), False)

    np.testing.assert_allclose(control, [3., 4.])


def test_saved_analysis_is_converted_to_preconditioned_control():
    background = np.array([2., -3.])
    analysis = np.array([5., 1.])
    covariance = Cov(np.array([3., 2.]))

    control = _analysis_to_minimizer_control(
        analysis, background, covariance, True)

    np.testing.assert_allclose(control, [1., 2.])
    np.testing.assert_allclose(
        background + covariance.sqr(control), analysis)
