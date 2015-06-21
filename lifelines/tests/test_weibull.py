
import numpy as np
from lifelines import weibull_fitter as wf


def test_lambda_gradient():

    E = np.array([1])
    T = np.array([10])
    rho = lambda_ = 1
    assert - wf._lambda_gradient([lambda_, rho], T, E) == -9

    E = np.array([1, 1])
    T = np.array([10, 10])
    assert - wf._lambda_gradient([lambda_, rho], T, E) == -9 * 2


def test_rho_gradient():

    E = np.array([1])
    T = np.array([10])
    rho = lambda_ = 1
    assert - wf._rho_gradient([lambda_, rho], T, E) == 1 + 1 * np.log(10) - np.log(10) * 10

    E = np.array([1, 1])
    T = np.array([10, 10])
    assert - wf._rho_gradient([lambda_, rho], T, E) == 2 * (1 + 1 * np.log(10) - np.log(10) * 10)
