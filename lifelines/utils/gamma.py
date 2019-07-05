# -*- coding: utf-8 -*-
from autograd import numpy as np
from autograd.scipy.special import gamma, gammaln

__all__ = ["upper_inc_gamma", "lower_inc_gamma"]


def upper_inc_gamma(s, z):
    return z ** s * np.exp(-z) / _upper_gamma2(s, z)


def log_upper_inc_gamma(s, z):
    return s * np.log(z) - z - np.log(_upper_gamma2(s, z))


def _upper_gamma(s, z):
    CONTD_FRAC_ITERATIONS = 60
    v = np.ones_like(s)
    for n in range(CONTD_FRAC_ITERATIONS, 0, -1):
        v = ((2 * n - 1) + z - s) + (n * (s - n)) / v
    return np.clip(v, 1e-15, np.inf)


def _upper_gamma2(s, z, frac=500):
    CONTD_FRAC_ITERATIONS = frac
    assert CONTD_FRAC_ITERATIONS % 2 == 0, "must be even"
    n = CONTD_FRAC_ITERATIONS
    v = 1.0

    while n > 0:
        v = z + (n / 2 + 1 - s) / v  # even
        n -= 1
        print(v)
        v = 1 + ((n + 1) / 2) / v  # odd
        n -= 1
        print(v)

    v = z + (1 - s) / v

    return v


def regularized_upper_inc_gamma(s, z):
    v = np.clip(upper_inc_gamma(s, z) / gamma(s), 1e-15, 1 - 1e-15)
    return v


def log_regularized_upper_inc_gamma(s, z):
    return np.clip(log_upper_inc_gamma(s, z) - gammaln(s), -1e15, 0)


def regularized_lower_inc_gamma(s, z):
    v = np.clip(lower_inc_gamma(s, z) / gamma(s), 1e-15, 1 - 1e-15)
    return v


def lower_inc_gamma(s, z):
    """
    Unstable when s << z because of floating point problems in _lower_gamma.
    In this case, s and z should be high floating point
    accurate numbers (float64 or float128)
    """
    v = z ** s * np.exp(-z) / _lower_gamma(s, z)
    return v


def _lower_gamma(s, z, frac=40):
    CONTD_FRAC_ITERATIONS = frac
    assert CONTD_FRAC_ITERATIONS % 2 == 0, "must be even"
    n = CONTD_FRAC_ITERATIONS
    v = 1.0

    while n > 0:
        v = (s + n) - ((s + n / 2) * z) / v  # even n
        n -= 1
        v = (s + n) + ((n + 1) / 2) * z / v  # odd n
        n -= 1
    v = s - s * z / v

    return np.clip(v, 1e-15, np.inf)
