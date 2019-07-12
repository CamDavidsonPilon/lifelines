# -*- coding: utf-8 -*-
from autograd.extend import primitive, defvjp
from autograd import numpy as np
from autograd.scipy.special import gammaln
from autograd.numpy.numpy_vjps import unbroadcast_f
from scipy.special import gammainc as _scipy_gammainc, gammaincc as _scipy_gammaincc, gamma

__all__ = [
    "gammainc",  # regularized lower incomplete gamma function
    "gammaincc",  # regularized upper incomplete gamma function
    "gamma",  # gamma function
    "reg_lower_inc_gamma",  # alias
    "reg_upper_inc_gamma",  # alias
]


LOG_EPISILON = 1e-35
MACHINE_EPISLON_POWER = np.finfo(float).eps ** (1 / 2)

gammainc = primitive(_scipy_gammainc)
gammaincc = primitive(_scipy_gammaincc)


@primitive
def gammainccln(a, x):
    return np.log(np.clip(gammaincc(a, x), LOG_EPISILON, 1 - LOG_EPISILON))


@primitive
def gammaincln(a, x):
    return np.log(np.clip(gammainc(a, x), LOG_EPISILON, 1 - LOG_EPISILON))


def central_difference_of_(f):
    def _central_difference(ans, a, x):
        # Why do we calculate a * MACHINE_EPSILON_POWER?
        # consider if a is massive, like, 2**100. Then even for a simple
        # function like the identity function, (2**100 + h) - 2**100 = 0  due
        # to floating points. (the correct answer should be 1.0)

        # another thing to consider (and later to add) is that x is machine representable, but x + h is
        # rarely, and will be rounded to be machine representable. This (x + h) - x != h.

        delta = np.maximum(a * MACHINE_EPISLON_POWER, 1e-8)
        return unbroadcast_f(
            a,
            lambda g: g
            * (-f(a + 2 * delta, x) + 8 * f(a + delta, x) - 8 * f(a - delta, x) + f(a - 2 * delta, x))
            / (12 * delta),
        )

    return _central_difference


defvjp(
    gammainc,
    central_difference_of_(gammainc),
    lambda ans, a, x: unbroadcast_f(x, lambda g: g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a))),
)

defvjp(
    gammaincc,
    central_difference_of_(gammaincc),
    lambda ans, a, x: unbroadcast_f(x, lambda g: -g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a))),
)

defvjp(
    gammainccln,
    central_difference_of_(gammainccln),
    lambda ans, a, x: unbroadcast_f(
        x, lambda g: -g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a) - gammainccln(a, x))
    ),
)

defvjp(
    gammaincln,
    central_difference_of_(gammaincln),
    lambda ans, a, x: unbroadcast_f(x, lambda g: g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a) - gammaincln(a, x))),
)


reg_lower_inc_gamma = gammainc
reg_upper_inc_gamma = gammaincc
