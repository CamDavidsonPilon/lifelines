# -*- coding: utf-8 -*-
from scipy.special import gammainc as _scipy_gammainc, gammaincc as _scipy_gammaincc
from autograd import numpy as np
from autograd.scipy.special import gamma
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f


delta = 1e-9
gammainc = primitive(_scipy_gammainc)


defvjp(
    gammainc,
    lambda ans, a, x: unbroadcast_f(a, lambda g: g * (gammainc(a + delta, x) - gammainc(a - delta, x)) / (2 * delta)),
    lambda ans, a, x: unbroadcast_f(x, lambda g: g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
)


gammaincc = primitive(_scipy_gammaincc)


defvjp(
    gammaincc,
    lambda ans, a, x: unbroadcast_f(a, lambda g: g * (gammaincc(a + delta, x) - gammaincc(a - delta, x)) / (2 * delta)),
    lambda ans, a, x: unbroadcast_f(x, lambda g: -g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
)
