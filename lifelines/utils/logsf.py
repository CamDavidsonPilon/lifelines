# -*- coding: utf-8 -*-

from scipy.stats import norm as _scipy_norm
import autograd.numpy as np
from autograd.scipy.stats import norm
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f

# TODO: next release of autograd will have this built in.

logsf = primitive(_scipy_norm.logsf)

defvjp(
    logsf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        x, lambda g: -g * np.exp(norm.logpdf(x, loc, scale) - logsf(x, loc, scale))
    ),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        loc, lambda g: g * np.exp(norm.logpdf(x, loc, scale) - logsf(x, loc, scale))
    ),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: g * np.exp(norm.logpdf(x, loc, scale) - logsf(x, loc, scale)) * (x - loc) / scale
    ),
)
