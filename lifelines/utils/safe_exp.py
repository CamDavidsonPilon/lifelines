# -*- coding: utf-8 -*-
import sys
from autograd.extend import primitive, defvjp
from autograd import numpy as np

MAX = np.log(sys.float_info.max) - 75


def safe_exp_vjp(ans, x):
    return lambda g: g * ans


@primitive
def safe_exp(x):
    return np.exp(np.clip(x, -np.inf, MAX))


defvjp(safe_exp, safe_exp_vjp)
