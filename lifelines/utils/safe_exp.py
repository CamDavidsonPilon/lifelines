# -*- coding: utf-8 -*-
"""
One interesting idea in autograd is the ability to "hijack"
derivatives if they start misbehaving. In this example, it's
possible for exponentials to get very large with even modestly
sized arguments, and their derivatives can be arbitrarily larger.

One solution is to limit the `exp` to never be above
some very large limit. That's what `np.clip(x, -np.inf, MAX)`
is doing. The function looks like this:

                     exp capped
+--------------------------------------------------------+
|                                                        |
|                                                        |
|                                        X+--------------+
|                                        X               |
|                                        X               |
|                                        X               |
|                                        X               |
|                                       X                |
|                                       X                |
|                                      X                 |
|                                     X                  |
|                                   XX                   |
|                                 XX                     |
|                              XX                        |
|                       X X XX                           |
| x x    x  X   X  X XX                                  |
+--------------------------------------------------------+
                      0                  M


However, the `clip` function has derivative 0 after it's reached its
max, M. That will mess up any derivatives:

                     (exp capped)' (bad)
+--------------------------------------------------------+
|                                                        |
|                                                        |
|                                        X               |
|                                        X               |
|                                        X               |
|                                        X               |
|                                        X               |
|                                       X                |
|                                       X                |
|                                      X                 |
|                                     X                  |
|                                   XX                   |
|                                 XX                     |
|                              XX                        |
|                       X X XX                           |
| x x    x  X   X  X XX                   +--------------+               |
+--------------------------------------------------------+
                      0                  M



So we make the derivative _the same_ as the original function:

                   (exp capped)' (good)
+--------------------------------------------------------+
|                                                        |
|                                                        |
|                                        X+--------------+
|                                        X               |
|                                        X               |
|                                        X               |
|                                        X               |
|                                       X                |
|                                       X                |
|                                      X                 |
|                                     X                  |
|                                   XX                   |
|                                 XX                     |
|                              XX                        |
|                       X X XX                           |
| x x    x  X   X  X XX                                  |
+--------------------------------------------------------+
                      0                  M


"""
from autograd.extend import primitive, defvjp
from autograd import numpy as np

MAX = np.log(np.finfo(float).max) - 75


def safe_exp_vjp(ans, x):
    return lambda g: g * ans


@primitive
def safe_exp(x):
    return np.exp(np.clip(x, -np.inf, MAX))


defvjp(safe_exp, safe_exp_vjp)
