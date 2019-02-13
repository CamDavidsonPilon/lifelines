"""
MIT License

Copyright (c) 2018 Better

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Edited in 2019, Cameron Davidson-Pilon
"""
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f  # This is not documented
from scipy.special import gammainc as gammainc_orig


@primitive
def gammainc(k, x):
    """ Lower regularized incomplete gamma function.
    We rely on `scipy.special.gammainc
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html>`_
    for this. However, there is a number of issues using this function
    together with `autograd <https://github.com/HIPS/autograd>`_:
    1. It doesn't let you take the gradient with respect to k
    2. The gradient with respect to x is really slow
    As a really stupid workaround, because we don't need the numbers to
    be 100% exact, we just approximate the gradient.
    Side note 1: if you truly want to compute the correct derivative, see the
    `Wikipedia articule about the Incomplete gamma function
    <https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives>`_
    where the T(3, s, x) function can be implemented as
    .. code-block:: python
       def T3(s, x):
           return mpmath.meijerg(a_s=([], [0, 0]), b_s=([s-1, -1, -1], []), z=x)
    I wasted a few hours on this but sadly it turns out to be extremely slow.
    Side note 2: TensorFlow actually has a `similar bug
    <https://github.com/tensorflow/tensorflow/issues/17995>`_
    """
    return gammainc_orig(k, x)


@primitive
def gammainc2(k, x):
    return gammainc_orig(k, x)


G_EPS = 1e-8

defvjp(
    gammainc2,
    lambda ans, k, x: unbroadcast_f(
        k, lambda g: g * (gammainc_orig(k + G_EPS, x) - 2 * ans + gammainc_orig(k - G_EPS, x)) / G_EPS ** 2
    ),
    lambda ans, k, x: unbroadcast_f(
        k, lambda g: g * (gammainc_orig(k, x + G_EPS) - 2 * ans + gammainc_orig(k, x - G_EPS)) / G_EPS ** 2
    ),
)

defvjp(
    gammainc,
    lambda ans, k, x: unbroadcast_f(k, lambda g: g * (gammainc2(k + G_EPS, x) - ans) / G_EPS),
    lambda ans, k, x: unbroadcast_f(x, lambda g: g * (gammainc2(k, x + G_EPS) - ans) / G_EPS),
)
