# -*- coding: utf-8 -*-
"""
Experimenting with some data generation and inference for left censorship with one or more
minimum detectable limits.

Recall that MLE bias is equal to 0 up to the order 1/sqrt(n), so we expect that for
small n, we will see a bias.

"""

from lifelines import WeibullFitter


def one_detection_limit(dist, N, fraction_below_limit):

    T_actual = 0.5 * np.random.weibull(1, size=N)

    MIN_1 = np.percentile(T_actual, fraction_below_limit)

    T = np.maximum(MIN_1, T_actual)
    E = T_actual > MIN_1

    wf = WeibullFitter().fit(T, E, left_censorship=True)
    return wf


def three_detection_limit(N):

    T_actual = 0.5 * np.random.weibull(5, size=N)

    MIN_0 = np.percentile(T_actual, 5)
    MIN_1 = np.percentile(T_actual, 10)
    MIN_2 = np.percentile(T_actual, 30)
    MIN_3 = np.percentile(T_actual, 50)

    T = T_actual.copy()
    ix = np.random.randint(4, size=N)

    T = np.where(ix == 0, np.maximum(T, MIN_0), T)
    T = np.where(ix == 1, np.maximum(T, MIN_1), T)
    T = np.where(ix == 2, np.maximum(T, MIN_2), T)
    T = np.where(ix == 3, np.maximum(T, MIN_3), T)
    E = T_actual == T

    wf = WeibullFitter().fit(T, E, left_censorship=True)
    return wf


# biased
np.mean([three_detection_limit(50).rho_ for _ in range(1000)])


# less biased
np.mean([three_detection_limit(500).rho_ for _ in range(1000)])
