# -*- coding: utf-8 -*-
import pytest
from lifelines.fitters.npmle import npmle, is_subset, create_turnbull_intervals, interval, reconstruct_survival_function
from numpy import testing as npt
from lifelines.datasets import load_mice
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd


def test_is_subset():
    assert is_subset(interval(4, 4), interval(3, 7))
    assert is_subset(interval(4, 4), interval(4, 7))
    assert is_subset(interval(4, 7), interval(4, 7))
    assert is_subset(interval(4, 7), interval(3.9, 7))
    assert is_subset(interval(4, np.inf), interval(3, np.inf))

    assert not is_subset(interval(4, 9), interval(3.9, 7))
    assert not is_subset(interval(2, 9), interval(3.9, 7))
    assert not is_subset(interval(4, 9), interval(5, 7))


def test_create_turnbull_intervals():
    left, right = zip(*([0, 1], [4, 6], [2, 6], [0, 3], [2, 4], [5, 7]))
    assert create_turnbull_intervals(left, right) == [interval(0, 1), interval(2, 3), interval(4, 4), interval(5, 6)]

    left, right = zip(*[(0, 1)])
    assert create_turnbull_intervals(left, right) == [interval(0, 1)]

    left, right = zip(*[(-1, 1)])
    assert create_turnbull_intervals(left, right) == [interval(-1, 1)]

    left, right = zip(*[(-1, 1), (-2, 2)])
    assert create_turnbull_intervals(left, right) == [interval(-1, 1)]


def test_npmle():
    left, right = [1, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    npt.assert_allclose(npmle(left, right, verbose=True)[0], np.array([0.16667016, 0.33332984, 0.125, 0.375]), rtol=1e-4)


def test_npmle_with_weights_is_identical_if_uniform_weights():
    left, right = [1, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    weights = 2 * np.ones_like(right)
    npt.assert_allclose(npmle(left, right, verbose=True)[0], np.array([0.16667016, 0.33332984, 0.125, 0.375]), rtol=1e-4)


def test_npmle_with_weights():
    sol = np.array([0.2051282, 0.4102564, 0.0961539, 0.2884615])

    left, right = [1, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    weights = np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
    npt.assert_allclose(npmle(left, right, weights=weights)[0], sol, rtol=1e-4)

    left, right = [1, 1, 8, 8, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 7, 8, 8, 10, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    npt.assert_allclose(npmle(left, right)[0], sol, rtol=1e-4)


def test_sf_doesnt_return_nans():
    left = [6, 7, 8, 7, 5]
    right = [7, 8, 10, 16, 20]
    results = npmle(left, right)
    npt.assert_allclose(results[1], [interval(7, 7), interval(8, 8)])
    npt.assert_allclose(results[0], [0.5, 0.5])
    sf = reconstruct_survival_function(*results, timeline=[6, 7, 8, 16, 20])
    assert not np.isnan(sf.values).any()


def test_mice_and_optimization_flag():
    df = load_mice()
    results = npmle(df["l"], df["u"], verbose=True, optimize=True)
    npt.assert_allclose(results[0][0], 1 - 0.8571429, rtol=1e-4)
    npt.assert_allclose(results[0][-1], 0.166667, rtol=1e-4)


def test_mice_scipy():
    df = load_mice()
    results = npmle(df["l"], df["u"], verbose=True, fit_method="scipy")
    npt.assert_allclose(results[0][0], 1 - 0.8571429, rtol=1e-4)
    npt.assert_allclose(results[0][-1], 0.166667, rtol=1e-4)


def test_max_lower_bound_less_than_min_upper_bound():
    # https://github.com/CamDavidsonPilon/lifelines/issues/1151
    import numpy as np
    import pandas as pd
    from lifelines import KaplanMeierFitter

    # Data
    np.random.seed(1)
    left0 = np.random.normal(loc=60, scale=2, size=20)
    add_time = np.random.normal(loc=100, scale=2, size=10)
    right1 = left0[0:10] + add_time
    right0 = right1.tolist() + [np.inf] * 10

    # KaplanMeier
    model = KaplanMeierFitter()
    model.fit_interval_censoring(lower_bound=left0, upper_bound=right0)
