# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from lifelines.utils.concordance import concordance_index
from lifelines.utils.concordance import concordance_index as fast_cindex
from lifelines.utils.concordance import naive_concordance_index as slow_cindex


def test_concordance_index_returns_same_after_shifting():
    T = np.array([1, 2, 3, 4, 5, 6])
    T_ = np.array([2, 1, 4, 6, 5, 3])
    assert (
        concordance_index(T, T_)
        == concordance_index(T - 5, T_ - 5)
        == concordance_index(T, T_ - 5)
        == concordance_index(T - 5, T_)
    )


def test_both_concordance_index_function_deal_with_ties_the_same_way():
    actual_times = np.array([1, 1, 2])
    predicted_times = np.array([1, 2, 3])
    obs = np.ones(3)
    assert fast_cindex(actual_times, predicted_times, obs) == slow_cindex(actual_times, predicted_times, obs) == 1.0


def test_both_concordance_index_with_only_censoring_fails_gracefully():
    actual_times = np.array([1, 2, 3])
    predicted_times = np.array([1, 2, 3])
    obs = np.zeros(3)
    with pytest.raises(ZeroDivisionError, match="admissable pairs"):
        fast_cindex(actual_times, predicted_times, obs)

    with pytest.raises(ZeroDivisionError, match="admissable pairs"):
        slow_cindex(actual_times, predicted_times, obs)


def test_concordance_index_function_exits():
    N = 10 * 1000
    actual_times = np.random.exponential(1, size=N)
    predicted_times = np.random.exponential(1, size=N)
    obs = np.ones(N)
    assert fast_cindex(actual_times, predicted_times, obs)


def test_concordance_index_will_not_overflow():
    a = np.arange(65536)
    assert concordance_index(a, a) == 1.0
    b = np.arange(65537)
    assert concordance_index(b, b) == 1.0
    assert concordance_index(b, b[::-1]) == 0.0


def test_concordance_index_fast_is_same_as_slow():
    size = 100
    T = np.random.normal(size=size)
    P = np.random.normal(size=size)
    C = np.random.choice([0, 1], size=size)
    Z = np.zeros_like(T)

    # Hard to imagine these failing
    assert slow_cindex(T, Z, C) == fast_cindex(T, Z, C)
    assert slow_cindex(T, T, C) == fast_cindex(T, T, C)
    # This is the real test though
    assert slow_cindex(T, P, C) == fast_cindex(T, P, C)

    cp = CoxPHFitter()
    df = load_rossi()
    cp.fit(df, duration_col="week", event_col="arrest")

    T = cp.durations.values.ravel()
    P = -cp.predict_partial_hazard(df[df.columns.difference(["week", "arrest"])]).values.ravel()

    E = cp.event_observed.values.ravel()

    assert slow_cindex(T, P, E) == fast_cindex(T, P, E)
