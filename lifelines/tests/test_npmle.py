# -*- coding: utf-8 -*-
import pytest
from lifelines.fitters.npmle import npmle, is_subset, create_turnbull_intervals, interval
from numpy import testing as npt
import numpy as np


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
    npt.assert_allclose(npmle(left, right)[0], np.array([0.16667016, 0.33332984, 0.125, 0.375]))


def test_npmle_with_weights_is_identical_if_uniform_weights():
    left, right = [1, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    weights = 2 * np.ones_like(right)
    npt.assert_allclose(npmle(left, right)[0], np.array([0.16667016, 0.33332984, 0.125, 0.375]))


def test_npmle_with_weights():
    sol = np.array([0.2051282, 0.4102564, 0.0961539, 0.2884615])

    left, right = [1, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    weights = np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
    npt.assert_allclose(npmle(left, right, weights=weights)[0], sol)

    left, right = [1, 1, 8, 8, 8, 8, 7, 7, 17, 37, 46, 46, 45], [7, 7, 8, 8, 10, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf]
    npt.assert_allclose(npmle(left, right)[0], sol, rtol=1e-4)
