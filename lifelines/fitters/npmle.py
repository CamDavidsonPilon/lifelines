# -*- coding: utf-8 -*-
"""
This code isn't to be called directly, but is the core logic of the KaplanMeierFitter.fit_interval_censoring

References
https://upcommons.upc.edu/bitstream/handle/2117/93831/01Rop01de01.pdf


"""
from collections import defaultdict, namedtuple
import numpy as np
from numpy.linalg import norm
import pandas as pd

interval = namedtuple("Interval", ["left", "right"])


def E_step_M_step(observation_intervals, p_old, turnball_interval_lookup):

    N = len(observation_intervals)
    T = p_old.shape[0]

    p_new = np.zeros_like(p_old)

    for observation_interval in observation_intervals:
        p_temp = np.zeros_like(p_old)

        # find all turnball intervals, t, that are contained in (ol, or). Call this set T
        # the denominator is sum of p_old[T] probabilities
        # the numerator is p_old[t]

        # TODO: I think I can remove the bottom for loop:
        # > ix = turnball_interval_lookup[observation_interval]
        # > p_temp[ix] = p_old[ix]
        for ix_t in turnball_interval_lookup[observation_interval]:
            p_temp[ix_t] = p_old[ix_t]

        p_new = p_new + p_temp / p_temp.sum()

    return p_new / N


def create_turnball_intervals(left, right):
    """
    TIHI X 10000
    """

    left = [[l, 0, "1l"] for l in left]
    right = [[r, 0, "0r"] for r in right]

    for l, r in zip(left, right):
        if l[0] == r[0]:
            l[1] -= 0.01
            r[1] += 0.01

    import copy

    union = sorted(list(left) + list(right))

    """
    # fix ties
    for k in range(len(union)-1):
        e_, e__ = union[k], union[k+1]
        if e_[2] == "1l" and e__[2] == "0r" and e_[0] == e__[0]:
            union_[k][1] += 0.01
    """
    intervals = []

    for k in range(len(union) - 1):
        e_, e__ = union[k], union[k + 1]
        if e_[2] == "1l" and e__[2] == "0r":
            intervals.append(interval(e_[0], e__[0]))

    return sorted(set(intervals))


def is_subset(query_interval, super_interval):
    return super_interval.left <= query_interval.left and query_interval.right <= super_interval.right


def create_turnball_lookup(turnball_intervals, observation_intervals):

    turnball_lookup = defaultdict(set)

    for i, turnball_interval in enumerate(turnball_intervals):
        # ask: which observations is this t_interval part of?
        for observation_interval in observation_intervals:
            # since left and right are sorted by left, we can stop after left > turnball_interval[1] value
            if observation_interval.left > turnball_interval.right:
                break
            if is_subset(turnball_interval, observation_interval):
                turnball_lookup[observation_interval].add(i)

    return turnball_lookup


def check_convergence(p_new, p_old, tol, i, verbose=False):
    if verbose:
        print("Iteration %d: norm(p_new - p_old): %.6f" % (i, norm(p_new - p_old)))
    if norm(p_new - p_old) < tol:
        return True
    return False


def create_observation_intervals(left, right):
    return [interval(l, r) for l, r in zip(left, right)]


def npmle(left, right, tol=1e-5, verbose=False):

    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    assert left.shape == right.shape

    # sort left, right arrays by (left, right).
    ix = np.lexsort((right, left))
    left = left[ix]
    right = right[ix]

    turnball_intervals = list(create_turnball_intervals(left, right))
    observation_intervals = create_observation_intervals(left, right)
    turnball_lookup = create_turnball_lookup(turnball_intervals, sorted(set(observation_intervals)))
    print(turnball_lookup)

    """ is correct
    turnball_lookup = {
        interval(6.0, 6.0): {0},
        interval(7.0, 7.0): {1},
        interval(7.0, np.inf): {2},  # {1, 2}
        interval(8., 8.): {2},
    }
    """
    print(turnball_lookup)

    T = len(turnball_intervals)

    converged = False

    # initialize to equal weight
    p = 1 / T * np.ones(T)
    i = 0
    while not converged:
        i += 1
        p_new = E_step_M_step(observation_intervals, p, turnball_lookup)
        converged = check_convergence(p_new, p, tol, i, verbose=verbose)
        p = p_new

    return p, turnball_intervals


def reconstruct_survival_function(probabilities, turnball_intervals, timeline, label="NPMLE"):
    """
    TIHI

    """
    index = []
    values = []

    for i, (p, interval) in enumerate(zip(probabilities, turnball_intervals)):
        if i == 0:
            index.append(interval.left)
            index.append(interval.right)
            values.append(1.0)
            values.append(1 - p)
            continue

        if interval.left != index[-1]:
            index.append(interval.left)
            values.append(values[-1])

        if interval.left == interval.right:
            values[-1] -= p
        else:
            index.append(interval.right)
            values.append(values[-1] - p)

    full_dataframe = pd.DataFrame(index=timeline, columns=[label + "_lower"])

    turnball_dataframe = pd.DataFrame(values, index=index, columns=[label + "_lower"])

    dataframe = full_dataframe.combine_first(turnball_dataframe).ffill().fillna(1)
    dataframe[label + "_upper"] = dataframe[label + "_lower"].shift(1).fillna(1)
    return dataframe


def npmle_compute_confidence_intervals(left, right, mle_, alpha=0.05, samples=1000):
    """
    uses basic bootstrap
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    all_times = np.unique(np.concatenate((left, right, [0])))

    N = left.shape[0]

    bootstrapped_samples = np.empty((all_times.shape[0], samples))

    for i in range(samples):
        ix = np.random.randint(low=0, high=N, size=N)
        left_ = left[ix]
        right_ = right[ix]

        bootstrapped_samples[:, i] = reconstruct_survival_function(*npmle(left_, right_), all_times).values[:, 0]

    return (
        2 * mle_.squeeze() - pd.Series(np.percentile(bootstrapped_samples, (alpha / 2) * 100, axis=1), index=all_times),
        2 * mle_.squeeze() - pd.Series(np.percentile(bootstrapped_samples, (1 - alpha / 2) * 100, axis=1), index=all_times),
    )


"""
Test cases
"""


ti = create_turnball_intervals(*zip(*[(0, 1), (4, 6), (2, 6), (0, 3), (2, 4), (5, 7)]))

assert ti == [interval(0, 1), interval(2, 3), interval(5, 6)]


ti = create_turnball_intervals(*zip(*[(4, 7), (3, 5), (0, 2), (1, 4), (6, 9), (8, 10)]))

assert ti == [interval(1, 2), interval(3, 4), interval(4, 5), interval(6, 7), interval(8, 9)]
