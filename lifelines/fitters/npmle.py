# -*- coding: utf-8 -*-
"""
This code isn't to be called directly, but is the core logic of the KaplanMeierFitter.fit_interval_censoring

References
https://upcommons.upc.edu/bitstream/handle/2117/93831/01Rop01de01.pdf
https://docs.ufpr.br/~giolo/CE063/Artigos/A4_Gomes%20et%20al%202009.pdf

"""
from collections import defaultdict, namedtuple
import warnings
import numpy as np
from numpy.linalg import norm
import pandas as pd
from lifelines.utils import ConvergenceWarning

interval = namedtuple("Interval", ["left", "right"])


class min_max:
    """
    Keep only the min/max of streaming values
    """

    def __init__(self):
        self.min = np.inf
        self.max = -np.inf

    def add(self, value):
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

    def __iter__(self):
        yield self.min
        yield self.max


def E_step_M_step(observation_intervals, p_old, turnbull_interval_lookup, weights):

    N = 0
    p_new = np.zeros_like(p_old)
    for observation_interval, w in zip(observation_intervals, weights):
        # find all turnbull intervals, t, that are contained in (ol, or). Call this set T
        # the denominator is sum of p_old[T] probabilities
        # the numerator is p_old[t]

        min_, max_ = turnbull_interval_lookup[observation_interval]
        p_new[min_ : max_ + 1] += w * p_old[min_ : max_ + 1] / p_old[min_ : max_ + 1].sum()
        N += w

    return p_new / N


def create_turnbull_intervals(left, right):
    """
    obs are []
    turnbulls are []
    """
    left = [[l, "l"] for l in left]
    right = [[r, "r"] for r in right]

    union = sorted(left + right)

    intervals = []

    for e1, e2 in zip(union, union[1:]):
        if e1[1] == "l" and e2[1] == "r":
            intervals.append(interval(e1[0], e2[0]))

    return intervals


def is_subset(query_interval, super_interval):
    """
    assumes query_interval is [], and super_interval is (]
    """
    return super_interval.left <= query_interval.left and query_interval.right <= super_interval.right


def create_turnbull_lookup(turnbull_intervals, observation_intervals):

    turnbull_lookup = defaultdict(min_max)

    for i, turnbull_interval in enumerate(turnbull_intervals):
        # ask: which observations is this t_interval part of?
        for observation_interval in observation_intervals:
            # since left and right are sorted by left, we can stop after left > turnbull_interval[1] value
            if observation_interval.left > turnbull_interval.right:
                break
            if is_subset(turnbull_interval, observation_interval):
                turnbull_lookup[observation_interval].add(i)

    return {o: list(s) for o, s in turnbull_lookup.items()}


def check_convergence(p_new, p_old, tol, i, verbose=False):
    if verbose:
        print("Iteration %d: delta: %.6f" % (i, norm(p_new - p_old)))
    if norm(p_new - p_old) < tol:
        return True
    return False


def create_observation_intervals(obs):
    return [interval(l, r) for l, r in obs]


def odds(p):
    return p / (1 - p)


def probs(o):
    return o / (o + 1)


def npmle(left, right, tol=1e-5, weights=None, verbose=False, max_iter=1e5):
    """
    left and right are closed intervals.
    TODO: extend this to open-closed intervals.
    """
    left, right = np.asarray(left), np.asarray(right)

    if weights is None:
        weights = np.ones_like(left)

    # perform a group by to get unique observations and weights
    df_ = pd.DataFrame({"l": left, "r": right, "w": weights}).groupby(["l", "r"]).sum()
    weights = df_["w"].values
    unique_obs = df_.index.values

    # create objects needed
    turnbull_intervals = create_turnbull_intervals(left, right)
    observation_intervals = create_observation_intervals(unique_obs)
    turnbull_lookup = create_turnbull_lookup(turnbull_intervals, observation_intervals)

    # convergence init
    converged = False
    i = 0

    # initialize to equal weight
    T = len(turnbull_intervals)
    p = 1 / T * np.ones(T)

    while (not converged) and (i < max_iter):
        p_new = E_step_M_step(observation_intervals, p, turnbull_lookup, weights)
        converged = check_convergence(p_new, p, tol, i, verbose=verbose)

        # find alpha that maximizes ll using a line search
        best_alpha, best_p, best_ll = None, None, -np.inf
        delta = odds(p_new) - odds(p)
        for alpha in np.array([1.0, 1.25, 1.95]):
            p_temp = probs(odds(p) + alpha * delta)
            ll_temp = log_likelihood(observation_intervals, p_temp, turnbull_lookup, weights)
            if best_ll < ll_temp:
                best_ll = ll_temp
                best_alpha = alpha
                best_p = p_temp

        p = best_p

        i += 1

    if i >= max_iter:
        warnings.warn("Exceeded max iterations", ConvergenceWarning)

    return p, turnbull_intervals


def log_likelihood(observation_intervals, p, turnbull_interval_lookup, weights):

    ll = 0
    for observation_interval, w in zip(observation_intervals, weights):
        min_, max_ = turnbull_interval_lookup[observation_interval]
        ll += w * np.log(p[min_ : max_ + 1].sum())
    return ll


def reconstruct_survival_function(probabilities, turnbull_intervals, timeline=None, label="NPMLE"):

    if timeline is None:
        timeline = []

    index = np.unique(turnbull_intervals)
    label_upper = label + "_upper"
    label_lower = label + "_lower"
    df = pd.DataFrame([], index=index, columns=[label_upper, label_lower])

    running_sum = 1.0
    # the below values may be overwritten later, but we
    # always default to starting at point (0, 1)
    df.loc[0, label_upper] = running_sum
    df.loc[0, label_lower] = running_sum

    for p, (left, right) in zip(probabilities, turnbull_intervals):
        df.loc[left, label_upper] = running_sum
        df.loc[left, label_lower] = running_sum

        if left != right:
            df.loc[right, label_upper] = running_sum
            df.loc[right, label_lower] = running_sum - p

        running_sum -= p

    full_dataframe = pd.DataFrame(index=timeline, columns=df.columns)

    return full_dataframe.combine_first(df).bfill().sort_index()


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
