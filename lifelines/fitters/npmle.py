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
from lifelines.exceptions import ConvergenceWarning
from typing import *

interval = namedtuple("interval", ["left", "right"])


class min_max:
    """
    Keep only the min/max of streaming values
    """

    def __init__(self):
        self.min = np.inf
        self.max = -np.inf

    def add(self, value: float):
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

    def __iter__(self):
        yield self.min
        yield self.max


def temper(i: int, optimize) -> float:
    if optimize:
        return 0.9 * (2 * np.arctan(i / 100) / np.pi) + 1
    else:
        return 1.0


def E_step_M_step(observation_intervals, p_old, turnbull_interval_lookup, weights, i, optimize) -> np.ndarray:
    """
    See [1], but also modifications.

    References
    -----------
    1. Clifford Anderson-Bergman (2016): An efficient implementation of the
     EMICM algorithm for the interval censored NPMLE, Journal of Computational and Graphical
     Statistics, DOI: 10.1080/10618600.2016.1208616
    """
    N = 0
    m = np.zeros_like(p_old)
    P = cumulative_sum(p_old)
    for observation_interval, w in zip(observation_intervals, weights):
        # find all turnbull intervals, t, that are contained in (ol, or). Call this set T
        # the denominator is sum of p_old[T] probabilities
        # the numerator is p_old[t]
        min_, max_ = turnbull_interval_lookup[observation_interval]
        m[min_ : max_ + 1] += w / (P[max_ + 1] - P[min_]).sum()
        N += w

    p_new = p_old * (m / N) ** temper(i, optimize)
    p_new /= p_new.sum()
    return p_new


def cumulative_sum(p: np.ndarray) -> np.ndarray:
    # return np.insert(p, 0, 0).cumsum()
    return np.concatenate((np.zeros(1), p)).cumsum()


def create_turnbull_intervals(left, right) -> List[interval]:
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


def is_subset(query_interval: interval, super_interval: interval) -> bool:
    """
    assumes query_interval is [], and super_interval is (]
    """
    return super_interval.left <= query_interval.left and query_interval.right <= super_interval.right


def create_turnbull_lookup(
    turnbull_intervals: List[interval], observation_intervals: List[interval]
) -> Dict[interval, List[interval]]:

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


def check_convergence(
    p_new: np.ndarray,
    p_old: np.ndarray,
    turnbull_lookup: Dict[interval, List[interval]],
    weights: np.ndarray,
    tol: float,
    i: int,
    verbose=False,
) -> bool:
    old_ll = log_likelihood(p_old, turnbull_lookup, weights)
    new_ll = log_likelihood(p_new, turnbull_lookup, weights)
    delta = new_ll - old_ll
    if verbose:
        print("Iteration %d " % i)
        print("   delta log-likelihood: %.10f" % delta)
        print("   log-like:             %.6f" % log_likelihood(p_new, turnbull_lookup, weights))
    if (delta < tol) and (delta >= 0):
        return True
    return False


def create_observation_intervals(obs) -> List[interval]:
    return [interval(l, r) for l, r in obs]


def log_odds(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log(1 - p)


def probs(log_odds: np.ndarray) -> np.ndarray:
    o = np.exp(log_odds)
    return o / (o + 1)


def npmle(left, right, tol=1e-7, weights=None, verbose=False, max_iter=1e5, optimize=False, fit_method="em"):
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

    if fit_method == "em":
        p = expectation_maximization_fit(
            observation_intervals, turnbull_intervals, turnbull_lookup, weights, tol, max_iter, optimize, verbose
        )
    elif fit_method == "scipy":
        p = scipy_minimize_fit(turnbull_lookup, turnbull_intervals, weights, tol, verbose)

    return p, turnbull_intervals


def scipy_minimize_fit(turnbull_interval_lookup, turnbull_intervals, weights, tol, verbose):
    import autograd.numpy as anp
    from autograd import value_and_grad
    from scipy.optimize import minimize

    def cumulative_sum(p):
        return anp.concatenate((anp.zeros(1), p)).cumsum()

    def negative_log_likelihood(p, turnbull_interval_lookup, weights):
        P = cumulative_sum(p)
        ix = anp.array(list(turnbull_interval_lookup.values()))
        return -(weights * anp.log(P[ix[:, 1] + 1] - P[ix[:, 0]])).sum()

    def con(p):
        return p.sum() - 1

    # initialize to equal weight
    T = len(turnbull_intervals)
    p = 1 / T * np.ones(T)

    cons = {"type": "eq", "fun": con}
    results = minimize(
        value_and_grad(negative_log_likelihood),
        args=(turnbull_interval_lookup, weights),
        x0=p,
        bounds=[(0, 1)] * T,
        jac=True,
        constraints=cons,
        tol=tol,
        options={"disp": verbose},
    )
    return results.x


def expectation_maximization_fit(
    observation_intervals, turnbull_intervals, turnbull_lookup, weights, tol, max_iter, optimize, verbose
):
    # convergence init
    converged = False
    i = 0

    # initialize to equal weight
    T = len(turnbull_intervals)
    p = 1 / T * np.ones(T)

    while (not converged) and (i < max_iter):
        new_p = E_step_M_step(observation_intervals, p, turnbull_lookup, weights, i, optimize)
        converged = check_convergence(new_p, p, turnbull_lookup, weights, tol, i, verbose=verbose)

        # find alpha that maximizes ll using a line search
        best_p, best_ll = p, -np.inf
        delta = log_odds(new_p) - log_odds(p)
        for alpha in np.array([1.0, 1.25, 1.95]):
            p_temp = probs(log_odds(p) + alpha * delta)
            ll_temp = log_likelihood(p_temp, turnbull_lookup, weights)
            if best_ll < ll_temp:
                best_ll = ll_temp
                best_p = p_temp

        p = best_p

        i += 1

    if i >= max_iter:
        warnings.warn("Exceeded max iterations.", ConvergenceWarning)

    return p


def log_likelihood(p: np.ndarray, turnbull_interval_lookup, weights) -> float:
    P = cumulative_sum(p)
    ix = np.array(list(turnbull_interval_lookup.values()))
    return (weights * np.log(P[ix[:, 1] + 1] - P[ix[:, 0]])).sum()


def reconstruct_survival_function(
    probabilities: np.ndarray, turnbull_intervals: List[interval], timeline=None, label="NPMLE"
) -> pd.DataFrame:

    if timeline is None:
        timeline = []

    index = np.unique(np.concatenate((turnbull_intervals, [(0, 0)])))
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

    # First backfill at events between known observations
    # Second fill all events _outside_ known obs with running_sum
    return full_dataframe.combine_first(df).astype(float).bfill().fillna(running_sum).clip(lower=0.0)


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
