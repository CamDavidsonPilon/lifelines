# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from collections import defaultdict

# use NamedTuples for intervals


def E_step_M_step(left, right, p_old, turnball_interval_lookup):

    N = left.shape[0]
    T = p_old.shape[0]

    p_new = np.zeros_like(p_old)

    for ix_i, (o_l, o_r) in enumerate(zip(left, right)):
        p_temp = np.zeros_like(p_old)
        denominator = 0

        # find all turnball intervals, t, that are contained in (ol, or). Call this set T
        # the denominator is sum of p_old[T] probabilities
        # the numerator is p_old[t]
        for ix_t in turnball_interval_lookup[(o_l, o_r)]:
            t_p = p_old[ix_t]
            p_temp[ix_t] = t_p
            denominator += t_p

        p_new = p_new + p_temp / denominator

    return p_new / N


def create_turnball_intervals(left, right):
    unique_times = np.unique(np.concatenate((left, right, [0.0, np.inf])))

    turnball_intervals = []
    turnball_lookup = defaultdict(set)

    for i, _ in enumerate(unique_times[:-1]):
        turnball_interval = (unique_times[i], unique_times[i + 1])
        turnball_intervals.append(turnball_interval)

        # ask: which observations is this t_interval part of?
        for l, r in zip(left, right):
            # since left and right are sorted by left, we can stop after left > turnball_interval[1] value
            if l > turnball_interval[1]:
                break
            if l <= turnball_interval[0] and turnball_interval[1] <= r:
                turnball_lookup[(l, r)].add(i)

    return turnball_intervals, turnball_lookup


def check_convergence(p_new, p_old):
    if np.abs(p_new - p_old).sum() < 0.001:
        return True
    return False


def npmle(left, right):

    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape

    # sort left, right arrays by left.
    ix = np.argsort(left)
    left = left[ix]
    right = right[ix]

    turnball_intervals, turnball_interval_lookup = create_turnball_intervals(left, right)

    N = left.shape[0]
    T = len(turnball_intervals)

    converged = False

    # initialize to equal weight
    p = 1 / T * np.ones(T)

    while not converged:

        p_new = E_step_M_step(left, right, p, turnball_interval_lookup)
        converged = check_convergence(p_new, p)
        p = p_new

    return p, turnball_intervals


def plot(p, intervals):
    lhs, _ = zip(*intervals)
    lhs = np.insert(np.array(lhs), len(lhs), 60)
    y = np.insert(1 - p.cumsum(), 0, 1)

    ax = plt.plot(lhs, y)
    return ax


left, right = zip(*[(0, 7), (0, 9), (6, 10), (7, 16), (7, 14), (17, np.inf), (37, 44), (45, np.inf), (46, np.inf), (46, np.inf)])
results = npmle(left, right)


plot(*results)


assert np.abs(results[0][0] - 5.23991462e-04) < 1e-5
assert np.abs(results[0][1] - 1.66151063e-01) < 1e-5
assert np.abs(results[0][-1] - 3.74999998e-01) < 1e-5
