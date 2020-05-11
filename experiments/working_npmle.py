# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from collections import defaultdict, namedtuple
from numpy.linalg import norm

# use NamedTuples for intervals
interval = namedtuple("Interval", ["left", "right"])


def E_step_M_step(observation_intervals, p_old, turnball_interval_lookup):

    N = len(observation_intervals)
    T = p_old.shape[0]

    p_new = np.zeros_like(p_old)

    for ix_i, observation_interval in enumerate(observation_intervals):
        p_temp = np.zeros_like(p_old)

        # find all turnball intervals, t, that are contained in (ol, or). Call this set T
        # the denominator is sum of p_old[T] probabilities
        # the numerator is p_old[t]
        for ix_t in turnball_interval_lookup[observation_interval]:
            t_p = p_old[ix_t]
            p_temp[ix_t] = t_p

        p_new = p_new + p_temp / p_temp.sum()

    return p_new / N


def create_turnball_intervals(left, right):
    """
    find the set of innermost intervals
    """

    # left and right can be deduped for faster performance.
    left, right = left.tolist(), right.tolist()
    heapify(left)
    heapify(right)

    left_, right_ = heappop(left), heappop(right)

    while True:
        try:
            left__ = heappop(left)
        except:
            break

        if right_ == np.inf:
            # find the last possible left_
            left_ = heapq.nlargest(1, left)[0]
            yield interval(left_, right_)
            break

        elif left__ < right_:
            left_ = left__
        elif right_ <= left__:
            yield interval(left_, right_)
            left_ = left__
            right_ = heappop(right)
            while right_ <= left_:
                right_ = heappop(right)
        else:
            print("why am I here")


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


def check_convergence(p_new, p_old):
    if norm(p_new - p_old) < 1e-4:
        return True
    return False


def create_observation_intervals(left, right):
    return [interval(l, r) for l, r in zip(left, right)]


def npmle(left, right):

    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    assert left.shape == right.shape

    # sort left, right arrays by (left, right).
    ix = np.lexsort((left, right))
    left = left[ix]
    right = right[ix]

    turnball_intervals = list(create_turnball_intervals(left, right))
    observation_intervals = create_observation_intervals(left, right)
    turnball_lookup = create_turnball_lookup(turnball_intervals, observation_intervals)

    T = len(turnball_intervals)

    converged = False

    # initialize to equal weight
    p = 1 / T * np.ones(T)

    while not converged:

        p_new = E_step_M_step(observation_intervals, p, turnball_lookup)
        converged = check_convergence(p_new, p)
        p = p_new

    return p, turnball_intervals


left, right = zip(*[(0, 8), (7, 16), (7, 14), (17, np.inf), (37, 44), (6, 10), (45, np.inf), (46, np.inf), (46, np.inf), (0, 7)])
results = npmle(left, right)
print(results)

# plot(*results)
