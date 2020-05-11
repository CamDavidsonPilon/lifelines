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

    for observation_interval in observation_intervals:
        p_temp = np.zeros_like(p_old)

        # find all turnball intervals, t, that are contained in (ol, or). Call this set T
        # the denominator is sum of p_old[T] probabilities
        # the numerator is p_old[t]
        for ix_t in turnball_interval_lookup[observation_interval]:
            t_p = p_old[ix_t]
            p_temp[ix_t] = t_p

        p_new = p_new + p_temp / p_temp.sum()

    return p_new / N


def heappop_(array):
    return heappop(array)


def create_turnball_intervals(left, right):
    """
    find the set of innermost intervals

    """

    left, right = list(set(left)), list(set(right))
    heapify(left)
    heapify(right)

    left_, right_ = heappop_(left), heappop_(right)
    assert left_ <= right_

    while True:
        try:
            # import pdb
            # pdb.set_trace()
            if left_ < right_:
                left__ = heappop_(left)
                while left__ < right_:
                    left_ = left__
                    left__ = heappop_(left)
                yield interval(left_, right_)

                left_ = left__
                if left_ != right_:
                    right_ = heappop_(right)
                else:
                    yield interval(left_, right_)
                    right_ = heappop_(right)

            elif left_ > right_:
                while right_ <= left_:
                    right_ = heappop_(right)

                yield interval(left_, right_)
                left_, right_ = heappop_(left), heappop_(right)

            elif left_ == right_:
                yield interval(left_, right_)
                right_ = heappop_(right)
        except IndexError:
            yield interval(left_, right_)
            break


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
    ix = np.lexsort((right, left))
    left = left[ix]
    right = right[ix]
    turnball_intervals = list(create_turnball_intervals(left, right))  # fix this set problem
    observation_intervals = create_observation_intervals(left, right)
    turnball_lookup = create_turnball_lookup(turnball_intervals, sorted(set(observation_intervals)))

    T = len(turnball_intervals)

    converged = False

    # initialize to equal weight
    p = 1 / T * np.ones(T)

    while not converged:

        p_new = E_step_M_step(observation_intervals, p, turnball_lookup)
        converged = check_convergence(p_new, p)
        p = p_new

    return p, turnball_intervals


def reconstruct_survival_function(probabilities, turnball_intervals, timeline):
    index = [0.0]
    values = [1.0]

    for p, interval in zip(probabilities, turnball_intervals):
        if interval.left != index[-1]:
            index.append(interval.left)
            values.append(values[-1])

        index.append(interval.right)
        values.append(values[-1] - p)

    full_dataframe = pd.DataFrame(index=timeline, columns=["survival function"])

    turnball_dataframe = pd.DataFrame(values, index=index, columns=["survival function"])

    dataframe = full_dataframe.combine_first(turnball_dataframe).ffill()
    return dataframe


def compute_confidence_intervals(left, right, mle_, alpha=0.05, samples=10):
    """
    uses basic bootstrap
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    all_times = np.unique(np.concatenate((left, right, [np.inf, 0])))

    N = left.shape[0]

    bootstrapped_samples = np.empty((all_times.shape[0], samples))

    for i in range(samples):
        ix = np.random.randint(low=0, high=N, size=N)
        left_ = left[ix]
        right_ = right[ix]

        bootstrapped_samples[:, i] = reconstruct_survival_function(*npmle(left_, right_), all_times).values[:, 0]

    return (
        2 * mle_.squeeze() - pd.Series(np.percentile(bootstraps, alpha / 2 * 100, axis=1), index=all_times),
        2 * mle_.squeeze() - pd.Series(np.percentile(bootstraps, (1 - alpha / 2) * 100, axis=1), index=all_times),
    )


from lifelines.datasets import load_diabetes

data = load_diabetes()

left, right = list(data["left"]), list(data["right"])
results = npmle(left, right)


timeline = np.unique(np.concatenate((left, right, [np.inf, 0])))
df = reconstruct_survival_function(*results, timeline)
CIs = compute_confidence_intervals(left, right, df)
