from __future__ import print_function

import numpy.testing as npt
from scipy.stats import beta

from ..statistics import *
from ..datasets import load_waltons, load_g3


def test_unequal_intensity_with_random_data():
    data1 = np.random.exponential(5, size=(2000, 1))
    data2 = np.random.exponential(1, size=(2000, 1))
    summary, p_value, result = logrank_test(data1, data2)
    assert result


def test_logrank_test_output_against_R():
    df = load_g3()
    ix = (df['group'] == 'RIT')
    d1, e1 = df.ix[ix]['time'], df.ix[ix]['event']
    d2, e2 = df.ix[~ix]['time'], df.ix[~ix]['event']

    expected = 0.0138
    summary, p_value, result = logrank_test(d1, d2, event_observed_A=e1, event_observed_B=e2)
    print(summary)
    assert abs(p_value - expected) < 0.000001


def test_unequal_intensity_event_observed():
    data1 = np.random.exponential(5, size=(2000, 1))
    data2 = np.random.exponential(1, size=(2000, 1))
    eventA = np.random.binomial(1, 0.5, size=(2000, 1))
    eventB = np.random.binomial(1, 0.5, size=(2000, 1))
    summary, p_value, result = logrank_test(data1, data2, event_observed_A=eventA, event_observed_B=eventB)
    assert result


def test_integer_times_logrank_test():
    data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
    data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
    summary, p_value, result = logrank_test(data1, data2)
    assert result


def test_waltons_dataset():
    df = load_waltons()
    ix = df['group'] == 'miR-137'
    waltonT1 = df.ix[ix]['T']
    waltonT2 = df.ix[~ix]['T']
    summary, p_value, result = logrank_test(waltonT1, waltonT2)
    assert result


def test_logrank_test_is_symmetric():
    data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
    data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
    summary1, p_value1, result1 = logrank_test(data1, data2)
    summary2, p_value2, result2 = logrank_test(data2, data1)
    assert abs(p_value2 - p_value1) < 10e-8
    assert result2 == result1


def test_multivariate_unequal_intensities():
    T = np.random.exponential(10, size=300)
    g = np.random.binomial(2, 0.5, size=300)
    T[g == 1] = np.random.exponential(1, size=(g == 1).sum())
    s, _, result = multivariate_logrank_test(T, g)
    assert result


def test_pairwise_waltons_dataset_is_significantly_different():
    waltons_dataset = load_waltons()
    _, _, R = pairwise_logrank_test(waltons_dataset['T'], waltons_dataset['group'])
    assert R.values[0, 1]


def test_pairwise_logrank_test_with_identical_data_returns_inconclusive():
    t = np.random.exponential(10, size=100)
    T = np.tile(t, 3)
    g = np.array([1, 2, 3]).repeat(100)
    S, P, R = pairwise_logrank_test(T, g, alpha=0.99)
    V = np.array([[np.nan, None, None], [None, np.nan, None], [None, None, np.nan]])
    npt.assert_array_equal(R, V)


def test_multivariate_inputs_return_identical_solutions():
    T = np.array([1, 2, 3])
    E = np.array([1, 1, 0], dtype=bool)
    G = np.array([1, 2, 1])
    multivariate_logrank_test(T, G, E)
    pairwise_logrank_test(T, G, E)

    T = pd.Series(T)
    E = pd.Series(E)
    G = pd.Series(G)
    multivariate_logrank_test(T, G, E)
    pairwise_logrank_test(T, G, E)


def test_pairwise_allows_dataframes():
    N = 100
    df = pd.DataFrame(np.empty((N, 3)), columns=["T", "C", "group"])
    df["T"] = np.random.exponential(1, size=N)
    df["C"] = np.random.binomial(1, 0.6, size=N)
    df["group"] = np.random.binomial(2, 0.5, size=N)
    pairwise_logrank_test(df['T'], df["group"], event_observed=df["C"])


def test_equal_intensity():
    """
    This is the (I think) fact that 1-alpha == false positive rate.
    I use a Bayesian test to test that we achieve this rate.
    """
    N = 100
    false_positives = 0
    alpha = 0.95
    for i in range(100):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(5, size=(200, 1))
        summary, p_value, result = logrank_test(data1, data2, alpha=0.95, suppress_print=True)
        false_positives += result is not None
    bounds = beta.interval(0.95, 1 + false_positives, N - false_positives + 1)
    assert bounds[0] < 1 - alpha < bounds[1]


def test_multivariate_equal_intensities():
    N = 100
    false_positives = 0
    alpha = 0.95
    for i in range(100):
        T = np.random.exponential(10, size=300)
        g = np.random.binomial(2, 0.5, size=300)
        s, _, result = multivariate_logrank_test(T, g, alpha=alpha, suppress_print=True)
        false_positives += result is not None
    bounds = beta.interval(0.95, 1 + false_positives, N - false_positives + 1)
    assert bounds[0] < 1 - alpha < bounds[1]
