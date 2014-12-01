from __future__ import print_function

import numpy.testing as npt
from scipy.stats import beta

from ..statistics import *
from ..datasets import load_waltons

def test_unequal_intensity():
    data1 = np.random.exponential(5, size=(2000, 1))
    data2 = np.random.exponential(1, size=(2000, 1))
    summary, p_value, result = logrank_test(data1, data2)
    assert result

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
    summary2, p_value2, result2 = logrank_test(data1, data2)
    assert p_value2 == p_value1
    assert result2 == result1
    
def test_multivariate_unequal_intensities():
    T = np.random.exponential(10, size=300)
    g = np.random.binomial(2, 0.5, size=300)
    T[g == 1] = np.random.exponential(6, size=(g == 1).sum())
    s, _, result = multivariate_logrank_test(T, g)
    assert result

def test_pairwise_waltons_dataset():
    waltons_dataset = load_waltons()
    _, _, R = pairwise_logrank_test(waltons_dataset['T'], waltons_dataset['group'])
    assert R.values[0, 1]

def test_pairwise_logrank_test():
    T = np.random.exponential(10, size=500)
    g = np.random.binomial(2, 0.7, size=500)
    S, P, R = pairwise_logrank_test(T, g, alpha=0.99)
    V = np.array([[np.nan, None, None], [None, np.nan, None], [None, None, np.nan]])
    npt.assert_array_equal(R, V)


def test_multivariate_inputs():
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

def test_concordance_index():
    size = 1000
    T = np.random.normal(size=size)
    P = np.random.normal(size=size)
    C = np.random.choice([0, 1], size=size)
    Z = np.zeros_like(T)

    # Zeros is exactly random
    assert concordance_index(T, Z) == 0.5
    assert concordance_index(T, Z, C) == 0.5

    # Itself is 1
    assert concordance_index(T, T) == 1.0
    assert concordance_index(T, T, C) == 1.0

    # Random is close to 0.5
    assert abs(concordance_index(T, P) - 0.5) < 0.05
    assert abs(concordance_index(T, P, C) - 0.5) < 0.05

def test_concordance_index_returns_same_after_shifting():
    T = np.array([1, 2, 3, 4, 5, 6])
    T_ = np.array([2, 1, 4, 6, 5, 3])
    assert concordance_index(T, T_) == concordance_index(T - 5, T_ - 5) == concordance_index(T, T_ - 5) == concordance_index(T - 5, T_)



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


