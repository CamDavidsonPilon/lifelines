from __future__ import print_function
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

from lifelines import statistics as stats
from lifelines.datasets import load_waltons, load_g3

def test_sample_size_necessary_under_cph():
    assert stats.sample_size_necessary_under_cph(0.8, 1, 0.8, 0.2, 0.139) == (14, 14)
    assert stats.sample_size_necessary_under_cph(0.8, 1, 0.5, 0.5, 1.2) == (950, 950)
    assert stats.sample_size_necessary_under_cph(0.8, 1.5, 0.5, 0.5, 1.2) == (1231, 821)
    assert stats.sample_size_necessary_under_cph(0.8, 1.5, 0.5, 0.5, 1.2, alpha=0.01) == (1832, 1221)

def test_power_under_cph():
    assert abs(stats.power_under_cph(12,12, 0.8, 0.2, 0.139) - 0.744937) < 10e-6
    assert abs(stats.power_under_cph(12,20, 0.8, 0.2, 1.2) - 0.05178317) < 10e-6

def test_unequal_intensity_with_random_data():
    data1 = np.random.exponential(5, size=(2000, 1))
    data2 = np.random.exponential(1, size=(2000, 1))
    test_result = stats.logrank_test(data1, data2)
    assert test_result.is_significant


def test_logrank_test_output_against_R_1():
    df = load_g3()
    ix = (df['group'] == 'RIT')
    d1, e1 = df.ix[ix]['time'], df.ix[ix]['event']
    d2, e2 = df.ix[~ix]['time'], df.ix[~ix]['event']

    expected = 0.0138
    result = stats.logrank_test(d1, d2, event_observed_A=e1, event_observed_B=e2)
    assert abs(result.p_value - expected) < 0.0001


def test_logrank_test_output_against_R_2():
    # from https://stat.ethz.ch/education/semesters/ss2011/seminar/contents/presentation_2.pdf
    control_T = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
    control_E = np.ones_like(control_T)

    treatment_T = [6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 25]
    treatment_E = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    result = stats.logrank_test(control_T, treatment_T, event_observed_A=control_E, event_observed_B=treatment_E)
    expected_p_value = 4.17e-05

    assert abs(result.p_value - expected_p_value) < 0.0001
    assert abs(result.test_statistic - 16.8) < 0.1


def test_rank_test_output_against_R_no_censorship():
    """
    > time <- c(10,20,30,10,20,50)
    > status <- c(1,1,1,1,1,1)
    > treatment <- c(1,1,1,0,0,0)
    > survdiff(Surv(time, status) ~ treatment)
    """
    result = stats.multivariate_logrank_test([10, 20, 30, 10, 20, 50], [1, 1, 1, 0, 0, 0])
    r_p_value = 0.614107
    r_stat = 0.254237
    assert abs(result.p_value - r_p_value) < 10e-6
    assert abs(result.test_statistic - r_stat) < 10e-6


def test_rank_test_output_against_R_with_censorship():
    """
    > time <- c(10,20,30,10,20,50)
    > status <- c(1,0,1,1,0,1)
    > treatment <- c(1,1,1,0,0,0)
    > survdiff(Surv(time, status) ~ treatment)
    """
    result = stats.multivariate_logrank_test([10, 20, 30, 10, 20, 50], [1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1])
    r_p_value = 0.535143
    r_stat = 0.384615
    assert abs(result.p_value - r_p_value) < 10e-6
    assert abs(result.test_statistic - r_stat) < 10e-6


def test_unequal_intensity_event_observed():
    data1 = np.random.exponential(5, size=(2000, 1))
    data2 = np.random.exponential(1, size=(2000, 1))
    eventA = np.random.binomial(1, 0.5, size=(2000, 1))
    eventB = np.random.binomial(1, 0.5, size=(2000, 1))
    result = stats.logrank_test(data1, data2, event_observed_A=eventA, event_observed_B=eventB)
    assert result.is_significant


def test_integer_times_logrank_test():
    data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
    data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
    result = stats.logrank_test(data1, data2)
    assert result.is_significant


def test_equal_intensity_with_negative_data():
    data1 = np.random.normal(0, size=(2000, 1))
    data1 -= data1.mean()
    data1 /= data1.std()
    data2 = np.random.normal(0, size=(2000, 1))
    data2 -= data2.mean()
    data2 /= data2.std()
    result = stats.logrank_test(data1, data2)
    assert not result.is_significant


def test_unequal_intensity_with_negative_data():
    data1 = np.random.normal(-5, size=(2000, 1))
    data2 = np.random.normal(5, size=(2000, 1))
    result = stats.logrank_test(data1, data2)
    assert result.is_significant


def test_waltons_dataset():
    df = load_waltons()
    ix = df['group'] == 'miR-137'
    waltonT1 = df.ix[ix]['T']
    waltonT2 = df.ix[~ix]['T']
    result = stats.logrank_test(waltonT1, waltonT2)
    assert result.is_significant


def test_logrank_test_is_symmetric():
    data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
    data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
    result1 = stats.logrank_test(data1, data2)
    result2 = stats.logrank_test(data2, data1)
    assert abs(result1.p_value - result2.p_value) < 10e-8
    assert result2.is_significant == result1.is_significant


def test_multivariate_unequal_intensities():
    T = np.random.exponential(10, size=300)
    g = np.random.binomial(2, 0.5, size=300)
    T[g == 1] = np.random.exponential(1, size=(g == 1).sum())
    result = stats.multivariate_logrank_test(T, g)
    assert result.is_significant


def test_pairwise_waltons_dataset_is_significantly_different():
    waltons_dataset = load_waltons()
    R = stats.pairwise_logrank_test(waltons_dataset['T'], waltons_dataset['group'])
    assert R.values[0, 1].is_significant


def test_pairwise_logrank_test_with_identical_data_returns_inconclusive():
    t = np.random.exponential(10, size=100)
    T = np.tile(t, 3)
    g = np.array([1, 2, 3]).repeat(100)
    R = stats.pairwise_logrank_test(T, g, alpha=0.99).applymap(lambda r: r.is_significant if r is not None else None)
    V = np.array([[None, False, False], [False, None, False], [False, False, None]])
    npt.assert_array_equal(R.values, V)


def test_pairwise_allows_dataframes():
    N = 100
    df = pd.DataFrame(np.empty((N, 3)), columns=["T", "C", "group"])
    df["T"] = np.random.exponential(1, size=N)
    df["C"] = np.random.binomial(1, 0.6, size=N)
    df["group"] = np.random.binomial(2, 0.5, size=N)
    stats.pairwise_logrank_test(df['T'], df["group"], event_observed=df["C"])


def test_log_rank_returns_None_if_equal_arrays():
    T = np.random.exponential(5, size=200)
    result = stats.logrank_test(T, T, alpha=0.95)
    assert not result.is_significant

    C = np.random.binomial(2, 0.8, size=200)
    result = stats.logrank_test(T, T, C, C, alpha=0.95)
    assert not result.is_significant


def test_multivariate_log_rank_is_identital_to_log_rank_for_n_equals_2():
    N = 200
    T1 = np.random.exponential(5, size=N)
    T2 = np.random.exponential(5, size=N)
    C1 = np.random.binomial(2, 0.9, size=N)
    C2 = np.random.binomial(2, 0.9, size=N)
    result = stats.logrank_test(T1, T2, C1, C2, alpha=0.95)

    T = np.r_[T1, T2]
    C = np.r_[C1, C2]
    G = np.array([1] * 200 + [2] * 200)
    result_m = stats.multivariate_logrank_test(T, G, C, alpha=0.95)
    assert result.is_significant == result_m.is_significant
    assert result.p_value == result_m.p_value


def test_StatisticalResult_class():
    sr = stats.StatisticalResult(True, 0.04, 5.0)
    assert sr.is_significant
    assert sr.test_result == "Reject Null"

    sr = stats.StatisticalResult(None, 0.04, 5.0)
    assert not sr.is_significant
    assert sr.test_result == "Cannot Reject Null"

    sr = stats.StatisticalResult(True, 0.05, 5.0, kw='some_value')
    assert hasattr(sr, 'kw')
    assert getattr(sr, 'kw') == 'some_value'
    assert 'some_value' in sr.__unicode__()


def test_valueerror_is_raised_if_alpha_out_of_bounds():
    data1 = np.random.exponential(5, size=(20, 1))
    data2 = np.random.exponential(1, size=(20, 1))
    with pytest.raises(ValueError):
        stats.logrank_test(data1, data2, alpha=95)
