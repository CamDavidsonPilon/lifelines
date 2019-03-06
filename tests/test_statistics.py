# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

from lifelines import statistics as stats
from lifelines import CoxPHFitter
from lifelines.utils import StatisticalWarning
from lifelines.datasets import load_waltons, load_g3, load_lymphoma, load_dd, load_regression_dataset


def test_sample_size_necessary_under_cph():
    assert stats.sample_size_necessary_under_cph(0.8, 1, 0.8, 0.2, 0.139) == (14, 14)
    assert stats.sample_size_necessary_under_cph(0.8, 1, 0.5, 0.5, 1.2) == (950, 950)
    assert stats.sample_size_necessary_under_cph(0.8, 1.5, 0.5, 0.5, 1.2) == (1231, 821)
    assert stats.sample_size_necessary_under_cph(0.8, 1.5, 0.5, 0.5, 1.2, alpha=0.01) == (1832, 1221)


def test_power_under_cph():
    assert abs(stats.power_under_cph(12, 12, 0.8, 0.2, 0.139) - 0.744937) < 10e-6
    assert abs(stats.power_under_cph(12, 20, 0.8, 0.2, 1.2) - 0.05178317) < 10e-6


def test_unequal_intensity_with_random_data():
    data1 = np.random.exponential(5, size=(2000, 1))
    data2 = np.random.exponential(1, size=(2000, 1))
    test_result = stats.logrank_test(data1, data2)
    assert test_result.p_value < 0.05


def test_logrank_test_output_against_R_1():
    df = load_g3()
    ix = df["group"] == "RIT"
    d1, e1 = df.loc[ix]["time"], df.loc[ix]["event"]
    d2, e2 = df.loc[~ix]["time"], df.loc[~ix]["event"]

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


def test_load_lymphoma_logrank():
    # from https://www.statsdirect.com/help/content/survival_analysis/logrank.htm
    df_ = load_lymphoma()
    results = stats.multivariate_logrank_test(df_["Time"], df_["Stage_group"], df_["Censor"])
    assert abs(results.test_statistic - 6.70971) < 1e-4
    assert abs(results.p_value - 0.0096) < 1e-4


def test_multivariate_logrank_on_dd_dataset():
    """
    library('survival')
    dd = read.csv('~/code/lifelines/lifelines/datasets/dd.csv')
    results = survdiff(Surv(duration, observed)~regime, data=dd, rho=0)
    results[5]
    """
    dd = load_dd()
    results = stats.multivariate_logrank_test(dd["duration"], dd["regime"], dd["observed"])
    assert abs(results.test_statistic - 322.5991) < 0.0001


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
    assert result.p_value < 0.05


def test_integer_times_logrank_test():
    data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
    data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
    result = stats.logrank_test(data1, data2)
    assert result.p_value < 0.05


def test_equal_intensity_with_negative_data():
    data1 = np.random.normal(0, size=(2000, 1))
    data1 -= data1.mean()
    data1 /= data1.std()
    data2 = np.random.normal(0, size=(2000, 1))
    data2 -= data2.mean()
    data2 /= data2.std()
    result = stats.logrank_test(data1, data2)
    assert result.p_value > 0.05


def test_unequal_intensity_with_negative_data():
    data1 = np.random.normal(-5, size=(2000, 1))
    data2 = np.random.normal(5, size=(2000, 1))
    result = stats.logrank_test(data1, data2)
    assert result.p_value < 0.05


def test_log_rank_test_on_waltons_dataset():
    df = load_waltons()
    ix = df["group"] == "miR-137"
    waltonT1 = df.loc[ix]["T"]
    waltonT2 = df.loc[~ix]["T"]
    result = stats.logrank_test(waltonT1, waltonT2)
    assert result.p_value < 0.05


def test_logrank_test_is_symmetric():
    data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
    data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
    result1 = stats.logrank_test(data1, data2)
    result2 = stats.logrank_test(data2, data1)
    assert abs(result1.p_value - result2.p_value) < 10e-8


def test_multivariate_unequal_intensities():
    T = np.random.exponential(10, size=300)
    g = np.random.binomial(2, 0.5, size=300)
    T[g == 1] = np.random.exponential(1, size=(g == 1).sum())
    result = stats.multivariate_logrank_test(T, g)
    assert result.p_value < 0.05


def test_pairwise_waltons_dataset_is_significantly_different():
    waltons_dataset = load_waltons()
    R = stats.pairwise_logrank_test(waltons_dataset["T"], waltons_dataset["group"])
    assert R.summary.loc[("control", "miR-137")]["p"] < 0.05


def test_pairwise_allows_dataframes_and_gives_correct_counts():
    N = 100
    N_groups = 5
    df = pd.DataFrame(np.empty((N, 3)), columns=["T", "C", "group"])
    df["T"] = np.random.exponential(1, size=N)
    df["C"] = np.random.binomial(1, 0.6, size=N)
    df["group"] = np.tile(np.arange(N_groups), 20)
    R = stats.pairwise_logrank_test(df["T"], df["group"], event_observed=df["C"])
    assert R.summary.shape[0] == N_groups * (N_groups - 1) / 2


def test_log_rank_returns_None_if_equal_arrays():
    T = np.random.exponential(5, size=200)
    result = stats.logrank_test(T, T)
    assert result.p_value > 0.05

    C = np.random.binomial(1, 0.8, size=200)
    result = stats.logrank_test(T, T, C, C)
    assert result.p_value > 0.05


def test_multivariate_log_rank_is_identital_to_log_rank_for_n_equals_2():
    N = 200
    T1 = np.random.exponential(5, size=N)
    T2 = np.random.exponential(5, size=N)
    C1 = np.random.binomial(1, 0.9, size=N)
    C2 = np.random.binomial(1, 0.9, size=N)
    result = stats.logrank_test(T1, T2, C1, C2)

    T = np.r_[T1, T2]
    C = np.r_[C1, C2]
    G = np.array([1] * 200 + [2] * 200)
    result_m = stats.multivariate_logrank_test(T, G, C)
    assert result.p_value == result_m.p_value


def test_StatisticalResult_kwargs():

    sr = stats.StatisticalResult(0.05, 5.0, kw="some_value")
    assert hasattr(sr, "kw")
    assert getattr(sr, "kw") == "some_value"
    assert "some_value" in sr._to_string()


def test_StatisticalResult_can_be_added():

    sr1 = stats.StatisticalResult(0.01, 1.0, name=["1"], kw1="some_value1")
    sr2 = stats.StatisticalResult([0.02], [2.0], name=["2"], kw2="some_value2")
    sr3 = stats.StatisticalResult([0.03, 0.04], [3.3, 4.4], name=["3", "4"], kw3=3)
    sr = sr1 + sr2 + sr3

    assert sr.summary.shape[0] == 4
    assert sr.summary.index.tolist() == ["1", "2", "3", "4"]
    assert "kw3" in sr._kwargs


def test_proportional_hazard_test():
    """
    c = coxph(formula=Surv(T, E) ~ var1 + var2 + var3, data=df)
    cz = cox.zph(c, transform='rank')
    cz
    """
    cph = CoxPHFitter()
    df = load_regression_dataset()
    cph.fit(df, "T", "E")
    results = stats.proportional_hazard_test(cph, df)
    npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 1.4938293, rtol=1e-3)
    npt.assert_allclose(results.summary.loc["var2"]["test_statistic"], 0.8792998, rtol=1e-3)
    npt.assert_allclose(results.summary.loc["var3"]["test_statistic"], 2.2686088, rtol=1e-3)
    npt.assert_allclose(results.summary.loc["var3"]["p"], 0.1320184, rtol=1e-3)


def test_proportional_hazard_test_with_log_transform():
    cph = CoxPHFitter()
    df = load_regression_dataset()
    cph.fit(df, "T", "E")

    results = stats.proportional_hazard_test(cph, df, time_transform="log")
    npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 2.227627, rtol=1e-3)
    npt.assert_allclose(results.summary.loc["var2"]["test_statistic"], 0.714427, rtol=1e-3)
    npt.assert_allclose(results.summary.loc["var3"]["test_statistic"], 1.466321, rtol=1e-3)
    npt.assert_allclose(results.summary.loc["var3"]["p"], 0.225927, rtol=1e-3)


def test_proportional_hazard_test_with_weights():
    """

    library(survival)
    df <- data.frame(
      "var1" = c(0.209325, 0.693919, 0.443804, 0.065636, 0.386294),
      "T" = c(5.269797, 6.601666, 7.335846, 11.684092, 12.678458),
      "E" = c(1, 1, 1, 1, 1),
      "w" = c(1, 0.5, 2, 1, 1)
    )

    c = coxph(formula=Surv(T, E) ~ var1 , data=df, weights=w)
    cox.zph(c, transform='rank')
    """

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "w": [1, 0.5, 2, 1, 1],
        }
    )
    df["E"] = True

    with pytest.warns(StatisticalWarning, match="weights are not integers"):

        cph = CoxPHFitter()
        cph.fit(df, "T", "E", weights_col="w")

        results = stats.proportional_hazard_test(cph, df)
        npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 0.1083698, rtol=1e-3)


def test_proportional_hazard_test_with_weights_and_strata():
    """
    library(survival)
    df <- data.frame(
      "var1" = c(0.209325, 0.693919, 0.443804, 0.065636, 0.386294),
      "T" = c(5.269797, 6.601666, 7.335846, 11.684092, 12.678458),
      "E" = c(1, 1, 1, 1, 1),
      "w" = c(1, 0.5, 2, 1, 1),
      "s" = c(1, 1, 0, 0, 0)
    )

    c = coxph(formula=Surv(T, E) ~ var1 + strata(s), data=df, weights=w)
    cz = cox.zph(c, transform='identity')

    """

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "w": [1, 0.5, 2, 1, 1],
            "s": [1, 1, 0, 0, 0],
        }
    )
    df["E"] = True

    cph = CoxPHFitter()
    cph.fit(df, "T", "E", weights_col="w", strata="s", robust=True)

    results = stats.proportional_hazard_test(cph, df, time_transform="identity")
    cph.print_summary()

    npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 0.0283, rtol=1e-3)


def test_proportional_hazard_test_with_kmf():
    """

    library(survival)
    df <- data.frame(
      "var1" = c(0.209325, 0.693919, 0.443804, 0.065636, 0.386294),
      "T" = c(5.269797, 6.601666, 7.335846, 11.684092, 12.678458),
      "E" = c(1, 1, 1, 1, 1)
    )

    c = coxph(formula=Surv(T, E) ~ var1 , data=df)
    cox.zph(c, transform='km')
    """

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "E": [1, 1, 1, 1, 1],
        }
    )

    cph = CoxPHFitter()
    cph.fit(df, "T", "E")

    results = stats.proportional_hazard_test(cph, df)
    npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 0.00971, rtol=1e-3)


def test_proportional_hazard_test_with_kmf_with_some_censorship():
    """

    library(survival)
    df <- data.frame(
      "var1" = c(0.209325, 0.693919, 0.443804, 0.065636, 0.386294),
      "T" = c(5.269797, 6.601666, 7.335846, 11.684092, 12.678458),
      "E" = c(1, 1, 1, 0, 1)
    )

    c = coxph(formula=Surv(T, E) ~ var1 , data=df)
    cox.zph(c, transform='km')
    """

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "E": [1, 1, 1, 0, 1],
        }
    )

    cph = CoxPHFitter()
    cph.fit(df, "T", "E")

    results = stats.proportional_hazard_test(cph, df)
    npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 1.013802, rtol=1e-3)


def test_proportional_hazard_test_with_kmf_with_some_censorship_and_weights():
    """

    library(survival)
    df <- data.frame(
      "var1" = c(0.209325, 0.693919, 0.443804, 0.065636, 0.386294),
      "T" = c(5.269797, 6.601666, 7.335846, 11.684092, 12.678458),
      "E" = c(1, 1, 1, 0, 1),
      "w" = c(1, 0.5, 2, 1, 1),
    )

    c = coxph(formula=Surv(T, E) ~ var1 , data=df, weights=w)
    cox.zph(c, transform='km')
    """

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "E": [1, 1, 1, 0, 1],
            "w": [1, 0.5, 5, 1, 1],
        }
    )

    cph = CoxPHFitter()
    with pytest.warns(StatisticalWarning, match="weights are not integers"):
        cph.fit(df, "T", "E", weights_col="w")
        results = stats.proportional_hazard_test(cph, df)
        npt.assert_allclose(results.summary.loc["var1"]["test_statistic"], 0.916, rtol=1e-2)


def test_proportional_hazard_test_with_all():

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "E": [1, 1, 1, 0, 1],
        }
    )

    cph = CoxPHFitter()
    cph.fit(df, "T", "E")
    results = stats.proportional_hazard_test(cph, df, time_transform="all")
    assert results.summary.shape[0] == 1 * 4


def test_proportional_hazard_test_with_list():

    df = pd.DataFrame(
        {
            "var1": [0.209325, 0.693919, 0.443804, 0.065636, 0.386294],
            "var2": [1, 0, 1, 0, 1],
            "T": [5.269797, 6.601666, 7.335846, 11.684092, 12.678458],
            "E": [1, 1, 1, 0, 1],
        }
    )

    cph = CoxPHFitter()
    cph.fit(df, "T", "E")
    results = stats.proportional_hazard_test(cph, df, time_transform=["rank", "km"])
    assert results.summary.shape[0] == 2 * 2
