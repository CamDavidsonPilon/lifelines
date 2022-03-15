# -*- coding: utf-8 -*-


import pytest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy.testing as npt
from numpy.linalg import norm, lstsq
from numpy.random import randn
from flaky import flaky

from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter, ExponentialFitter
from lifelines.datasets import load_regression_dataset, load_larynx, load_waltons, load_rossi
from lifelines import utils
from lifelines import exceptions
from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines.utils.safe_exp import safe_exp


def test_format_p_values():
    assert utils.format_p_value(2)(0.004) == "<0.005"
    assert utils.format_p_value(3)(0.004) == "0.004"

    assert utils.format_p_value(3)(0.000) == "<0.0005"
    assert utils.format_p_value(3)(0.005) == "0.005"
    assert utils.format_p_value(3)(0.2111) == "0.211"
    assert utils.format_p_value(3)(0.2119) == "0.212"


def test_ridge_regression_with_penalty_is_less_than_without_penalty():
    X = randn(2, 2)
    Y = randn(2)
    assert norm(utils.ridge_regression(X, Y, c1=2.0)[0]) <= norm(utils.ridge_regression(X, Y)[0])
    assert norm(utils.ridge_regression(X, Y, c1=1.0, c2=1.0)[0]) <= norm(utils.ridge_regression(X, Y)[0])


def test_ridge_regression_with_extreme_c1_penalty_equals_close_to_zero_vector():
    c1 = 10e8
    c2 = 0.0
    offset = np.ones(2)
    X = randn(2, 2)
    Y = randn(2)
    assert norm(utils.ridge_regression(X, Y, c1, c2, offset)[0]) < 10e-4


def test_ridge_regression_with_extreme_c2_penalty_equals_close_to_offset():
    c1 = 0.0
    c2 = 10e8
    offset = np.ones(2)
    X = randn(2, 2)
    Y = randn(2)
    assert norm(utils.ridge_regression(X, Y, c1, c2, offset)[0] - offset) < 10e-4


def test_lstsq_returns_similar_values_to_ridge_regression():
    X = randn(2, 2)
    Y = randn(2)
    expected = lstsq(X, Y, rcond=None)[0]
    assert norm(utils.ridge_regression(X, Y)[0] - expected) < 10e-4


def test_lstsq_returns_correct_values():
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]
    beta, V = utils.ridge_regression(X, y)
    expected_beta = [-0.98684211, -0.07894737]
    expected_v = [
        [-0.03289474, -0.49342105, 0.06578947, 0.03289474, 0.49342105],
        [-0.30263158, 0.46052632, -0.39473684, 0.30263158, -0.46052632],
    ]
    assert norm(beta - expected_beta) < 10e-4
    for V_row, e_v_row in zip(V, expected_v):
        assert norm(V_row - e_v_row) < 1e-4


def test_unnormalize():
    df = load_larynx()
    m = df.mean(0)
    s = df.std(0)

    ndf = utils.normalize(df)

    npt.assert_almost_equal(df.values, utils.unnormalize(ndf, m, s).values)


def test_normalize():
    df = load_larynx()
    n, d = df.shape
    npt.assert_almost_equal(utils.normalize(df).mean(0).values, np.zeros(d))
    npt.assert_almost_equal(utils.normalize(df).std(0).values, np.ones(d))


def test_median():
    sv = pd.DataFrame(1 - np.linspace(0, 1, 1000))
    assert utils.median_survival_times(sv) == 500


def test_median_accepts_series():
    sv = pd.Series(1 - np.linspace(0, 1, 1000))
    assert utils.median_survival_times(sv) == 500


def test_qth_survival_times_with_varying_datatype_inputs():
    sf_list = [1.0, 0.75, 0.5, 0.25, 0.0]
    sf_array = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    sf_df_no_index = pd.DataFrame([1.0, 0.75, 0.5, 0.25, 0.0])
    sf_df_index = pd.DataFrame([1.0, 0.75, 0.5, 0.25, 0.0], index=[10, 20, 30, 40, 50])
    sf_series_index = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0], index=[10, 20, 30, 40, 50])
    sf_series_no_index = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0])

    q = 0.5

    assert utils.qth_survival_times(q, sf_list) == 2
    assert utils.qth_survival_times(q, sf_array) == 2
    assert utils.qth_survival_times(q, sf_df_no_index) == 2
    assert utils.qth_survival_times(q, sf_df_index) == 30
    assert utils.qth_survival_times(q, sf_series_index) == 30
    assert utils.qth_survival_times(q, sf_series_no_index) == 2


def test_qth_survival_times_multi_dim_input():
    sf = np.linspace(1, 0, 50)
    sf_multi_df = pd.DataFrame({"sf": sf, "sf**2": sf ** 2})
    medians = utils.qth_survival_times(0.5, sf_multi_df)
    assert medians["sf"].loc[0.5] == 25
    assert medians["sf**2"].loc[0.5] == 15


def test_qth_survival_time_returns_inf():
    sf = pd.Series([1.0, 0.7, 0.6])
    assert utils.qth_survival_time(0.5, sf) == np.inf


def test_qth_survival_time_accepts_a_model():
    kmf = KaplanMeierFitter().fit([1.0, 0.7, 0.6])
    assert utils.qth_survival_time(0.8, kmf) > 0


def test_qth_survival_time_with_dataframe():
    sf_df_no_index = pd.DataFrame([1.0, 0.75, 0.5, 0.25, 0.0])
    sf_df_index = pd.DataFrame([1.0, 0.75, 0.5, 0.25, 0.0], index=[10, 20, 30, 40, 50])
    sf_df_too_many_columns = pd.DataFrame([[1, 2], [3, 4]])

    assert utils.qth_survival_time(0.5, sf_df_no_index) == 2
    assert utils.qth_survival_time(0.5, sf_df_index) == 30

    with pytest.raises(ValueError):
        utils.qth_survival_time(0.5, sf_df_too_many_columns)


def test_qth_survival_times_with_multivariate_q():
    sf = np.linspace(1, 0, 50)
    sf_multi_df = pd.DataFrame({"sf": sf, "sf**2": sf ** 2})

    assert_frame_equal(
        utils.qth_survival_times([0.2, 0.5], sf_multi_df),
        pd.DataFrame([[40, 28], [25, 15]], index=[0.2, 0.5], columns=["sf", "sf**2"]),
    )
    assert_frame_equal(
        utils.qth_survival_times([0.2, 0.5], sf_multi_df["sf"]), pd.DataFrame([40, 25], index=[0.2, 0.5], columns=["sf"])
    )
    assert_frame_equal(utils.qth_survival_times(0.5, sf_multi_df), pd.DataFrame([[25, 15]], index=[0.5], columns=["sf", "sf**2"]))
    assert utils.qth_survival_times(0.5, sf_multi_df["sf"]) == 25


def test_qth_survival_times_with_duplicate_q_returns_valid_index_and_shape():
    sf = pd.DataFrame(np.linspace(1, 0, 50))

    q = pd.Series([0.5, 0.5, 0.2, 0.0, 0.0])
    actual = utils.qth_survival_times(q, sf)
    assert actual.shape[0] == len(q)
    assert actual.index[0] == actual.index[1]
    assert_series_equal(actual.iloc[0], actual.iloc[1])

    npt.assert_almost_equal(actual.index.values, q.values)


def test_datetimes_to_durations_with_different_frequencies():
    # days
    start_date = ["2013-10-10 0:00:00", "2013-10-09", "2012-10-10"]
    end_date = ["2013-10-13", "2013-10-10 0:00:00", "2013-10-15"]
    T, C = utils.datetimes_to_durations(start_date, end_date)
    npt.assert_almost_equal(T, np.array([3, 1, 5 + 365]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))

    # years
    start_date = ["2013-10-10", "2013-10-09", "2012-10-10"]
    end_date = ["2013-10-13", "2013-10-10", "2013-10-15"]
    T, C = utils.datetimes_to_durations(start_date, end_date, freq="Y")
    npt.assert_almost_equal(T, np.array([0, 0, 1]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))

    # hours
    start_date = ["2013-10-10 17:00:00", "2013-10-09 0:00:00", "2013-10-10 23:00:00"]
    end_date = ["2013-10-10 18:00:00", "2013-10-10 0:00:00", "2013-10-11 2:00:00"]
    T, C = utils.datetimes_to_durations(start_date, end_date, freq="h")
    npt.assert_almost_equal(T, np.array([1, 24, 3]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))


def test_datetimes_to_durations_will_handle_dates_above_fill_date():
    start_date = ["2013-10-08", "2013-10-09", "2013-10-10"]
    end_date = ["2013-10-10", "2013-10-12", "2013-10-15"]
    T, C = utils.datetimes_to_durations(start_date, end_date, freq="D", fill_date="2013-10-12")
    npt.assert_almost_equal(C, np.array([1, 1, 0], dtype=bool))
    npt.assert_almost_equal(T, np.array([2, 3, 2]))


def test_datetimes_to_durations_will_handle_dates_above_multi_fill_date():
    start_date = ["2013-10-08", "2013-10-09", "2013-10-10"]
    end_date = ["2013-10-10", None, None]
    last_observation = ["2013-10-10", "2013-10-12", "2013-10-14"]
    T, E = utils.datetimes_to_durations(start_date, end_date, freq="D", fill_date=last_observation)
    npt.assert_almost_equal(E, np.array([1, 0, 0], dtype=bool))
    npt.assert_almost_equal(T, np.array([2, 3, 4]))


def test_datetimes_to_durations_will_handle_dates_above_multi_fill_date2():
    start_date = ["2013-10-08", "2013-10-09", "2013-10-10"]
    end_date = ["2013-10-10", None, "2013-10-20"]
    last_observation = ["2013-10-10", "2013-10-12", "2013-10-14"]
    T, E = utils.datetimes_to_durations(start_date, end_date, freq="D", fill_date=last_observation)
    npt.assert_almost_equal(E, np.array([1, 0, 0], dtype=bool))
    npt.assert_almost_equal(T, np.array([2, 3, 4]))


def test_datetimes_to_durations_censor():
    start_date = ["2013-10-10", "2013-10-09", "2012-10-10"]
    end_date = ["2013-10-13", None, ""]
    T, C = utils.datetimes_to_durations(start_date, end_date, freq="Y")
    npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))


def test_datetimes_to_durations_custom_censor():
    start_date = ["2013-10-10", "2013-10-09", "2012-10-10"]
    end_date = ["2013-10-13", "NaT", ""]
    T, C = utils.datetimes_to_durations(start_date, end_date, freq="Y", na_values=["NaT", ""])
    npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))


def test_survival_events_from_table_no_ties():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([1, 0, 1, 1, 0, 1])
    d = utils.survival_table_from_events(T, C)
    T_, C_, W_ = utils.survival_events_from_table(d[["censored", "observed"]])
    npt.assert_array_equal(T, T_)
    npt.assert_array_equal(C, C_)
    npt.assert_array_equal(W_, np.ones_like(T))


def test_survival_events_from_table_with_ties():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([1, 0, 1, 1, 1, 1])
    d = utils.survival_table_from_events(T, C)
    T_, C_, W_ = utils.survival_events_from_table(d[["censored", "observed"]])
    npt.assert_array_equal([1, 2, 3, 4, 5], T_)
    npt.assert_array_equal([1, 0, 1, 1, 1], C_)
    npt.assert_array_equal([1, 1, 1, 2, 1], W_)


def test_survival_table_from_events_with_non_trivial_censorship_column():
    T = np.random.exponential(5, size=50)
    malformed_C = np.random.binomial(2, p=0.8)  # set to 2 on purpose!
    proper_C = malformed_C > 0  # (proper "boolean" array)
    table1 = utils.survival_table_from_events(T, malformed_C, np.zeros_like(T))
    table2 = utils.survival_table_from_events(T, proper_C, np.zeros_like(T))

    assert_frame_equal(table1, table2)


def test_group_survival_table_from_events_on_waltons_data():
    df = load_waltons()
    g, removed, observed, censored = utils.group_survival_table_from_events(df["group"], df["T"], df["E"])
    assert len(g) == 2
    assert all(removed.columns == ["removed:miR-137", "removed:control"])
    assert all(removed.index == observed.index)
    assert all(removed.index == censored.index)


def test_group_survival_table_with_weights():
    df = load_waltons()

    dfw = df.groupby(["T", "E", "group"]).size().reset_index().rename(columns={0: "weights"})
    gw, removedw, observedw, censoredw = utils.group_survival_table_from_events(
        dfw["group"], dfw["T"], dfw["E"], weights=dfw["weights"]
    )
    assert len(gw) == 2
    assert all(removedw.columns == ["removed:miR-137", "removed:control"])
    assert all(removedw.index == observedw.index)
    assert all(removedw.index == censoredw.index)

    g, removed, observed, censored = utils.group_survival_table_from_events(df["group"], df["T"], df["E"])

    assert_frame_equal(removedw, removed)
    assert_frame_equal(observedw, observed)
    assert_frame_equal(censoredw, censored)


def test_survival_table_from_events_binned_with_empty_bin():
    df = load_waltons()
    ix = df["group"] == "miR-137"
    event_table = utils.survival_table_from_events(df.loc[ix]["T"], df.loc[ix]["E"], intervals=[0, 10, 20, 30, 40, 50])
    assert not pd.isnull(event_table).any().any()


def test_survival_table_from_events_at_risk_column():
    df = load_waltons()
    # from R
    expected = [
        163.0,
        162.0,
        160.0,
        157.0,
        154.0,
        152.0,
        151.0,
        148.0,
        144.0,
        139.0,
        134.0,
        133.0,
        130.0,
        128.0,
        126.0,
        119.0,
        118.0,
        108.0,
        107.0,
        99.0,
        96.0,
        89.0,
        87.0,
        69.0,
        65.0,
        49.0,
        38.0,
        36.0,
        27.0,
        24.0,
        14.0,
        1.0,
    ]
    df = utils.survival_table_from_events(df["T"], df["E"])
    assert list(df["at_risk"][1:]) == expected  # skip the first event as that is the birth time, 0.


def test_survival_table_to_events_casts_to_float():
    T, C = (np.array([1, 2, 3, 4, 4, 5]), np.array([True, False, True, True, True, True]))
    d = utils.survival_table_from_events(T, C, np.zeros_like(T))
    npt.assert_array_equal(d["censored"].values, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    npt.assert_array_equal(d["removed"].values, np.array([0.0, 1.0, 1.0, 1.0, 2.0, 1.0]))


def test_group_survival_table_from_events_works_with_series():
    df = pd.DataFrame([[1, True, 3], [1, True, 3], [4, False, 2]], columns=["duration", "E", "G"])
    ug, _, _, _ = utils.group_survival_table_from_events(df.G, df.duration, df.E, np.array([[0, 0, 0]]))
    npt.assert_array_equal(ug, np.array([3, 2]))


def test_survival_table_from_events_will_collapse_if_asked():
    T, C = np.array([1, 3, 4, 5]), np.array([True, True, True, True])
    table = utils.survival_table_from_events(T, C, collapse=True)
    assert table.index.tolist() == [
        pd.Interval(-0.001, 3.5089999999999999, closed="right"),
        pd.Interval(3.5089999999999999, 7.0179999999999998, closed="right"),
    ]


def test_survival_table_from_events_will_collapse_to_desired_bins():
    T, C = np.array([1, 3, 4, 5]), np.array([True, True, True, True])
    table = utils.survival_table_from_events(T, C, collapse=True, intervals=[0, 4, 8])
    assert table.index.tolist() == [pd.Interval(-0.001, 4, closed="right"), pd.Interval(4, 8, closed="right")]


def test_cross_validator_returns_k_results():
    cf = CoxPHFitter()
    results = utils.k_fold_cross_validation(cf, load_regression_dataset(), duration_col="T", event_col="E", k=3)
    assert len(results) == 3

    results = utils.k_fold_cross_validation(cf, load_regression_dataset(), duration_col="T", event_col="E", k=5)
    assert len(results) == 5


def test_cross_validator_returns_fitters_k_results():
    cf = CoxPHFitter()
    fitters = [cf, cf]
    results = utils.k_fold_cross_validation(fitters, load_regression_dataset(), duration_col="T", event_col="E", k=3)
    assert len(results) == 2
    assert len(results[0]) == len(results[1]) == 3

    results = utils.k_fold_cross_validation(fitters, load_regression_dataset(), duration_col="T", event_col="E", k=5)
    assert len(results) == 2
    assert len(results[0]) == len(results[1]) == 5


def test_cross_validator_with_predictor():
    cf = CoxPHFitter()
    results = utils.k_fold_cross_validation(cf, load_regression_dataset(), duration_col="T", event_col="E", k=3)
    assert len(results) == 3


def test_cross_validator_with_stratified_cox_model():
    cf = CoxPHFitter(strata=["race"])
    utils.k_fold_cross_validation(cf, load_rossi(), duration_col="week", event_col="arrest")


def test_cross_validator_with_specific_loss_function():
    cf = CoxPHFitter()
    results_sq = utils.k_fold_cross_validation(
        cf, load_regression_dataset(), scoring_method="concordance_index", duration_col="T", event_col="E"
    )


def test_concordance_index():
    size = 1000
    T = np.random.normal(size=size)
    P = np.random.normal(size=size)
    C = np.random.choice([0, 1], size=size)
    Z = np.zeros_like(T)

    # Zeros is exactly random
    assert utils.concordance_index(T, Z) == 0.5
    assert utils.concordance_index(T, Z, C) == 0.5

    # Itself is 1
    assert utils.concordance_index(T, T) == 1.0
    assert utils.concordance_index(T, T, C) == 1.0

    # Random is close to 0.5
    assert abs(utils.concordance_index(T, P) - 0.5) < 0.05
    assert abs(utils.concordance_index(T, P, C) - 0.5) < 0.05


def test_survival_table_from_events_with_non_negative_T_and_no_lagged_births():
    n = 10
    T = np.arange(n)
    C = [True] * n
    min_obs = [0] * n
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]["entrance"] == n
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_with_negative_T_and_no_lagged_births():
    n = 10
    T = np.arange(-n / 2, n / 2)
    C = [True] * n
    min_obs = None
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]["entrance"] == n
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_with_non_negative_T_and_lagged_births():
    n = 10
    T = np.arange(n)
    C = [True] * n
    min_obs = np.linspace(0, 2, n)
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]["entrance"] == 1
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_with_negative_T_and_lagged_births():
    n = 10
    T = np.arange(-n / 2, n / 2)
    C = [True] * n
    min_obs = np.linspace(-n / 2, 2, n)
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]["entrance"] == 1
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_raises_value_error_if_too_early_births():
    n = 10
    T = np.arange(0, n)
    C = [True] * n
    min_obs = T.copy()
    min_obs[1] = min_obs[1] + 10
    with pytest.raises(ValueError):
        utils.survival_table_from_events(T, C, min_obs)


class TestLongDataFrameUtils(object):
    @pytest.fixture
    def seed_df(self):
        df = pd.DataFrame.from_records([{"id": 1, "var1": 0.1, "T": 10, "E": 1}, {"id": 2, "var1": 0.5, "T": 12, "E": 0}])
        return utils.to_long_format(df, "T")

    @pytest.fixture
    def cv1(self):
        return pd.DataFrame.from_records(
            [
                {"id": 1, "t": 0, "var2": 1.4},
                {"id": 1, "t": 4, "var2": 1.2},
                {"id": 1, "t": 8, "var2": 1.5},
                {"id": 2, "t": 0, "var2": 1.6},
            ]
        )

    @pytest.fixture
    def cv2(self):
        return pd.DataFrame.from_records(
            [{"id": 1, "t": 0, "var3": 0}, {"id": 1, "t": 6, "var3": 1}, {"id": 2, "t": 0, "var3": 0}]
        )

    def test_order_of_adding_covariates_doesnt_matter(self, seed_df, cv1, cv2):
        df12 = seed_df.pipe(utils.add_covariate_to_timeline, cv1, "id", "t", "E").pipe(
            utils.add_covariate_to_timeline, cv2, "id", "t", "E"
        )

        df21 = seed_df.pipe(utils.add_covariate_to_timeline, cv2, "id", "t", "E").pipe(
            utils.add_covariate_to_timeline, cv1, "id", "t", "E"
        )

        assert_frame_equal(df21, df12, check_like=True)

    def test_order_of_adding_covariates_doesnt_matter_in_cumulative_sum(self, seed_df, cv1, cv2):
        df12 = seed_df.pipe(utils.add_covariate_to_timeline, cv1, "id", "t", "E", cumulative_sum=True).pipe(
            utils.add_covariate_to_timeline, cv2, "id", "t", "E", cumulative_sum=True
        )

        df21 = seed_df.pipe(utils.add_covariate_to_timeline, cv2, "id", "t", "E", cumulative_sum=True).pipe(
            utils.add_covariate_to_timeline, cv1, "id", "t", "E", cumulative_sum=True
        )

        assert_frame_equal(df21, df12, check_like=True)

    def test_adding_cvs_with_the_same_column_name_will_insert_appropriately(self, seed_df):
        seed_df = seed_df[seed_df["id"] == 1]

        cv = pd.DataFrame.from_records([{"id": 1, "t": 1, "var1": 1.0}, {"id": 1, "t": 2, "var1": 2.0}])
        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E")
        expected = pd.DataFrame.from_records(
            [
                {"E": False, "id": 1, "stop": 1.0, "start": 0, "var1": 0.1},
                {"E": False, "id": 1, "stop": 2.0, "start": 1, "var1": 1.0},
                {"E": True, "id": 1, "stop": 10.0, "start": 2, "var1": 2.0},
            ]
        )
        assert_frame_equal(df, expected, check_like=True)

    def test_adding_cvs_with_the_same_column_name_will_sum_update_appropriately(self, seed_df):
        seed_df = seed_df[seed_df["id"] == 1]

        new_value_at_time_0 = 1.0
        old_value_at_time_0 = seed_df["var1"].iloc[0]
        cv = pd.DataFrame.from_records([{"id": 1, "t": 0, "var1": new_value_at_time_0, "var2": 2.0}])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E", overwrite=False)

        expected = pd.DataFrame.from_records(
            [{"E": True, "id": 1, "stop": 10.0, "start": 0, "var1": new_value_at_time_0 + old_value_at_time_0, "var2": 2.0}]
        )
        assert_frame_equal(df, expected, check_like=True)

    def test_adding_cvs_with_the_same_column_name_will_overwrite_update_appropriately(self, seed_df):
        seed_df = seed_df[seed_df["id"] == 1]

        new_value_at_time_0 = 1.0
        cv = pd.DataFrame.from_records([{"id": 1, "t": 0, "var1": new_value_at_time_0}])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E", overwrite=True)

        expected = pd.DataFrame.from_records([{"E": True, "id": 1, "stop": 10.0, "start": 0, "var1": new_value_at_time_0}])
        assert_frame_equal(df, expected, check_like=True)

    def test_enum_flag(self, seed_df, cv1, cv2):
        df = seed_df.pipe(utils.add_covariate_to_timeline, cv1, "id", "t", "E", add_enum=True).pipe(
            utils.add_covariate_to_timeline, cv2, "id", "t", "E", add_enum=True
        )

        idx = df["id"] == 1
        n = idx.sum()
        try:
            assert_series_equal(df["enum"].loc[idx], pd.Series(np.arange(1, n + 1)), check_names=False)
        except AssertionError as e:
            # Windows Numpy and Pandas sometimes have int32 or int64 as default dtype
            if os.name == "nt" and "int32" in str(e) and "int64" in str(e):
                assert_series_equal(
                    df["enum"].loc[idx], pd.Series(np.arange(1, n + 1), dtype=df["enum"].loc[idx].dtypes), check_names=False
                )
            else:
                raise e

    def test_event_col_is_properly_inserted(self, seed_df, cv2):
        df = seed_df.pipe(utils.add_covariate_to_timeline, cv2, "id", "t", "E")
        assert df.groupby("id").last()["E"].tolist() == [1, 0]

    def test_redundant_cv_columns_are_dropped(self, seed_df):
        seed_df = seed_df[seed_df["id"] == 1]
        cv = pd.DataFrame.from_records(
            [
                {"id": 1, "t": 0, "var3": 0, "var4": 1},
                {"id": 1, "t": 1, "var3": 0, "var4": 1},  # redundant, as nothing changed during the interval
                {"id": 1, "t": 3, "var3": 0, "var4": 1},  # redundant, as nothing changed during the interval
                {"id": 1, "t": 6, "var3": 1, "var4": 1},
                {"id": 1, "t": 9, "var3": 1, "var4": 1},  # redundant, as nothing changed during the interval
            ]
        )

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E")
        assert df.shape[0] == 2

    def test_will_convert_event_column_to_bools(self, seed_df, cv1):
        seed_df["E"] = seed_df["E"].astype(int)

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv1, "id", "t", "E")
        assert df.dtypes["E"] == bool

    def test_if_cvs_include_a_start_time_after_the_final_time_it_is_excluded(self, seed_df):
        max_T = seed_df["stop"].max()
        cv = pd.DataFrame.from_records(
            [
                {"id": 1, "t": 0, "var3": 0},
                {"id": 1, "t": max_T + 10, "var3": 1},  # will be excluded
                {"id": 2, "t": 0, "var3": 0},
            ]
        )

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E")
        assert df.shape[0] == 2

    def test_if_cvs_include_a_start_time_before_it_is_included(self, seed_df):
        min_T = seed_df["start"].min()
        cv = pd.DataFrame.from_records(
            [{"id": 1, "t": 0, "var3": 0}, {"id": 1, "t": min_T - 1, "var3": 1}, {"id": 2, "t": 0, "var3": 0}]
        )

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E")
        assert df.shape[0] == 3

    def test_cvs_with_null_values_are_dropped(self, seed_df):
        seed_df = seed_df[seed_df["id"] == 1]
        cv = pd.DataFrame.from_records(
            [{"id": None, "t": 0, "var3": 0}, {"id": 1, "t": None, "var3": 1}, {"id": 2, "t": 0, "var3": None}]
        )

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E")
        assert df.shape[0] == 1

    def test_a_new_row_is_not_created_if_start_times_are_the_same(self, seed_df):
        seed_df = seed_df[seed_df["id"] == 1]
        cv1 = pd.DataFrame.from_records([{"id": 1, "t": 0, "var3": 0}, {"id": 1, "t": 5, "var3": 1}])

        cv2 = pd.DataFrame.from_records(
            [{"id": 1, "t": 0, "var4": 0}, {"id": 1, "t": 5, "var4": 1.5}, {"id": 1, "t": 6, "var4": 1.7}]
        )

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv1, "id", "t", "E").pipe(
            utils.add_covariate_to_timeline, cv2, "id", "t", "E"
        )
        assert df.shape[0] == 3

    def test_error_is_raised_if_columns_are_missing_in_seed_df(self, seed_df, cv1):
        del seed_df["start"]
        with pytest.raises(IndexError):
            utils.add_covariate_to_timeline(seed_df, cv1, "id", "t", "E")

    def test_cumulative_sum(self):
        seed_df = pd.DataFrame.from_records([{"id": 1, "start": 0, "stop": 5, "E": 1}])
        cv = pd.DataFrame.from_records([{"id": 1, "t": 0, "var4": 1}, {"id": 1, "t": 1, "var4": 1}, {"id": 1, "t": 3, "var4": 1}])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, "id", "t", "E", cumulative_sum=True)
        expected = pd.DataFrame.from_records(
            [
                {"id": 1, "start": 0, "stop": 1.0, "cumsum_var4": 1, "E": False},
                {"id": 1, "start": 1, "stop": 3.0, "cumsum_var4": 2, "E": False},
                {"id": 1, "start": 3, "stop": 5.0, "cumsum_var4": 3, "E": True},
            ]
        )
        assert_frame_equal(expected, df, check_like=True)

    def test_delay(self, cv2):
        seed_df = pd.DataFrame.from_records([{"id": 1, "start": 0, "stop": 50, "E": 1}])

        cv3 = pd.DataFrame.from_records(
            [{"id": 1, "t": 0, "varA": 2}, {"id": 1, "t": 10, "varA": 4}, {"id": 1, "t": 20, "varA": 6}]
        )

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv3, "id", "t", "E", delay=2).fillna(0)

        expected = pd.DataFrame.from_records(
            [
                {"start": 0, "stop": 2.0, "varA": 0.0, "id": 1, "E": False},
                {"start": 2, "stop": 12.0, "varA": 2.0, "id": 1, "E": False},
                {"start": 12, "stop": 22.0, "varA": 4.0, "id": 1, "E": False},
                {"start": 22, "stop": 50.0, "varA": 6.0, "id": 1, "E": True},
            ]
        )
        assert_frame_equal(expected, df, check_like=True)

    def test_covariates_from_event_matrix_with_simple_addition(self):

        base_df = pd.DataFrame([[1, 0, 5, 1], [2, 0, 4, 1], [3, 0, 8, 1], [4, 0, 4, 1]], columns=["id", "start", "stop", "e"])

        event_df = pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, None]], columns=["id", "poison"])
        cv = utils.covariates_from_event_matrix(event_df, "id")
        ldf = utils.add_covariate_to_timeline(base_df, cv, "id", "duration", "e", cumulative_sum=True)
        assert pd.notnull(ldf).all().all()

        expected = pd.DataFrame(
            [
                (0.0, 0.0, 1.0, 1, False),
                (1.0, 1.0, 5.0, 1, True),
                (0.0, 0.0, 2.0, 2, False),
                (2.0, 1.0, 4.0, 2, True),
                (0.0, 0.0, 3.0, 3, False),
                (3.0, 1.0, 8.0, 3, True),
                (0.0, 0.0, 4.0, 4, True),
            ],
            columns=["start", "cumsum_poison", "stop", "id", "e"],
        )
        assert_frame_equal(expected, ldf, check_dtype=False, check_like=True)

    def test_covariates_from_event_matrix(self):

        base_df = pd.DataFrame([[1, 0, 5, 1], [2, 0, 4, 1], [3, 0, 8, 1], [4, 0, 4, 1]], columns=["id", "start", "stop", "e"])

        event_df = pd.DataFrame(
            [[1, 1, None, 2], [2, None, 5, None], [3, 3, 3, 7]], columns=["id", "promotion", "movement", "raise"]
        )

        cv = utils.covariates_from_event_matrix(event_df, "id")
        ldf = utils.add_covariate_to_timeline(base_df, cv, "id", "duration", "e", cumulative_sum=True)
        expected = pd.DataFrame.from_records(
            [
                {
                    "cumsum_movement": 0.0,
                    "cumsum_promotion": 0.0,
                    "cumsum_raise": 0.0,
                    "e": 0.0,
                    "id": 1.0,
                    "start": 0.0,
                    "stop": 1.0,
                },
                {
                    "cumsum_movement": 0.0,
                    "cumsum_promotion": 1.0,
                    "cumsum_raise": 0.0,
                    "e": 0.0,
                    "id": 1.0,
                    "start": 1.0,
                    "stop": 2.0,
                },
                {
                    "cumsum_movement": 0.0,
                    "cumsum_promotion": 1.0,
                    "cumsum_raise": 1.0,
                    "e": 1.0,
                    "id": 1.0,
                    "start": 2.0,
                    "stop": 5.0,
                },
                {
                    "cumsum_movement": 0.0,
                    "cumsum_promotion": 0.0,
                    "cumsum_raise": 0.0,
                    "e": 1.0,
                    "id": 2.0,
                    "start": 0.0,
                    "stop": 4.0,
                },
                {
                    "cumsum_movement": 0.0,
                    "cumsum_promotion": 0.0,
                    "cumsum_raise": 0.0,
                    "e": 0.0,
                    "id": 3.0,
                    "start": 0.0,
                    "stop": 3.0,
                },
                {
                    "cumsum_movement": 1.0,
                    "cumsum_promotion": 1.0,
                    "cumsum_raise": 0.0,
                    "e": 0.0,
                    "id": 3.0,
                    "start": 3.0,
                    "stop": 7.0,
                },
                {
                    "cumsum_movement": 1.0,
                    "cumsum_promotion": 1.0,
                    "cumsum_raise": 1.0,
                    "e": 1.0,
                    "id": 3.0,
                    "start": 7.0,
                    "stop": 8.0,
                },
                {
                    "cumsum_movement": None,
                    "cumsum_promotion": None,
                    "cumsum_raise": None,
                    "e": 1.0,
                    "id": 4.0,
                    "start": 0.0,
                    "stop": 4.0,
                },
            ]
        )

        assert_frame_equal(expected, ldf, check_dtype=False, check_like=True)

    def test_to_episodic_format_with_long_time_gap_is_identical(self):
        rossi = load_rossi()
        rossi["id"] = np.arange(rossi.shape[0])

        long_rossi = utils.to_episodic_format(rossi, duration_col="week", event_col="arrest", id_col="id", time_gaps=1000.0)

        # using astype(int) would fail on Windows because int32 and int64 are used as dtype
        long_rossi["week"] = long_rossi["stop"].astype(rossi["week"].dtype)
        del long_rossi["start"]
        del long_rossi["stop"]

        assert_frame_equal(long_rossi, rossi, check_like=True)

    def test_to_episodic_format_preserves_outcome(self):
        E = [1, 1, 0, 0]
        df = pd.DataFrame({"T": [1, 3, 1, 3], "E": E, "id": [1, 2, 3, 4]})
        long_df = utils.to_episodic_format(df, "T", "E", id_col="id").sort_values(["id", "stop"])
        assert long_df.shape[0] == 1 + 3 + 1 + 3

        assert long_df.groupby("id").last()["E"].tolist() == E

    def test_to_episodic_format_handles_floating_durations(self):
        df = pd.DataFrame({"T": [0.1, 3.5], "E": [1, 1], "id": [1, 2]})
        long_df = utils.to_episodic_format(df, "T", "E", id_col="id").sort_values(["id", "stop"])
        assert long_df.shape[0] == 1 + 4
        assert long_df["stop"].tolist() == [0.1, 1, 2, 3, 3.5]

    def test_to_episodic_format_handles_floating_durations_with_time_gaps(self):
        df = pd.DataFrame({"T": [0.1, 3.5], "E": [1, 1], "id": [1, 2]})
        long_df = utils.to_episodic_format(df, "T", "E", id_col="id", time_gaps=2.0).sort_values(["id", "stop"])
        assert long_df["stop"].tolist() == [0.1, 2, 3.5]

    def test_to_episodic_format_handles_floating_durations_and_preserves_events(self):
        df = pd.DataFrame({"T": [0.1, 3.5], "E": [1, 0], "id": [1, 2]})
        long_df = utils.to_episodic_format(df, "T", "E", id_col="id", time_gaps=2.0).sort_values(["id", "stop"])
        assert long_df.groupby("id").last()["E"].tolist() == [1, 0]

    def test_to_episodic_format_adds_id_col(self):
        df = pd.DataFrame({"T": [1, 3], "E": [1, 0]})
        long_df = utils.to_episodic_format(df, "T", "E")
        assert "id" in long_df.columns

    def test_to_episodic_format_uses_custom_index_as_id(self):
        df = pd.DataFrame({"T": [1, 3], "E": [1, 0]}, index=["A", "B"])
        long_df = utils.to_episodic_format(df, "T", "E")
        assert long_df["id"].tolist() == ["A", "B", "B", "B"]


class TestStepSizer:
    def test_StepSizer_step_will_decrease_if_unstable(self):
        start = 0.95
        ss = utils.StepSizer(start)
        assert ss.next() == start
        ss.update(1.0)
        ss.update(2.0)
        ss.update(1.0)
        ss.update(2.0)

        assert ss.next() < start

    def test_StepSizer_step_will_increase_if_stable(self):
        start = 0.5
        ss = utils.StepSizer(start)
        assert ss.next() == start
        ss.update(1.0)
        ss.update(0.5)
        ss.update(0.4)
        ss.update(0.1)

        assert ss.next() > start

    def test_StepSizer_step_will_decrease_if_explodes(self):
        start = 0.5
        ss = utils.StepSizer(start)
        assert ss.next() == start
        ss.update(20.0)
        assert ss.next() < start


class TestSklearnAdapter:
    @pytest.fixture
    def X(self):
        return load_regression_dataset().drop("T", axis=1)

    @pytest.fixture
    def Y(self):
        return load_regression_dataset().pop("T")

    def test_model_has_correct_api(self, X, Y):
        base_model = sklearn_adapter(CoxPHFitter, event_col="E")
        cph = base_model()
        assert hasattr(cph, "fit")
        cph.fit(X, Y)
        assert hasattr(cph, "predict")
        cph.predict(X)
        assert hasattr(cph, "score")
        cph.score(X, Y)

    def test_sklearn_cross_val_score_accept_model(self, X, Y):
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        base_model = sklearn_adapter(WeibullAFTFitter, event_col="E")
        wf = base_model(penalizer=1.0)
        assert len(cross_val_score(wf, X, Y, cv=3)) == 3

    def test_sklearn_GridSearchCV_accept_model(self, X, Y):
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        base_model = sklearn_adapter(WeibullAFTFitter, event_col="E")

        grid_params = {"penalizer": 10.0 ** np.arange(-2, 3), "model_ancillary": [True, False]}
        clf = GridSearchCV(base_model(), grid_params, cv=4)
        clf.fit(X, Y)

        assert clf.best_params_ == {"model_ancillary": True, "penalizer": 100.0}
        assert clf.predict(X).shape[0] == X.shape[0]

    def test_model_can_accept_things_like_strata(self, X, Y):
        X["strata"] = np.random.randint(0, 2, size=X.shape[0])
        base_model = sklearn_adapter(CoxPHFitter, event_col="E")
        cph = base_model(strata="strata")
        cph.fit(X, Y)

    def test_we_can_user_other_prediction_methods(self, X, Y):

        base_model = sklearn_adapter(WeibullAFTFitter, event_col="E", predict_method="predict_median")
        wf = base_model(strata="strata")
        wf.fit(X, Y)
        assert wf.predict(X).shape[0] == X.shape[0]

    def test_dill(self, X, Y):
        import dill

        base_model = sklearn_adapter(CoxPHFitter, event_col="E")
        cph = base_model()
        cph.fit(X, Y)

        s = dill.dumps(cph)
        s = dill.loads(s)
        assert cph.predict(X).shape[0] == X.shape[0]

    def test_pickle(self, X, Y):
        import pickle

        base_model = sklearn_adapter(CoxPHFitter, event_col="E")
        cph = base_model()
        cph.fit(X, Y)

        s = pickle.dumps(cph, protocol=-1)
        s = pickle.loads(s)
        assert cph.predict(X).shape[0] == X.shape[0]

    def test_isinstance(self):
        from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin

        base_model = sklearn_adapter(CoxPHFitter, event_col="E")
        assert isinstance(base_model(), BaseEstimator)
        assert isinstance(base_model(), RegressorMixin)
        assert isinstance(base_model(), MetaEstimatorMixin)

    @pytest.mark.xfail
    def test_sklearn_GridSearchCV_accept_model_with_parallelization(self, X, Y):
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        base_model = sklearn_adapter(WeibullAFTFitter, event_col="E")

        grid_params = {"penalizer": 10.0 ** np.arange(-2, 3), "l1_ratio": [0.05, 0.5, 0.95], "model_ancillary": [True, False]}
        # note the n_jobs
        clf = GridSearchCV(base_model(), grid_params, cv=4, n_jobs=-1)
        clf.fit(X, Y)

        assert clf.best_params_ == {"l1_ratio": 0.5, "model_ancillary": False, "penalizer": 0.01}
        assert clf.predict(X).shape[0] == X.shape[0]

    def test_joblib(self, X, Y):
        from joblib import dump, load

        base_model = sklearn_adapter(WeibullAFTFitter, event_col="E")

        clf = base_model()
        clf.fit(X, Y)
        dump(clf, "filename.joblib")
        clf = load("filename.joblib")

    @pytest.mark.xfail
    def test_sklearn_check(self):
        from sklearn.utils.estimator_checks import check_estimator

        base_model = sklearn_adapter(WeibullAFTFitter, event_col="E")
        check_estimator(base_model())


def test_rmst_works_at_kaplan_meier_edge_case():

    T = [1, 2, 3, 4, 10]
    kmf = KaplanMeierFitter().fit(T)

    # when S(t)=0, doesn't matter about extending past
    assert utils.restricted_mean_survival_time(kmf, t=10) == utils.restricted_mean_survival_time(kmf, t=10.001)

    assert utils.restricted_mean_survival_time(kmf, t=9.9) <= utils.restricted_mean_survival_time(kmf, t=10.0)

    assert abs((utils.restricted_mean_survival_time(kmf, t=4) - (1.0 + 0.8 + 0.6 + 0.4))) < 0.0001
    assert abs((utils.restricted_mean_survival_time(kmf, t=4 + 0.1) - (1.0 + 0.8 + 0.6 + 0.4 + 0.2 * 0.1))) < 0.0001


def test_rmst_works_at_kaplan_meier_with_left_censoring():

    T = [5]
    kmf = KaplanMeierFitter().fit_left_censoring(T)

    results = utils.restricted_mean_survival_time(kmf, t=10, return_variance=True)
    assert abs(results[0] - 5) < 0.0001
    assert abs(results[1] - 0) < 0.0001


def test_rmst_exactely_with_known_solution():
    T = np.random.exponential(2, 100)
    exp = ExponentialFitter().fit(T)
    lambda_ = exp.lambda_

    assert abs(utils.restricted_mean_survival_time(exp) - lambda_) < 0.001
    assert abs(utils.restricted_mean_survival_time(exp, t=lambda_) - lambda_ * (np.e - 1) / np.e) < 0.001


@flaky
def test_rmst_approximate_solution():
    T = np.random.exponential(2, 4000)
    exp = ExponentialFitter().fit(T, timeline=np.linspace(0, T.max(), 10000))
    lambda_ = exp.lambda_

    with pytest.warns(exceptions.ApproximationWarning) as w:

        assert (
            abs(
                utils.restricted_mean_survival_time(exp, t=lambda_)
                - utils.restricted_mean_survival_time(exp.survival_function_, t=lambda_)
            )
            < 0.001
        )


def test_rmst_variance():

    T = np.random.exponential(2, 1000)
    expf = ExponentialFitter().fit(T)
    hazard = 1 / expf.lambda_
    t = 1

    sq = 2 / hazard ** 2 * (1 - np.exp(-hazard * t) * (1 + hazard * t))
    actual_mean = 1 / hazard * (1 - np.exp(-hazard * t))
    actual_var = sq - actual_mean ** 2

    assert abs(utils.restricted_mean_survival_time(expf, t=t, return_variance=True)[0] - actual_mean) < 0.001
    assert abs(utils.restricted_mean_survival_time(expf, t=t, return_variance=True)[1] - actual_var) < 0.001


def test_find_best_parametric_model():
    T = np.random.exponential(2, 1000)
    E = np.ones_like(T)

    model, score = utils.find_best_parametric_model(T, E)
    assert True


def test_find_best_parametric_model_can_accept_other_models():
    T = np.random.exponential(2, 1000)
    model, score = utils.find_best_parametric_model(T, additional_models=[ExponentialFitter(), ExponentialFitter()])
    assert True


def test_find_best_parametric_model_with_BIC():
    T = np.random.exponential(2, 1000)
    model, score = utils.find_best_parametric_model(T, scoring_method="BIC")
    assert True


def test_find_best_parametric_model_works_for_left_censoring():
    T = np.random.exponential(2, 100)
    model, score = utils.find_best_parametric_model(T, censoring_type="left", show_progress=True)
    assert True


def test_find_best_parametric_model_works_for_interval_censoring():
    T_1 = np.random.exponential(2, 100)
    T_2 = T_1 + 1
    model, score = utils.find_best_parametric_model((T_1, T_2), censoring_type="interval", show_progress=True)
    assert True


def test_find_best_parametric_model_works_with_weights_and_entry():
    T = np.random.exponential(5, 100)
    W = np.random.randint(1, 5, size=100)
    entry = np.random.exponential(0.01, 100)
    model, score = utils.find_best_parametric_model(T, weights=W, entry=entry, show_progress=True)
    assert True


def test_safe_exp():
    from lifelines.utils.safe_exp import MAX

    assert safe_exp(4.0) == np.exp(4.0)
    assert safe_exp(MAX) == np.exp(MAX)
    assert safe_exp(MAX + 1) == np.exp(MAX)

    from autograd import grad

    assert grad(safe_exp)(4.0) == np.exp(4.0)
    assert grad(safe_exp)(MAX) == np.exp(MAX)
    assert grad(safe_exp)(MAX + 1) == np.exp(MAX)
