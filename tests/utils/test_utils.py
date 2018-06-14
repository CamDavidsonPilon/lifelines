from __future__ import print_function

import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy.testing as npt
from numpy.linalg import norm, lstsq
from numpy.random import randn

from lifelines.estimation import CoxPHFitter
from lifelines.datasets import (load_regression_dataset, load_larynx,
                                load_waltons, load_rossi)
from lifelines import utils
from lifelines.utils import _concordance_index as fast_cindex
from lifelines.utils import _naive_concordance_index as slow_cindex
from lifelines.utils import _BTree as BTree


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
    expected = lstsq(X, Y)[0]
    assert norm(utils.ridge_regression(X, Y)[0] - expected) < 10e-4


def test_lstsq_returns_correct_values():
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-.8, -1.0],
                  [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]
    beta,V = utils.ridge_regression(X, y)
    expected_beta = [-0.98684211, -0.07894737]
    expected_v = [[-0.03289474, -0.49342105, 0.06578947, 0.03289474, 0.49342105],
		  [-0.30263158, 0.46052632, -0.39473684, 0.30263158, -0.46052632]]
    assert(norm(beta - expected_beta) < 10e-4)
    for V_row, e_v_row in zip(V, expected_v):
       assert(norm(V_row - e_v_row) < 1e-4)


def test_l1_log_loss_with_no_observed():
    actual = np.array([1, 1, 1])
    predicted = np.array([1, 1, 1])
    assert utils.l1_log_loss(actual, predicted) == 0.0
    predicted = predicted + 1
    assert utils.l1_log_loss(actual, predicted) == np.log(2)


def test_l1_log_loss_with_observed():
    E = np.array([0, 1, 1])
    actual = np.array([1, 1, 1])
    predicted = np.array([1, 1, 1])
    assert utils.l1_log_loss(actual, predicted, E) == 0.0
    predicted = np.array([2, 1, 1])
    assert utils.l1_log_loss(actual, predicted, E) == 0.0


def test_l2_log_loss_with_no_observed():
    actual = np.array([1, 1, 1])
    predicted = np.array([1, 1, 1])
    assert utils.l2_log_loss(actual, predicted) == 0.0
    predicted = predicted + 1
    assert abs(utils.l2_log_loss(actual, predicted) - np.log(2) ** 2) < 10e-8


def test_l2_log_loss_with_observed():
    E = np.array([0, 1, 1])
    actual = np.array([1, 1, 1])
    predicted = np.array([1, 1, 1])
    assert utils.l2_log_loss(actual, predicted, E) == 0.0
    predicted = np.array([2, 1, 1])
    assert utils.l2_log_loss(actual, predicted, E) == 0.0


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
    sf_multi_df = pd.DataFrame({'sf': sf, 'sf**2': sf ** 2})
    medians = utils.qth_survival_times(0.5, sf_multi_df)
    assert medians['sf'].loc[0.5] == 25
    assert medians['sf**2'].loc[0.5] == 15


def test_qth_survival_time_returns_inf():
    sf = pd.Series([1., 0.7, 0.6])
    assert utils.qth_survival_time(0.5, sf) == np.inf


def test_qth_survival_times_with_multivariate_q():
    sf = np.linspace(1, 0, 50)
    sf_multi_df = pd.DataFrame({'sf': sf, 'sf**2': sf ** 2})

    assert_frame_equal(utils.qth_survival_times([0.2, 0.5], sf_multi_df), pd.DataFrame([[40, 28], [25, 15]], index=[0.2, 0.5], columns=['sf', 'sf**2']))
    assert_frame_equal(utils.qth_survival_times([0.2, 0.5], sf_multi_df['sf']), pd.DataFrame([40, 25], index=[0.2, 0.5], columns=['sf']))
    assert_frame_equal(utils.qth_survival_times(0.5, sf_multi_df), pd.DataFrame([[25, 15]], index=[0.5], columns=['sf', 'sf**2']))
    assert utils.qth_survival_times(0.5, sf_multi_df['sf']) == 25


def test_qth_survival_times_with_duplicate_q_returns_valid_index_and_shape():
    sf = pd.DataFrame(np.linspace(1, 0, 50))

    q = pd.Series([0.5, 0.5, 0.2, 0.0, 0.0])
    actual = utils.qth_survival_times(q, sf)
    assert actual.shape[0] == len(q)
    npt.assert_almost_equal(actual.index.values, q.values)


def test_qth_survival_time_with_cdf_instead_of_survival_function():
    cdf = np.linspace(0, 1, 50)
    assert utils.qth_survival_times(0.5, cdf, cdf=True) == 25


def test_datetimes_to_durations_with_different_frequencies():
    # days
    start_date = ['2013-10-10 0:00:00', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', '2013-10-10 0:00:00', '2013-10-15']
    T, C = utils.datetimes_to_durations(start_date, end_date)
    npt.assert_almost_equal(T, np.array([3, 1, 5 + 365]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))

    # years
    start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', '2013-10-10', '2013-10-15']
    T, C = utils.datetimes_to_durations(start_date, end_date, freq='Y')
    npt.assert_almost_equal(T, np.array([0, 0, 1]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))

    # hours
    start_date = ['2013-10-10 17:00:00', '2013-10-09 0:00:00', '2013-10-10 23:00:00']
    end_date = ['2013-10-10 18:00:00', '2013-10-10 0:00:00', '2013-10-11 2:00:00']
    T, C = utils.datetimes_to_durations(start_date, end_date, freq='h')
    npt.assert_almost_equal(T, np.array([1, 24, 3]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))


def test_datetimes_to_durations_will_handle_dates_above_fill_date():
    start_date = ['2013-10-08', '2013-10-09', '2013-10-10']
    end_date = ['2013-10-10', '2013-10-12', '2013-10-15']
    T, C = utils.datetimes_to_durations(start_date, end_date, freq='Y', fill_date='2013-10-12')
    npt.assert_almost_equal(C, np.array([1, 1, 0], dtype=bool))


def test_datetimes_to_durations_censor():
    start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', None, '']
    T, C = utils.datetimes_to_durations(start_date, end_date, freq='Y')
    npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))


def test_datetimes_to_durations_custom_censor():
    start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', "NaT", '']
    T, C = utils.datetimes_to_durations(start_date, end_date, freq='Y', na_values=["NaT", ""])
    npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))


def test_survival_table_to_events():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([1, 0, 1, 1, 1, 1])
    d = utils.survival_table_from_events(T, C, np.zeros_like(T))
    T_, C_ = utils.survival_events_from_table(d[['censored', 'observed']])
    npt.assert_array_equal(T, T_)
    npt.assert_array_equal(C, C_)


def test_group_survival_table_from_events_on_waltons_data():
    df = load_waltons()
    first_obs = np.zeros(df.shape[0])
    g, removed, observed, censored = utils.group_survival_table_from_events(df['group'], df['T'], df['E'], first_obs)
    assert len(g) == 2
    assert all(removed.columns == ['removed:miR-137', 'removed:control'])
    assert all(removed.index == observed.index)
    assert all(removed.index == censored.index)


def test_survival_table_from_events_at_risk_column():
    df = load_waltons()
    # from R
    expected = [163.0, 162.0, 160.0, 157.0, 154.0, 152.0, 151.0, 148.0, 144.0, 139.0, 134.0, 133.0, 130.0, 128.0, 126.0, 119.0, 118.0,
                108.0, 107.0, 99.0, 96.0, 89.0, 87.0, 69.0, 65.0, 49.0, 38.0, 36.0, 27.0, 24.0, 14.0, 1.0]
    df = utils.survival_table_from_events(df['T'], df['E'])
    assert list(df['at_risk'][1:]) == expected  # skip the first event as that is the birth time, 0.


def test_survival_table_to_events_casts_to_float():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([True, False, True, True, True, True])
    d = utils.survival_table_from_events(T, C, np.zeros_like(T))
    npt.assert_array_equal(d['censored'].values, np.array([0.,  0.,  1.,  0.,  0.,  0.]))
    npt.assert_array_equal(d['removed'].values, np.array([0.,  1.,  1.,  1.,  2.,  1.]))


def test_group_survival_table_from_events_works_with_series():
    df = pd.DataFrame([[1, True, 3], [1, True, 3], [4, False, 2]], columns=['duration', 'E', 'G'])
    ug, _, _, _ = utils.group_survival_table_from_events(df.G, df.duration, df.E, np.array([[0, 0, 0]]))
    npt.assert_array_equal(ug, np.array([3, 2]))


def test_survival_table_from_events_will_collapse_if_asked():
    T, C = np.array([1, 3, 4, 5]), np.array([True, True, True, True])
    table = utils.survival_table_from_events(T, C, collapse=True)
    assert table.index.tolist() == [pd.Interval(0, 3.5089999999999999, closed='right'), pd.Interval(3.5089999999999999,  7.0179999999999998, closed='right')]


def test_survival_table_from_events_will_collapse_to_desired_bins():
    T, C = np.array([1, 3, 4, 5]), np.array([True, True, True, True])
    table = utils.survival_table_from_events(T, C, collapse=True, intervals=[0, 4, 8])
    assert table.index.tolist() == [pd.Interval(0, 4, closed='right'), pd.Interval(4,  8, closed='right')]


def test_cross_validator_returns_k_results():
    cf = CoxPHFitter()
    results = utils.k_fold_cross_validation(cf, load_regression_dataset(), duration_col='T', event_col='E', k=3)
    assert len(results) == 3

    results = utils.k_fold_cross_validation(cf, load_regression_dataset(), duration_col='T', event_col='E', k=5)
    assert len(results) == 5


def test_cross_validator_returns_fitters_k_results():
    cf = CoxPHFitter()
    fitters = [cf, cf]
    results = utils.k_fold_cross_validation(fitters, load_regression_dataset(), duration_col='T', event_col='E', k=3)
    assert len(results) == 2
    assert len(results[0]) == len(results[1]) == 3

    results = utils.k_fold_cross_validation(fitters, load_regression_dataset(), duration_col='T', event_col='E', k=5)
    assert len(results) == 2
    assert len(results[0]) == len(results[1]) == 5


def test_cross_validator_with_predictor():
    cf = CoxPHFitter()
    results = utils.k_fold_cross_validation(cf, load_regression_dataset(),
                                            duration_col='T', event_col='E', k=3,
                                            predictor="predict_expectation")
    assert len(results) == 3


def test_cross_validator_with_predictor_and_kwargs():
    cf = CoxPHFitter()
    results_06 = utils.k_fold_cross_validation(cf, load_regression_dataset(),
                                               duration_col='T', k=3,
                                               predictor="predict_percentile", predictor_kwargs={'p': 0.6})
    assert len(results_06) == 3


def test_cross_validator_with_stratified_cox_model():
    cf = CoxPHFitter(strata=['race'])
    utils.k_fold_cross_validation(cf, load_rossi(), duration_col='week', event_col='arrest')


def test_cross_validator_with_specific_loss_function():
    def square_loss(y_actual, y_pred):
        return ((y_actual - y_pred) ** 2).mean()

    cf = CoxPHFitter()
    results_sq = utils.k_fold_cross_validation(cf, load_regression_dataset(), evaluation_measure=square_loss,
                                               duration_col='T', event_col='E')
    results_con = utils.k_fold_cross_validation(cf, load_regression_dataset(), duration_col='T', event_col='E')
    assert list(results_sq) != list(results_con)


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


def test_concordance_index_returns_same_after_shifting():
    T = np.array([1, 2, 3, 4, 5, 6])
    T_ = np.array([2, 1, 4, 6, 5, 3])
    assert utils.concordance_index(T, T_) == utils.concordance_index(T - 5, T_ - 5) == utils.concordance_index(T, T_ - 5) == utils.concordance_index(T - 5, T_)


def test_both_concordance_index_function_deal_with_ties_the_same_way():
    actual_times = np.array([1, 1, 2])
    predicted_times = np.array([1, 2, 3])
    obs = np.ones(3)
    assert fast_cindex(actual_times, predicted_times, obs) == slow_cindex(actual_times, predicted_times, obs) == 1.0


def test_concordance_index_function_exits():
    N = 10 * 1000
    actual_times = np.random.exponential(1, size=N)
    predicted_times = np.random.exponential(1, size=N)
    obs = np.ones(N)
    assert fast_cindex(actual_times, predicted_times, obs)


def test_survival_table_from_events_with_non_negative_T_and_no_lagged_births():
    n = 10
    T = np.arange(n)
    C = [True] * n
    min_obs = [0] * n
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]['entrance'] == n
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_with_negative_T_and_no_lagged_births():
    n = 10
    T = np.arange(-n / 2, n / 2)
    C = [True] * n
    min_obs = None
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]['entrance'] == n
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_with_non_negative_T_and_lagged_births():
    n = 10
    T = np.arange(n)
    C = [True] * n
    min_obs = np.linspace(0, 2, n)
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]['entrance'] == 1
    assert df.index[0] == T.min()
    assert df.index[-1] == T.max()


def test_survival_table_from_events_with_negative_T_and_lagged_births():
    n = 10
    T = np.arange(-n / 2, n / 2)
    C = [True] * n
    min_obs = np.linspace(-n / 2, 2, n)
    df = utils.survival_table_from_events(T, C, min_obs)
    assert df.iloc[0]['entrance'] == 1
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


def test_btree():
    t = BTree(np.arange(10))
    for i in range(10):
        assert t.rank(i) == (0, 0)

    assert len(t) == 0
    t.insert(5)
    t.insert(6)
    t.insert(6)
    t.insert(0)
    t.insert(9)
    assert len(t) == 5

    assert t.rank(0) == (0, 1)
    assert t.rank(0.5) == (1, 0)
    assert t.rank(4.5) == (1, 0)
    assert t.rank(5) == (1, 1)
    assert t.rank(5.5) == (2, 0)
    assert t.rank(6) == (2, 2)
    assert t.rank(6.5) == (4, 0)
    assert t.rank(8.5) == (4, 0)
    assert t.rank(9) == (4, 1)
    assert t.rank(9.5) == (5, 0)

    for i in range(1, 32):
        BTree(np.arange(i))

    with pytest.raises(ValueError):
        # This has to go last since it screws up the counts
        t.insert(5.5)


def test_concordance_index_fast_is_same_as_slow():
    size = 100
    T = np.random.normal(size=size)
    P = np.random.normal(size=size)
    C = np.random.choice([0, 1], size=size)
    Z = np.zeros_like(T)

    # Hard to imagine these failing
    assert slow_cindex(T, Z, C) == fast_cindex(T, Z, C)
    assert slow_cindex(T, T, C) == fast_cindex(T, T, C)
    # This is the real test though
    assert slow_cindex(T, P, C) == fast_cindex(T, P, C)

    cp = CoxPHFitter()
    df = load_rossi()
    cp.fit(df, duration_col='week', event_col='arrest')

    T = cp.durations.values.ravel()
    P = -cp.predict_partial_hazard(df[df.columns.difference(["week", "arrest"])]).values.ravel()

    E = cp.event_observed.values.ravel()

    assert slow_cindex(T, P, E) == fast_cindex(T, P, E)


class TestLongDataFrameUtils(object):

    @pytest.fixture
    def seed_df(self):
        df = pd.DataFrame.from_records([
            {'id': 1, 'var1': 0.1, 'T': 10, 'E': 1},
            {'id': 2, 'var1': 0.5, 'T': 12, 'E': 0}
        ])
        return utils.to_long_format(df, 'T')

    @pytest.fixture
    def cv1(self):
        return pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var2': 1.4},
            {'id': 1, 't': 4, 'var2': 1.2},
            {'id': 1, 't': 8, 'var2': 1.5},
            {'id': 2, 't': 0, 'var2': 1.6},
        ])

    @pytest.fixture
    def cv2(self):
        return pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var3': 0},
            {'id': 1, 't': 6, 'var3': 1},
            {'id': 2, 't': 0, 'var3': 0},
        ])

    def test_order_of_adding_covariates_doesnt_matter(self, seed_df, cv1, cv2):
        df12 = seed_df.pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E')\
                      .pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E')

        df21 = seed_df.pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E')\
                      .pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E')

        assert_frame_equal(df21, df12, check_like=True)

    def test_order_of_adding_covariates_doesnt_matter_in_cumulative_sum(self, seed_df, cv1, cv2):
        df12 = seed_df.pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E', cumulative_sum=True)\
                      .pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E', cumulative_sum=True)

        df21 = seed_df.pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E', cumulative_sum=True)\
                      .pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E', cumulative_sum=True)

        assert_frame_equal(df21, df12, check_like=True)

    def test_adding_cvs_with_the_same_column_name_will_insert_appropriately(self, seed_df):
        seed_df = seed_df[seed_df['id'] == 1]

        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 1, 'var1': 1.0},
            {'id': 1, 't': 2, 'var1': 2.0},
        ])
        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E')
        expected = pd.DataFrame.from_records([
            {'E': False, 'id': 1, 'stop': 1.0, 'start': 0, 'var1': 0.1},
            {'E': False, 'id': 1, 'stop': 2.0, 'start': 1,  'var1': 1.0},
            {'E': True, 'id': 1, 'stop': 10.0, 'start': 2, 'var1': 2.0},
        ])
        assert_frame_equal(df, expected, check_like=True)

    def test_adding_cvs_with_the_same_column_name_will_sum_update_appropriately(self, seed_df):
        seed_df = seed_df[seed_df['id'] == 1]

        new_value_at_time_0 = 1.0
        old_value_at_time_0 = seed_df['var1'].iloc[0]
        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var1': new_value_at_time_0, 'var2': 2.0},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E', overwrite=False)

        expected = pd.DataFrame.from_records([
            {'E': True, 'id': 1, 'stop': 10.0, 'start': 0, 'var1': new_value_at_time_0 + old_value_at_time_0, 'var2': 2.0},
        ])
        assert_frame_equal(df, expected, check_like=True)

    def test_adding_cvs_with_the_same_column_name_will_overwrite_update_appropriately(self, seed_df):
        seed_df = seed_df[seed_df['id'] == 1]

        new_value_at_time_0 = 1.0
        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var1': new_value_at_time_0},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E', overwrite=True)

        expected = pd.DataFrame.from_records([
            {'E': True, 'id': 1, 'stop': 10.0, 'start': 0, 'var1': new_value_at_time_0},
        ])
        assert_frame_equal(df, expected, check_like=True)

    def test_enum_flag(self, seed_df, cv1, cv2):
        df = seed_df.pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E', add_enum=True)\
                    .pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E', add_enum=True)

        idx = df['id'] == 1
        n = idx.sum()
        assert_series_equal(df['enum'].loc[idx], pd.Series(np.arange(1, n + 1)), check_names=False)

    def test_event_col_is_properly_inserted(self, seed_df, cv2):
        df = seed_df.pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E')
        assert df.groupby('id').last()['E'].tolist() == [1, 0]

    def test_redundant_cv_columns_are_dropped(self, seed_df):
        seed_df = seed_df[seed_df['id'] == 1]
        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var3': 0, 'var4': 1},
            {'id': 1, 't': 1, 'var3': 0, 'var4': 1},  # redundant, as nothing changed during the interval
            {'id': 1, 't': 3, 'var3': 0, 'var4': 1},  # redundant, as nothing changed during the interval
            {'id': 1, 't': 6, 'var3': 1, 'var4': 1},
            {'id': 1, 't': 9, 'var3': 1, 'var4': 1},  # redundant, as nothing changed during the interval
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E')
        assert df.shape[0] == 2

    def test_will_convert_event_column_to_bools(self, seed_df, cv1):
        seed_df['E'] = seed_df['E'].astype(int)

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E')
        assert df.dtypes['E'] == bool

    def test_if_cvs_include_a_start_time_after_the_final_time_it_is_excluded(self, seed_df):
        max_T = seed_df['stop'].max()
        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var3': 0},
            {'id': 1, 't': max_T + 10, 'var3': 1},  # will be excluded
            {'id': 2, 't': 0, 'var3': 0},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E')
        assert df.shape[0] == 2

    def test_if_cvs_include_a_start_time_before_it_is_included(self, seed_df):
        min_T = seed_df['start'].min()
        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var3': 0},
            {'id': 1, 't': min_T - 1, 'var3': 1},
            {'id': 2, 't': 0, 'var3': 0},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E')
        assert df.shape[0] == 3

    def test_cvs_with_null_values_are_dropped(self, seed_df):
        seed_df = seed_df[seed_df['id'] == 1]
        cv = pd.DataFrame.from_records([
            {'id': None, 't': 0, 'var3': 0},
            {'id': 1, 't': None, 'var3': 1},
            {'id': 2, 't': 0, 'var3': None},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E')
        assert df.shape[0] == 1

    def test_a_new_row_is_not_created_if_start_times_are_the_same(self, seed_df):
        seed_df = seed_df[seed_df['id'] == 1]
        cv1 = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var3': 0},
            {'id': 1, 't': 5, 'var3': 1},
        ])

        cv2 = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var4': 0},
            {'id': 1, 't': 5, 'var4': 1.5},
            {'id': 1, 't': 6, 'var4': 1.7},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv1, 'id', 't', 'E')\
                    .pipe(utils.add_covariate_to_timeline, cv2, 'id', 't', 'E')
        assert df.shape[0] == 3

    def test_error_is_raised_if_columns_are_missing_in_seed_df(self, seed_df, cv1):
        del seed_df['start']
        with pytest.raises(IndexError):
            utils.add_covariate_to_timeline(seed_df, cv1, 'id', 't', 'E')

    def test_cumulative_sum(self):
        seed_df = pd.DataFrame.from_records([{'id': 1, 'start': 0, 'stop': 5, 'E': 1}])
        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var4': 1},
            {'id': 1, 't': 1, 'var4': 1},
            {'id': 1, 't': 3, 'var4': 1},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv, 'id', 't', 'E', cumulative_sum=True)
        expected = pd.DataFrame.from_records([
            {'id': 1, 'start': 0, 'stop': 1.0, 'cumsum_var4': 1, 'E': False},
            {'id': 1, 'start': 1, 'stop': 3.0, 'cumsum_var4': 2, 'E': False},
            {'id': 1, 'start': 3, 'stop': 5.0, 'cumsum_var4': 3, 'E': True},
        ])
        assert_frame_equal(expected, df, check_like=True)


    def test_delay(self, cv2):
        seed_df = pd.DataFrame.from_records([{'id': 1, 'start': 0, 'stop': 50, 'E': 1}])

        cv3 = pd.DataFrame.from_records([
            {'id': 1, 't': 0,  'varA': 2},
            {'id': 1, 't': 10, 'varA': 4},
            {'id': 1, 't': 20, 'varA': 6},
        ])

        df = seed_df.pipe(utils.add_covariate_to_timeline, cv3, 'id', 't', 'E', delay=2).fillna(0)

        expected = pd.DataFrame.from_records([
            {'start': 0,   'stop': 2.0,  'varA': 0.0,'id': 1, 'E': False},
            {'start': 2,   'stop': 12.0, 'varA': 2.0, 'id': 1, 'E': False},
            {'start': 12,  'stop': 22.0, 'varA': 4.0, 'id': 1, 'E': False},
            {'start': 22,  'stop': 50.0, 'varA': 6.0, 'id': 1, 'E': True}]
        )
        assert_frame_equal(expected, df, check_like=True)


    def test_covariates_from_event_matrix(self):
        df = pd.DataFrame([
                [1,   1,    None, 2   ],
                [2,   None, 5,    None],
                [3,   3,    3,    7   ]
             ], columns=['id', 'promotion', 'movement', 'raise'])

        ldf = pd.DataFrame([[1, 0, 5, 1], [2, 0, 4, 1], [3, 0, 8, 1], [4, 0, 4, 1]], columns=['id', 'start', 'stop', 'e'])

        cv = utils.covariates_from_event_matrix(df, 'id')
        ldf = utils.add_covariate_to_timeline(ldf, cv, 'id', 'duration', 'e', cumulative_sum=True)
        assert ldf.loc[ldf['id'] == 1]['cumsum_movement'].tolist() == [0, 0, 0]
        assert ldf.loc[ldf['id'] == 1]['cumsum_promotion'].tolist() == [0, 1, 1]
        assert ldf.loc[ldf['id'] == 1]['cumsum_raise'].tolist() == [0, 0, 1]
        assert ldf.loc[ldf['id'] == 1]['start'].tolist() == [0, 1., 2.]


def test_StepSizer_step_will_decrease_if_unstable():
    start = 0.95
    ss = utils.StepSizer(start)
    assert ss.next() == start
    ss.update(1.0)
    ss.update(2.0)
    ss.update(1.0)
    ss.update(2.0)

    assert ss.next() < start


def test_StepSizer_step_will_increase_if_stable():
    start = 0.5
    ss = utils.StepSizer(start)
    assert ss.next() == start
    ss.update(1.0)
    ss.update(0.5)
    ss.update(0.4)
    ss.update(0.1)

    assert ss.next() > start


def test_StepSizer_step_will_decrease_if_explodes():
    start = 0.5
    ss = utils.StepSizer(start)
    assert ss.next() == start
    ss.update(20.0)
    assert ss.next() < start
