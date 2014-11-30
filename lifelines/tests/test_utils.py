from __future__ import print_function
import numpy as np
import pandas as pd

from pandas.util.testing import assert_frame_equal
import numpy.testing as npt

from ..utils import group_survival_table_from_events, survival_table_from_events, survival_events_from_table, \
                    datetimes_to_durations, k_fold_cross_validation, normalize, unnormalize, significance_code,\
                    qth_survival_time, qth_survival_times
from ..estimation import CoxPHFitter, AalenAdditiveFitter
from ..datasets import load_regression_dataset, load_larynx, load_waltons

def test_unnormalize():
    df = load_larynx()
    m = df.mean(0)
    s = df.std(0)

    ndf = normalize(df)

    npt.assert_almost_equal(df.values, unnormalize(ndf, m, s).values)

def test_normalize():
    df = load_larynx()
    n, d = df.shape
    npt.assert_almost_equal(normalize(df).mean(0).values, np.zeros(d))
    npt.assert_almost_equal(normalize(df).std(0).values, np.ones(d))

def test_qth_survival_times_with_varying_datatype_inputs():
    sf_list = [1.0, 0.75, 0.5, 0.25, 0.0]
    sf_array = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    sf_df_no_index = pd.DataFrame([1.0, 0.75, 0.5, 0.25, 0.0])
    sf_df_index = pd.DataFrame([1.0, 0.75, 0.5, 0.25, 0.0], index=[10, 20, 30, 40, 50])
    sf_series_index = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0], index=[10, 20, 30, 40, 50])
    sf_series_no_index = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0])

    q = 0.5

    assert qth_survival_times(q, sf_list) == 2
    assert qth_survival_times(q, sf_array) == 2
    assert qth_survival_times(q, sf_df_no_index) == 2
    assert qth_survival_times(q, sf_df_index) == 30
    assert qth_survival_times(q, sf_series_index) == 30
    assert qth_survival_times(q, sf_series_no_index) == 2

def test_qth_survival_times_multi_dim_input():
    sf = np.linspace(1, 0, 50)
    sf_multi_df = pd.DataFrame({'sf': sf, 'sf**2': sf ** 2})

    medians = qth_survival_times(0.5, sf_multi_df)
    assert medians.ix['sf'][0.5] == 25
    assert medians.ix['sf**2'][0.5] == 15

def test_qth_survival_time_returns_inf():
    sf = pd.Series([1., 0.7, 0.6])
    assert qth_survival_time(0.5, sf) == np.inf

def test_qth_survival_times_with_multivariate_q():
    sf = np.linspace(1, 0, 50)
    sf_multi_df = pd.DataFrame({'sf': sf, 'sf**2': sf ** 2})

    assert_frame_equal(qth_survival_times([0.2, 0.5], sf_multi_df), pd.DataFrame([[40, 25], [28, 15]], columns=[0.2, 0.5], index=['sf', 'sf**2']))
    assert_frame_equal(qth_survival_times([0.2, 0.5], sf_multi_df['sf']), pd.DataFrame([[40, 25]], columns=[0.2, 0.5], index=['sf']))
    assert_frame_equal(qth_survival_times(0.5, sf_multi_df), pd.DataFrame([[25], [15]], columns=[0.5], index=['sf', 'sf**2']))
    assert qth_survival_times(0.5, sf_multi_df['sf']) == 25

def test_datetimes_to_durations_days():
    start_date = ['2013-10-10 0:00:00', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', '2013-10-10 0:00:00', '2013-10-15']
    T, C = datetimes_to_durations(start_date, end_date)
    npt.assert_almost_equal(T, np.array([3, 1, 5 + 365]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))
    return

def test_datetimes_to_durations_years():
    start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', '2013-10-10', '2013-10-15']
    T, C = datetimes_to_durations(start_date, end_date, freq='Y')
    npt.assert_almost_equal(T, np.array([0, 0, 1]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))
    return

def test_datetimes_to_durations_hours():
    start_date = ['2013-10-10 17:00:00', '2013-10-09 0:00:00', '2013-10-10 23:00:00']
    end_date = ['2013-10-10 18:00:00', '2013-10-10 0:00:00', '2013-10-11 2:00:00']
    T, C = datetimes_to_durations(start_date, end_date, freq='h')
    npt.assert_almost_equal(T, np.array([1, 24, 3]))
    npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))
    return

def test_datetimes_to_durations_censor():
    start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', None, '']
    T, C = datetimes_to_durations(start_date, end_date, freq='Y')
    npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))
    return

def test_datetimes_to_durations_custom_censor():
    start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
    end_date = ['2013-10-13', "NaT", '']
    T, C = datetimes_to_durations(start_date, end_date, freq='Y', na_values="NaT")
    npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))
    return

def test_survival_table_to_events():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([1, 0, 1, 1, 1, 1])
    d = survival_table_from_events(T, C, np.zeros_like(T))
    T_, C_ = survival_events_from_table(d[['censored', 'observed']])
    npt.assert_array_equal(T, T_)
    npt.assert_array_equal(C, C_)

def test_group_survival_table_from_events_on_waltons_data():
    df = load_waltons()
    first_obs = np.zeros(df.shape[0])
    g, removed, observed, censored = group_survival_table_from_events(df['group'], df['T'], df['E'], first_obs)
    assert len(g) == 2
    assert all(removed.columns == ['removed:miR-137', 'removed:control'])
    assert all(removed.index == observed.index)
    assert all(removed.index == censored.index)

def test_survival_table_to_events_casts_to_float():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([True, False, True, True, True, True])
    d = survival_table_from_events(T, C, np.zeros_like(T))
    npt.assert_array_equal(d['censored'].values, np.array([0.,  0.,  1.,  0.,  0.,  0.]))
    npt.assert_array_equal(d['removed'].values, np.array([0.,  1.,  1.,  1.,  2.,  1.]))

def test_group_survival_table_from_events_works_with_series():
    df = pd.DataFrame([[1, True, 3], [1, True, 3], [4, False, 2]], columns=['duration', 'E', 'G'])
    ug, _, _, _ = group_survival_table_from_events(df.G, df.duration, df.E, np.array([[0, 0, 0]]))
    npt.assert_array_equal(ug, np.array([3, 2]))

def test_cross_validator_returns_k_results():
    cf = CoxPHFitter()
    results = k_fold_cross_validation(cf, load_regression_dataset(), duration_col='T', event_col='E', k=3)
    assert len(results) == 3

    results = k_fold_cross_validation(cf, load_regression_dataset(), duration_col='T', event_col='E', k=5)
    assert len(results) == 5

def test_cross_validator_with_predictor():
    cf = CoxPHFitter()
    results = k_fold_cross_validation(cf, load_regression_dataset(),
                                      duration_col='T', event_col='E', k=3,
                                      predictor="predict_expectation")

def test_cross_validator_with_predictor_and_kwargs():
    cf = CoxPHFitter()
    results_06 = k_fold_cross_validation(cf, load_regression_dataset(),
                                         duration_col='T', event_col='E', k=3,
                                         predictor="predict_percentile", predictor_kwargs={'p': 0.6})
