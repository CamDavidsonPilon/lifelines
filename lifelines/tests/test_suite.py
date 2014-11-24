# -*- coding: utf-8 -*-
"""
python -m lifelines.tests.test_suit

"""
from __future__ import print_function

import os
import unittest
from collections import Counter
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
import numpy.testing as npt
from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd
from pandas.util.testing import assert_frame_equal

from ..estimation import KaplanMeierFitter, NelsonAalenFitter, AalenAdditiveFitter, \
    median_survival_times, BreslowFlemingHarringtonFitter, BayesianFitter, \
    CoxPHFitter, qth_survival_times, qth_survival_time

from ..statistics import (logrank_test, multivariate_logrank_test,
                          pairwise_logrank_test, concordance_index)
from ..generate_datasets import *
from ..plotting import plot_lifetimes
from ..utils import *
from ..datasets import generate_lcd_dataset, generate_rossi_dataset, \
    generate_waltons_dataset, generate_regression_dataset


class MiscTests(unittest.TestCase):

    def test_qth_survival_times_with_varying_datatype_inputs(self):
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

    def test_qth_survival_times_multi_dim_input(self):
        sf = np.linspace(1, 0, 50)
        sf_multi_df = pd.DataFrame({'sf': sf, 'sf**2': sf ** 2})

        medians = qth_survival_times(0.5, sf_multi_df)
        assert medians.ix['sf'][0.5] == 25
        assert medians.ix['sf**2'][0.5] == 15

    def test_qth_survival_time_returns_inf(self):
        sf = pd.Series([1., 0.7, 0.6])
        assert qth_survival_time(0.5, sf) == np.inf

    def test_qth_survival_times_with_multivariate_q(self):
        sf = np.linspace(1, 0, 50)
        sf_multi_df = pd.DataFrame({'sf': sf, 'sf**2': sf ** 2})

        assert_frame_equal(qth_survival_times([0.2, 0.5], sf_multi_df), pd.DataFrame([[40, 25], [28, 15]], columns=[0.2, 0.5], index=['sf', 'sf**2']))
        assert_frame_equal(qth_survival_times([0.2, 0.5], sf_multi_df['sf']), pd.DataFrame([[40, 25]], columns=[0.2, 0.5], index=['sf']))
        assert_frame_equal(qth_survival_times(0.5, sf_multi_df), pd.DataFrame([[25], [15]], columns=[0.5], index=['sf', 'sf**2']))
        assert qth_survival_times(0.5, sf_multi_df['sf']) == 25

    def test_datetimes_to_durations_days(self):
        start_date = ['2013-10-10 0:00:00', '2013-10-09', '2012-10-10']
        end_date = ['2013-10-13', '2013-10-10 0:00:00', '2013-10-15']
        T, C = datetimes_to_durations(start_date, end_date)
        npt.assert_almost_equal(T, np.array([3, 1, 5 + 365]))
        npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))
        return

    def test_datetimes_to_durations_years(self):
        start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
        end_date = ['2013-10-13', '2013-10-10', '2013-10-15']
        T, C = datetimes_to_durations(start_date, end_date, freq='Y')
        npt.assert_almost_equal(T, np.array([0, 0, 1]))
        npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))
        return

    def test_datetimes_to_durations_hours(self):
        start_date = ['2013-10-10 17:00:00', '2013-10-09 0:00:00', '2013-10-10 23:00:00']
        end_date = ['2013-10-10 18:00:00', '2013-10-10 0:00:00', '2013-10-11 2:00:00']
        T, C = datetimes_to_durations(start_date, end_date, freq='h')
        npt.assert_almost_equal(T, np.array([1, 24, 3]))
        npt.assert_almost_equal(C, np.array([1, 1, 1], dtype=bool))
        return

    def test_datetimes_to_durations_censor(self):
        start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
        end_date = ['2013-10-13', None, '']
        T, C = datetimes_to_durations(start_date, end_date, freq='Y')
        npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))
        return

    def test_datetimes_to_durations_custom_censor(self):
        start_date = ['2013-10-10', '2013-10-09', '2012-10-10']
        end_date = ['2013-10-13', "NaT", '']
        T, C = datetimes_to_durations(start_date, end_date, freq='Y', na_values="NaT")
        npt.assert_almost_equal(C, np.array([1, 0, 0], dtype=bool))
        return

    def test_survival_table_to_events(self):
        T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([1, 0, 1, 1, 1, 1])
        d = survival_table_from_events(T, C, np.zeros_like(T))
        T_, C_ = survival_events_from_table(d[['censored', 'observed']])
        npt.assert_array_equal(T, T_)
        npt.assert_array_equal(C, C_)

    def test_ci_labels(self):
        naf = NelsonAalenFitter()
        expected = ['upper', 'lower']
        naf.fit(LIFETIMES, ci_labels=expected)
        npt.assert_array_equal(naf.confidence_interval_.columns, expected)
        kmf = KaplanMeierFitter()
        kmf.fit(LIFETIMES, ci_labels=expected)
        npt.assert_array_equal(kmf.confidence_interval_.columns, expected)

    def test_group_survival_table_from_events_works_with_series(self):
        df = pd.DataFrame([[1, True, 3], [1, True, 3], [4, False, 2]], columns=['duration', 'E', 'G'])
        ug, _, _, _ = group_survival_table_from_events(df.G, df.duration, df.E, np.array([[0, 0, 0]]))
        npt.assert_array_equal(ug, np.array([3, 2]))

    def test_cross_validator_returns_k_results(self):
        cf = CoxPHFitter()
        results = k_fold_cross_validation(cf, generate_regression_dataset(), duration_col='T', event_col='E', k=3)
        self.assertTrue(len(results) == 3)

        aaf = AalenAdditiveFitter()
        results = k_fold_cross_validation(cf, generate_regression_dataset(), duration_col='T', event_col='E', k=5)
        self.assertTrue(len(results) == 5)

    def test_cross_validator_with_predictor(self):
        cf = CoxPHFitter()
        results = k_fold_cross_validation(cf, generate_regression_dataset(),
                                          duration_col='T', event_col='E', k=3,
                                          predictor="predict_expectation")

    def test_cross_validator_with_predictor_and_kwargs(self):
        cf = CoxPHFitter()
        results_06 = k_fold_cross_validation(cf, generate_regression_dataset(),
                                             duration_col='T', event_col='E', k=3,
                                             predictor="predict_percentile", predictor_kwargs={'p': 0.6})

    def test_label_is_a_property(self):
        kmf = KaplanMeierFitter()
        kmf.fit(LIFETIMES, label='Test Name')
        assert kmf._label == 'Test Name'
        assert kmf.confidence_interval_.columns[0] == 'Test Name_upper_0.95'
        assert kmf.confidence_interval_.columns[1] == 'Test Name_lower_0.95'

        naf = NelsonAalenFitter()
        naf.fit(LIFETIMES, label='Test Name')
        assert naf._label == 'Test Name'
        assert naf.confidence_interval_.columns[0] == 'Test Name_upper_0.95'
        assert naf.confidence_interval_.columns[1] == 'Test Name_lower_0.95'


class StatisticalTests(unittest.TestCase):

    def setUp(self):
        self.lifetimes = Counter(LIFETIMES)
        self.km = self.kaplan_meier()
        self.kmc = self.kaplan_meier(censor=True)
        self.na = self.nelson_aalen()
        self.nac = self.nelson_aalen(censor=True)

    def test_kaplan_meier(self):
        kmf = KaplanMeierFitter()
        kmf.fit(LIFETIMES)
        npt.assert_almost_equal(kmf.survival_function_.values, self.km)

    def test_nelson_aalen(self):
        naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
        naf.fit(LIFETIMES)
        npt.assert_almost_equal(naf.cumulative_hazard_.values, self.na)

    def test_censor_nelson_aalen(self):
        naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
        naf.fit(LIFETIMES, event_observed=OBSERVED)
        npt.assert_almost_equal(naf.cumulative_hazard_.values, self.nac)

    def test_censor_kaplan_meier(self):
        kmf = KaplanMeierFitter()
        kmf.fit(LIFETIMES, event_observed=OBSERVED)
        npt.assert_almost_equal(kmf.survival_function_.values, self.kmc)

    def test_median(self):
        sv = pd.DataFrame(1 - np.linspace(0, 1, 1000))
        self.assertTrue(median_survival_times(sv) == 500)

    def test_not_to_break(self):
        try:
            naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
            naf.fit(LIFETIMES)
            naf = NelsonAalenFitter(nelson_aalen_smoothing=True)
            naf.fit(LIFETIMES)
            self.assertTrue(True)
        except Exception as e:
            print(e)
            self.assertTrue(False)

    def test_equal_intensity(self):
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
        self.assertTrue(bounds[0] < 1 - alpha, bounds[0])

    def test_unequal_intensity(self):
        data1 = np.random.exponential(5, size=(2000, 1))
        data2 = np.random.exponential(1, size=(2000, 1))
        summary, p_value, result = logrank_test(data1, data2)
        self.assertTrue(result)

    def test_unequal_intensity_event_observed(self):
        data1 = np.random.exponential(5, size=(2000, 1))
        data2 = np.random.exponential(1, size=(2000, 1))
        eventA = np.random.binomial(1, 0.5, size=(2000, 1))
        eventB = np.random.binomial(1, 0.5, size=(2000, 1))
        summary, p_value, result = logrank_test(data1, data2, event_observed_A=eventA, event_observed_B=eventB)
        self.assertTrue(result)

    def test_integer_times_logrank_test(self):
        data1 = np.random.exponential(5, size=(2000, 1)).astype(int)
        data2 = np.random.exponential(1, size=(2000, 1)).astype(int)
        summary, p_value, result = logrank_test(data1, data2)
        self.assertTrue(result)

    def test_waltons_dataset(self):
        summary, p_value, result = logrank_test(waltonT1, waltonT2)
        self.assertTrue(result)

    def test_smoothing_hazard_ties(self):
        T = np.random.binomial(20, 0.7, size=300)
        C = np.random.binomial(1, 0.8, size=300)
        naf = NelsonAalenFitter()
        naf.fit(T, C)
        naf.smoothed_hazard_(1.)
        self.assertTrue(True)

    def test_smoothing_hazard_nontied(self):
        T = np.random.exponential(20, size=300) ** 2
        C = np.random.binomial(1, 0.8, size=300)
        naf = NelsonAalenFitter()
        naf.fit(T, C)
        naf.smoothed_hazard_(1.)
        naf.fit(T)
        naf.smoothed_hazard_(1.)
        self.assertTrue(True)

    def test_smoothing_hazard_ties_no_event_observed(self):
        T = np.random.binomial(20, 0.7, size=300)
        naf = NelsonAalenFitter()
        naf.fit(T)
        naf.smoothed_hazard_(1.)
        self.assertTrue(True)

    def test_smoothing_hazard_with_spike_at_time_0(self):
        T = np.random.binomial(20, 0.7, size=300)
        T[np.random.binomial(1, 0.3, size=300).astype(bool)] = 0
        naf = NelsonAalenFitter()
        naf.fit(T)
        df = naf.smoothed_hazard_(bandwidth=0.1)
        self.assertTrue(df.iloc[0].values[0] > df.iloc[1].values[0])

    def test_multivariate_unequal_intensities(self):
        T = np.random.exponential(10, size=300)
        g = np.random.binomial(2, 0.5, size=300)
        T[g == 1] = np.random.exponential(6, size=(g == 1).sum())
        s, _, result = multivariate_logrank_test(T, g)
        self.assertTrue(result)

    def test_multivariate_equal_intensities(self):
        N = 100
        false_positives = 0
        alpha = 0.95
        for i in range(100):
            T = np.random.exponential(10, size=300)
            g = np.random.binomial(2, 0.5, size=300)
            s, _, result = multivariate_logrank_test(T, g, alpha=alpha, suppress_print=True)
            false_positives += result is not None
        bounds = beta.interval(0.95, 1 + false_positives, N - false_positives + 1)
        self.assertTrue(bounds[0] < 1 - alpha < bounds[1])

    def test_pairwise_waltons_dataset(self):
        _, _, R = pairwise_logrank_test(waltons_dataset['T'], waltons_dataset['group'])
        self.assertTrue(R.values[0, 1])

    def test_pairwise_logrank_test(self):
        T = np.random.exponential(10, size=500)
        g = np.random.binomial(2, 0.7, size=500)
        S, P, R = pairwise_logrank_test(T, g, alpha=0.99)
        V = np.array([[np.nan, None, None], [None, np.nan, None], [None, None, np.nan]])
        npt.assert_array_equal(R, V)

    def test_multivariate_inputs(self):
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

    def test_lists_to_KaplanMeierFitter(self):
        T = [2, 3, 4., 1., 6, 5.]
        C = [1, 0, 0, 0, 1, 1]
        kmf = KaplanMeierFitter()
        with_list = kmf.fit(T, C).survival_function_.values
        with_array = kmf.fit(np.array(T), np.array(C)).survival_function_.values
        npt.assert_array_equal(with_list, with_array)

    def test_lists_to_NelsonAalenFitter(self):
        T = [2, 3, 4., 1., 6, 5.]
        C = [1, 0, 0, 0, 1, 1]
        naf = NelsonAalenFitter()
        with_list = naf.fit(T, C).cumulative_hazard_.values
        with_array = naf.fit(np.array(T), np.array(C)).cumulative_hazard_.values
        npt.assert_array_equal(with_list, with_array)

    def test_timeline_to_NelsonAalenFitter(self):
        T = [2, 3, 1., 6, 5.]
        C = [1, 0, 0, 1, 1]
        timeline = [2, 3, 4., 1., 6, 5.]
        naf = NelsonAalenFitter()
        with_list = naf.fit(T, C, timeline=timeline).cumulative_hazard_.values
        with_array = naf.fit(T, C, timeline=np.array(timeline)).cumulative_hazard_.values
        npt.assert_array_equal(with_list, with_array)

    def test_pairwise_allows_dataframes(self):
        N = 100
        df = pd.DataFrame(np.empty((N, 3)), columns=["T", "C", "group"])
        df["T"] = np.random.exponential(1, size=N)
        df["C"] = np.random.binomial(1, 0.6, size=N)
        df["group"] = np.random.binomial(2, 0.5, size=N)
        try:
            pairwise_logrank_test(df['T'], df["group"], event_observed=df["C"])
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_exponential_data_sets_correct_censor(self):
        N = 20000
        censorship = 0.2
        T, C = exponential_survival_data(N, censorship, scale=10)
        self.assertTrue(abs(C.mean() - (1 - censorship)) < 0.02)

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_exponential_data_sets_fit(self):
        N = 20000
        T, C = exponential_survival_data(N, 0.2, scale=10)
        naf = NelsonAalenFitter()
        naf.fit(T, C).plot()
        plt.title("Should be a linear with slope = 0.1")

    def test_subtraction_function(self):
        kmf = KaplanMeierFitter()
        kmf.fit(waltons_dataset['T'])
        npt.assert_array_almost_equal(kmf.subtract(kmf).sum().values, 0.0)

    def test_divide_function(self):
        kmf = KaplanMeierFitter()
        kmf.fit(waltons_dataset['T'])
        npt.assert_array_almost_equal(np.log(kmf.divide(kmf)).sum().values, 0.0)

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_kmf_minimum_observation_bias(self):
        N = 250
        kmf = KaplanMeierFitter()
        T, C = exponential_survival_data(N, 0.2, scale=10)
        B, _ = exponential_survival_data(N, 0.0, scale=2)
        kmf.fit(T, C, entry=B)
        kmf.plot()
        plt.title("Should have larger variances in the tails")

    def test_stat_error(self):
        births = np.array([51, 58, 55, 28, 21, 19, 25, 48, 47, 25, 31, 24, 25, 30, 33, 36, 30,
                           41, 43, 45, 35, 29, 35, 32, 36, 32, 10])
        observations = np.array([1,  1,  2, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31,
                                 22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        kmf = KaplanMeierFitter()
        with self.assertRaises(StatError):
            kmf.fit(observations, entry=births)

    def test_BHF_fit(self):
        bfh = BreslowFlemingHarringtonFitter()
        births = np.array([51, 58, 55, 28, 21, 19, 25, 48, 47, 25, 31, 24, 25, 30, 33, 36, 30,
                           41, 43, 45, 35, 29, 35, 32, 36, 32, 10])
        observations = np.array([1,  1,  2, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31,
                                 22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        bfh.fit(observations, entry=births)
        return

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_bayesian_fitter_low_data(self):
        bf = BayesianFitter(samples=15)
        bf.fit(waltonT1)
        ax = bf.plot(alpha=.2)
        bf.fit(waltonT2)
        bf.plot(ax=ax, alpha=0.2, c='k')
        plt.show()
        return

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_bayesian_fitter_large_data(self):
        bf = BayesianFitter()
        bf.fit(np.random.exponential(10, size=1000))
        bf.plot()
        plt.show()
        return

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_kmf_left_censorship_plots(self):
        kmf = KaplanMeierFitter()
        lcd_dataset = generate_lcd_dataset()
        kmf.fit(lcd_dataset['alluvial_fan']['T'], lcd_dataset['alluvial_fan']['C'], left_censorship=True, label='alluvial_fan')
        ax = kmf.plot()

        kmf.fit(lcd_dataset['basin_trough']['T'], lcd_dataset['basin_trough']['C'], left_censorship=True, label='basin_trough')
        ax = kmf.plot(ax=ax)
        plt.show()
        return

    def test_kmf_left_censorship_stats(self):
        T = [3, 5, 5, 5, 6, 6, 10, 12]
        C = [1, 0, 0, 1, 1, 1, 0, 1]
        kmf = KaplanMeierFitter()
        kmf.fit(T, C, left_censorship=True)

    def test_sort_doesnt_affect_kmf(self):
        T = [3, 5, 5, 5, 6, 6, 10, 12]
        kmf = KaplanMeierFitter()
        assert_frame_equal(kmf.fit(T).survival_function_, kmf.fit(sorted(T)).survival_function_)

    def kaplan_meier(self, censor=False):
        km = np.zeros((len(list(self.lifetimes.keys())), 1))
        ordered_lifetimes = np.sort(list(self.lifetimes.keys()))
        N = len(LIFETIMES)
        v = 1.
        n = N * 1.0
        for i, t in enumerate(ordered_lifetimes):
            if censor:
                ix = LIFETIMES == t
                c = sum(1 - OBSERVED[ix])
                if n != 0:
                    v *= (1 - (self.lifetimes.get(t) - c) / n)
                n -= self.lifetimes.get(t)
            else:
                v *= (1 - self.lifetimes.get(t) / n)
                n -= self.lifetimes.get(t)
            km[i] = v
        if km[0] < 1.:
            km = np.insert(km, 0, 1.)
        return km.reshape(len(km), 1)

    def nelson_aalen(self, censor=False):
        na = np.zeros((len(list(self.lifetimes.keys())), 1))
        ordered_lifetimes = np.sort(list(self.lifetimes.keys()))
        N = len(LIFETIMES)
        v = 0.
        n = N * 1.0
        for i, t in enumerate(ordered_lifetimes):
            if censor:
                ix = LIFETIMES == t
                c = sum(1 - OBSERVED[ix])
                if n != 0:
                    v += ((self.lifetimes.get(t) - c) / n)
                n -= self.lifetimes.get(t)
            else:
                v += (self.lifetimes.get(t) / n)
                n -= self.lifetimes.get(t)
            na[i] = v
        if na[0] > 0:
            na = np.insert(na, 0, 0.)
        return na.reshape(len(na), 1)

    def test_concordance_index(self):
        size = 1000
        T = np.random.normal(size=size)
        P = np.random.normal(size=size)
        C = np.random.choice([0, 1], size=size)
        Z = np.zeros_like(T)

        # Zeros is exactly random
        self.assertTrue(concordance_index(T, Z) == 0.5)
        self.assertTrue(concordance_index(T, Z, C) == 0.5)

        # Itself is 1
        self.assertTrue(concordance_index(T, T) == 1.0)
        self.assertTrue(concordance_index(T, T, C) == 1.0)

        # Random is close to 0.5
        self.assertTrue(abs(concordance_index(T, P) - 0.5) < 0.05)
        self.assertTrue(abs(concordance_index(T, P, C) - 0.5) < 0.05)


class AalenAdditiveModelTests(unittest.TestCase):

    def setUp(self):
        self.aaf = AalenAdditiveFitter(penalizer=0.1, fit_intercept=False)

    def test_large_dimensions_for_recursion_error(self):
        n = 500
        d = 50
        X = pd.DataFrame(np.random.randn(n, d))
        T = np.random.exponential(size=n)
        X['T'] = T
        X['E'] = 1
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        return True

    def test_tall_data_points(self):
        n = 20000
        d = 2
        X = pd.DataFrame(np.random.randn(n, d))
        T = np.random.exponential(size=n)
        X['T'] = T
        X['E'] = 1
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        return True

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_aaf_panel_dataset(self):
        aaf = AalenAdditiveFitter()
        panel_dataset = pd.read_csv('./datasets/panel_test.csv')
        aaf.fit(panel_dataset, id_col='id', duration_col='t', event_col='E')
        aaf.plot()
        return

    def test_aalen_additive_median_predictions_split_data(self):
        # This tests to make sure that my median predictions statisfy
        # the prediction are greater than the actual 1/2 the time.
        # generate some hazard rates and a survival data set
        n = 2500
        d = 5
        timeline = np.linspace(0, 70, 5000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        T = generate_random_lifetimes(hz, timeline)
        X['T'] = T
        X['E'] = 1
        # fit it to Aalen's model
        aaf = AalenAdditiveFitter(penalizer=0.1, fit_intercept=False)
        aaf.fit(X)

        # predictions
        T_pred = aaf.predict_median(X[list(range(6))])
        self.assertTrue(abs((T_pred.values > T).mean() - 0.5) < 0.05)

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_aalen_additive_fit_no_censor(self):
        # this is a visual test of the fitting the cumulative
        # hazards.
        n = 2500
        d = 6
        timeline = np.linspace(0, 70, 10000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        X.columns = coef.columns
        cumulative_hazards = pd.DataFrame(cumulative_integral(coef.values, timeline),
                                          index=timeline, columns=coef.columns)
        T = generate_random_lifetimes(hz, timeline)
        X['T'] = T
        X['E'] = np.random.binomial(1, 1, n)
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        for i in range(d + 1):
            ax = plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].ix[:15].plot(legend=False, ax=ax)
            ax = aaf.plot(ix=slice(0, 15), ax=ax, columns=[col], legend=False)
        plt.show()
        return

    @unittest.skipUnless("DISPLAY" in os.environ, "requires display")
    def test_aalen_additive_fit_with_censor(self):
        # this is a visual test of the fitting the cumulative
        # hazards.
        n = 2500
        d = 6
        timeline = np.linspace(0, 70, 10000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        X.columns = coef.columns
        cumulative_hazards = pd.DataFrame(cumulative_integral(coef.values, timeline),
                                          index=timeline, columns=coef.columns)
        T = generate_random_lifetimes(hz, timeline)
        X['T'] = T
        X['E'] = np.random.binomial(1, 0.99, n)
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        for i in range(d + 1):
            ax = plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].ix[:15].plot(legend=False, ax=ax)
            ax = aaf.plot(ix=slice(0, 15), ax=ax, columns=[col], legend=False)
        plt.show()
        return

    def test_dataframe_input_with_nonstandard_index(self):

        df = pd.DataFrame([(16, True, True), (1, True, True), (4, False, True)],
                          columns=['duration', 'done_feeding', 'white'],
                          index=['a', 'b', 'c'])
        aaf = AalenAdditiveFitter()
        aaf.fit(df, duration_col='duration', event_col='done_feeding')
        return

    def test_predict_percentile_returns_a_series(self):
        X = generate_regression_dataset()
        x = X[X.columns - ['T', 'E']]
        aaf = AalenAdditiveFitter()
        aaf.fit(X, duration_col='T', event_col='E')
        result = aaf.predict_percentile(x)
        self.assertTrue(isinstance(result, pd.Series))
        self.assertTrue(result.shape == (x.shape[0],))

class RegressionTests(unittest.TestCase):

    def setUp(self):
        self.aaf = AalenAdditiveFitter()
        self.cph = CoxPHFitter()

    def test_predict_methods_in_regression_return_same_types(self):
        X = generate_regression_dataset()
        x = X[X.columns - ['T', 'E']]

        self.aaf.fit(X, duration_col='T', event_col='E')
        self.cph.fit(X, duration_col='T', event_col='E')

        for fit_method in ['predict_percentile', 'predict_median', 'predict_expectation', 'predict_survival_function', 'predict', 'predict_cumulative_hazard']:
            self.assertEqual(type(getattr(self.aaf,fit_method)(x)), type(getattr(self.cph,fit_method)(x)))

    def test_duration_vector_can_be_normalized(self):
        df = pd.read_csv('./datasets/kidney_transplant.csv')
        t = df['time']
        normalized_df = df.copy()
        normalized_df['time'] = (normalized_df['time'] - t.mean())/t.std()
        
        for fitter in [self.cph, self.aaf]:
            # we drop indexs since aaf will have a different "time" index.
            hazards = fitter.fit(df, duration_col='time', event_col='death').hazards_.reset_index(drop=True)
            hazards_norm = fitter.fit(normalized_df, duration_col='time', event_col='death').hazards_.reset_index(drop=True)
            assert_frame_equal(hazards, hazards_norm)



@unittest.skipUnless("DISPLAY" in os.environ, "requires display")
class PlottingTests(unittest.TestCase):

    def test_aalen_additive_plot(self):
        # this is a visual test of the fitting the cumulative
        # hazards.
        n = 2500
        d = 3
        timeline = np.linspace(0, 70, 10000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        T = generate_random_lifetimes(hz, timeline)
        C = np.random.binomial(1, 1., size=n)
        X['T'] = T
        X['E'] = C

        # fit the aaf, no intercept as it is already built into X, X[2] is ones
        aaf = AalenAdditiveFitter(penalizer=0.1, fit_intercept=False)

        aaf.fit(X)
        ax = aaf.plot(iloc=slice(0, aaf.cumulative_hazards_.shape[0] - 100))
        ax.set_xlabel("time")
        ax.set_title('.plot() cumulative hazards')
        return

    def test_aalen_additive_smoothed_plot(self):
        # this is a visual test of the fitting the cumulative
        # hazards.
        n = 2500
        d = 3
        timeline = np.linspace(0, 150, 5000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        T = generate_random_lifetimes(hz, timeline) + 0.1 * np.random.uniform(size=(n, 1))
        C = np.random.binomial(1, 0.8, size=n)
        X['T'] = T
        X['E'] = C

        # fit the aaf, no intercept as it is already built into X, X[2] is ones
        aaf = AalenAdditiveFitter(penalizer=0.1, fit_intercept=False)
        aaf.fit(X)
        ax = aaf.smoothed_hazards_(1).iloc[0:aaf.cumulative_hazards_.shape[0] - 500].plot()
        ax.set_xlabel("time")
        ax.set_title('.plot() smoothed hazards')
        return

    def test_kmf_plotting(self):
        data1 = np.random.exponential(10, size=(100))
        data2 = np.random.exponential(2, size=(200, 1))
        data3 = np.random.exponential(4, size=(500, 1))
        kmf = KaplanMeierFitter()
        kmf.fit(data1, label='test label 1')
        ax = kmf.plot()
        kmf.fit(data2, label='test label 2')
        kmf.plot(ax=ax)
        kmf.fit(data3, label='test label 3')
        kmf.plot(ax=ax)
        plt.title("testing kmf")
        return

    def test_naf_plotting(self):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(500))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(color="r")
        naf.fit(data2)
        naf.plot(ax=ax, c="k")
        plt.title('testing naf + custom colors')
        return

    def test_estimators_accept_lists(self):
        data = [1, 2, 3, 4, 5]
        NelsonAalenFitter().fit(data)
        KaplanMeierFitter().fit(data)
        return True

    def test_ix_slicing(self):
        naf = NelsonAalenFitter().fit(waltons_dataset['T'])
        self.assertTrue(naf.cumulative_hazard_.ix[0:10].shape[0] == 4)
        return

    def test_iloc_slicing(self):
        naf = NelsonAalenFitter().fit(waltons_dataset['T'])
        self.assertTrue(naf.cumulative_hazard_.iloc[0:10].shape[0] == 10)
        self.assertTrue(naf.cumulative_hazard_.iloc[0:-1].shape[0] == 32)
        return

    def test_naf_plotting_slice(self):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(200, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(ix=slice(0, None))
        naf.fit(data2)
        naf.plot(ax=ax, ci_force_lines=True, iloc=slice(100, 180))
        plt.title('testing slicing')
        return

    def test_plot_lifetimes_calendar(self):
        plt.figure()
        t = np.linspace(0, 20, 1000)
        hz, coef, covrt = generate_hazard_rates(1, 5, t)
        N = 20
        current = 10
        birthtimes = current * np.random.uniform(size=(N,))
        T, C = generate_random_lifetimes(hz, t, size=N, censor=current - birthtimes)
        plot_lifetimes(T, event_observed=C, birthtimes=birthtimes)

    def test_plot_lifetimes_relative(self):
        plt.figure()
        t = np.linspace(0, 20, 1000)
        hz, coef, covrt = generate_hazard_rates(1, 5, t)
        N = 20
        T, C = generate_random_lifetimes(hz, t, size=N, censor=True)
        plot_lifetimes(T, event_observed=C)

    def test_naf_plot_cumulative_hazard(self):
        data1 = np.random.exponential(5, size=(200, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot()
        naf.plot_cumulative_hazard(ax=ax, ci_force_lines=True)
        plt.title("I should have plotted the same thing, but different styles + color!")
        return

    def test_naf_plot_cumulative_hazard_bandwidth_2(self):
        data1 = np.random.exponential(5, size=(2000, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=1., ix=slice(0, 7.))
        plt.title('testing smoothing hazard')
        return

    def test_naf_plot_cumulative_hazard_bandwith_1(self):
        data1 = np.random.exponential(5, size=(2000, 1)) ** 2
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=5., iloc=slice(0, 1700))
        plt.title('testing smoothing hazard')
        return

    def test_show_censor_with_discrete_date(self):
        T = np.random.binomial(20, 0.1, size=100)
        C = np.random.binomial(1, 0.8, size=100)
        kmf = KaplanMeierFitter()
        kmf.fit(T, C).plot(show_censors=True)
        return

    def test_show_censor_with_index_0(self):
        T = np.random.binomial(20, 0.9, size=100)  # lifelines should auto put a 0 in.
        C = np.random.binomial(1, 0.8, size=100)
        kmf = KaplanMeierFitter()
        kmf.fit(T, C).plot(show_censors=True)
        return

    def test_flat_style_and_marker(self):
        data1 = np.random.exponential(10, size=200)
        data2 = np.random.exponential(2, size=200)
        C1 = np.random.binomial(1, 0.9, size=200)
        C2 = np.random.binomial(1, 0.95, size=200)
        kmf = KaplanMeierFitter()
        kmf.fit(data1, C1, label='test label 1')
        ax = kmf.plot(flat=True, censor_styles={'marker': '+', 'mew': 2, 'ms': 7})
        kmf.fit(data2, C2, label='test label 2')
        kmf.plot(ax=ax, censor_styles={'marker': 'o', 'ms': 7}, flat=True)
        plt.title("testing kmf flat styling + marker")
        return

    def test_flat_style_no_censor(self):
        data1 = np.random.exponential(10, size=200)
        kmf = KaplanMeierFitter()
        kmf.fit(data1, label='test label 1')
        ax = kmf.plot(flat=True, censor_styles={'marker': '+', 'mew': 2, 'ms': 7})
        return


class CoxRegressionTests(unittest.TestCase):

    def test_log_likelihood_is_available_in_output(self):
        cox = CoxPHFitter()
        cox.fit(data_nus, duration_col='t', event_col='E', include_likelihood=True)
        assert abs(cox._log_likelihood - -12.7601409152) < 0.001

    def test_efron_computed_by_hand_examples(self):
        cox = CoxPHFitter()

        X = data_nus['x'][:, None]
        T = data_nus['t']
        E = data_nus['E']

        # Enforce numpy arrays
        X = np.array(X)
        T = np.array(T)
        E = np.array(E)

        # Want as bools
        E = E.astype(bool)

        # tests from http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
        beta = np.array([[0]])

        l, u = cox._get_efron_values(X, beta, T, E)
        l = -l

        assert np.abs(l[0][0] - 77.13) < 0.05
        assert np.abs(u[0] - -2.51) < 0.05
        beta = beta + u / l
        assert np.abs(beta - -0.0326) < 0.05

        l, u = cox._get_efron_values(X, beta, T, E)
        l = -l

        assert np.abs(l[0][0] - 72.83) < 0.05
        assert np.abs(u[0] - -0.069) < 0.05
        beta = beta + u / l
        assert np.abs(beta - -0.0325) < 0.01

        l, u = cox._get_efron_values(X, beta, T, E)
        l = -l

        assert np.abs(l[0][0] - 72.70) < 0.01
        assert np.abs(u[0] - -0.000061) < 0.01
        beta = beta + u / l
        assert np.abs(beta - -0.0335) < 0.01

    def test_efron_newtons_method(self):
        newton = CoxPHFitter()._newton_rhaphson
        X, T, E = data_nus['x'][:, None], data_nus['t'], data_nus['E']
        assert np.abs(newton(X, T, E)[0][0] - -0.0335) < 0.0001

    def test_fit_method(self):
        cf = CoxPHFitter()
        cf.fit(data_nus, duration_col='t', event_col='E')
        self.assertTrue(np.abs(cf.hazards_.ix[0][0] - -0.0335) < 0.0001)

    def test_crossval_for_cox_ph(self):
        cf = CoxPHFitter()

        for data_pred in [data_pred1, data_pred2]:
            scores = k_fold_cross_validation(cf, data_pred,
                                             duration_col='t',
                                             event_col='E', k=3)
            expected = 0.9
            msg = "Expected min-mean c-index {:.2f} < {:.2f}"
            self.assertTrue(scores.mean() > expected,
                            msg.format(expected, scores.mean()))

    def test_crossval_with_normalized_data(self):
        cf = CoxPHFitter()
        for data_pred in [data_pred1, data_pred2]:
            data_norm = data_pred.copy()

            times = data_norm['t']
            # Normalize to mean = 0 and standard deviation = 1
            times -= np.mean(times)
            times /= np.std(times)
            data_norm['t'] = times

            x1 = data_norm['x1']
            x1 -= np.mean(x1)
            x1 /= np.std(x1)
            data_norm['x1'] = x1

            if 'x2' in data_norm.columns:
                x2 = data_norm['x2']
                x2 -= np.mean(x2)
                x2 /= np.std(x2)
                data_norm['x2'] = x2

            scores = k_fold_cross_validation(cf, data_norm,
                                             duration_col='t',
                                             event_col='E', k=3)
            expected = 0.9
            msg = "Expected min-mean c-index {:.2f} < {:.2f}"
            self.assertTrue(scores.mean() > expected,
                            msg.format(expected, scores.mean()))

    def test_output_against_R(self):
        # from http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-cox-regression.pdf
        expected = np.array([[-0.3794, -0.0574, 0.3139, -0.1498, -0.4337, -0.0849,  0.0915]])
        df = generate_rossi_dataset()
        cf = CoxPHFitter()
        cf.fit(df, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=3)

    def test_coef_output_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see example 8.3 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = pd.read_csv('./datasets/kidney_transplant.csv', usecols=['time','death','black_male','white_male','black_female'])
        cf = CoxPHFitter()
        cf.fit(df, duration_col='time', event_col='death')

        # coefs
        actual_coefs = cf.hazards_.values 
        expected_coefs = np.array([[0.1596, 0.2484, 0.6567]])
        npt.assert_array_almost_equal(actual_coefs, expected_coefs, decimal=4)

    def test_se_and_p_value_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see table 8.1 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = pd.read_csv('./datasets/larynx.csv')
        cf = CoxPHFitter()
        cf.fit(df, duration_col='time', event_col='death')

        #standard errors
        actual_se = cf._compute_standard_errors().values
        expected_se = np.array([[0.0143,  0.4623,  0.3561,  0.4222]])
        npt.assert_array_almost_equal(actual_se, expected_se, decimal=4)

        #p-values
        actual_p = cf._compute_p_values()
        expected_p = np.array([0.1847, 0.7644,  0.0730, 0.00])
        npt.assert_array_almost_equal(actual_p, expected_p, decimal=3)


# some data
LIFETIMES = np.array([2, 4, 4, 4, 5, 7, 10, 11, 11, 12])
OBSERVED = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0])

# walton's data
waltons_dataset = generate_waltons_dataset()
ix = waltons_dataset['group'] == 'miR-137'
waltonT1 = waltons_dataset.ix[ix]['T']
waltonT2 = waltons_dataset.ix[~ix]['T']

data_nus = pd.DataFrame([
    [6, 31.4],
    [98, 21.5],
    [189, 27.1],
    [374, 22.7],
    [1002, 35.7],
    [1205, 30.7],
    [2065, 26.5],
    [2201, 28.3],
    [2421, 27.9]],
    columns=['t', 'x'])
data_nus['E'] = True

# Simple sets for predictions
N = 50
data_pred1 = pd.DataFrame()
data_pred1['x1'] = np.random.uniform(size=N)
data_pred1['t'] = 1 + data_pred1['x1'] + np.random.normal(0, 0.07, size=N)
data_pred1['E'] = True

data_pred2 = pd.DataFrame()
data_pred2['x1'] = np.random.uniform(size=N)
data_pred2['x2'] = np.random.uniform(size=N)
data_pred2['t'] = (1 + data_pred2['x1'] + data_pred2['x2'] +
                   np.random.normal(0, 0.07, size=N))
data_pred2['E'] = True


if __name__ == '__main__':
    unittest.main()
