"""
python -m lifelines.tests.test_suit

"""
from __future__ import print_function

import unittest
from StringIO import StringIO

import numpy as np
import numpy.testing as npt
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

from ..estimation import KaplanMeierFitter, NelsonAalenFitter, AalenAdditiveFitter, \
                         median_survival_times, BreslowFlemingHarringtonFitter, BayesianFitter
from ..statistics import logrank_test, multivariate_logrank_test, pairwise_logrank_test
from ..generate_datasets import *
from ..plotting import plot_lifetimes
from ..utils import *
from ..datasets import lcd_dataset

class MiscTests(unittest.TestCase):

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

    """
    def test_datetimes_to_durations_max_dates(self):
        start_date = ['2013-10-10', '2013-10-09', '2013-10-08']
        end_date =  ['2014-10-10', '2014-10-09', '2014-10-08']
        T,C = datetimes_to_durations(start_date, end_date, fill_date = "2014-10-09" )
        npt.assert_almost_equal(C, np.array([1,0,0], dtype=bool) )
        return
    """

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
        df = pd.DataFrame([[1,True,3],[1,True,3],[4,False,2]], columns = ['duration', 'E', 'G'])
        ug,_,_,_ = group_survival_table_from_events(df.G, df.duration, df.E, np.array([[0,0,0]]))
        npt.assert_array_equal(ug, np.array([3,2]))


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
        self.assertTrue(median_survival_times(sv).ix[0] == 500)

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
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(5, size=(200, 1))
        summary, p_value, result = logrank_test(data1, data2)
        print(summary)
        self.assertTrue(result is None)

    def test_unequal_intensity(self):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(200, 1))
        summary, p_value, result = logrank_test(data1, data2)
        print(summary)
        self.assertTrue(result)

    def test_unequal_intensity_event_observed(self):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(200, 1))
        eventA = np.random.binomial(1, 0.5, size=(200, 1))
        eventB = np.random.binomial(1, 0.5, size=(200, 1))
        summary, p_value, result = logrank_test(data1, data2, event_observed_A=eventA, event_observed_B=eventB)
        print(summary)
        self.assertTrue(result)

    def test_integer_times_logrank_test(self):
        data1 = np.random.exponential(5, size=(200, 1)).astype(int)
        data2 = np.random.exponential(1, size=(200, 1)).astype(int)
        summary, p_value, result = logrank_test(data1, data2)
        print(summary)
        self.assertTrue(result)

    def test_waltons_data(self):
        summary, p_value, result = logrank_test(waltonT1, waltonT2)
        print(summary)
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
        print(s)
        self.assertTrue(result == True)

    def test_multivariate_equal_intensities(self):
        T = np.random.exponential(10, size=300)
        g = np.random.binomial(2, 0.5, size=300)
        s, _, result = multivariate_logrank_test(T, g)
        print(s)
        self.assertTrue(result is None)

    def test_pairwise_waltons_data(self):
        _, _, R = pairwise_logrank_test(waltonT, waltonG)
        print(R)
        self.assertTrue(R.values[0, 1])

    def test_pairwise_logrank_test(self):
        T = np.random.exponential(10, size=300)
        g = np.random.binomial(2, 0.7, size=300)
        S, P, R = pairwise_logrank_test(T, g, alpha=0.95)
        V = np.array([[np.nan, None, None], [None, np.nan, None], [None, None, np.nan]])
        npt.assert_array_equal(R, V)
    
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
        T, C = exponential_survival_data(N, 0.2, scale=10)
        self.assertTrue(abs(C.mean() - 0.8) < 0.02)

    def test_exponential_data_sets_fit(self):
        N = 20000
        T, C = exponential_survival_data(N, 0.2, scale=10)
        naf = NelsonAalenFitter()
        naf.fit(T, C).plot()
        plt.title("Should be a linear with slope = 0.1")

    def test_subtraction_function(self):
        kmf = KaplanMeierFitter()
        kmf.fit(waltonT)
        npt.assert_array_almost_equal(kmf.subtract(kmf).sum().values, 0.0)

    def test_divide_function(self):
        kmf = KaplanMeierFitter()
        kmf.fit(waltonT)
        npt.assert_array_almost_equal(np.log(kmf.divide(kmf)).sum().values, 0.0)

    def test_kmf_minimum_observation_bias(self):
        N = 250
        kmf = KaplanMeierFitter()
        T, C = exponential_survival_data(N, 0.2, scale=10)
        B, _ = exponential_survival_data(N, 0.0, scale=2)
        kmf.fit(T,C, entry=B)
        kmf.plot()
        plt.title("Should have larger variances in the tails")

    def test_stat_error(self):
        births = np.array([51, 58, 55, 28, 21, 19, 25, 48, 47, 25, 31, 24, 25, 30, 33, 36, 30, \
            41, 43, 45, 35, 29, 35, 32, 36, 32, 10])
        observations = np.array([ 1,  1,  2, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31, \
            22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        kmf = KaplanMeierFitter()
        with self.assertRaises(StatError):
            kmf.fit(observations, entry=births)

    def test_BHF_fit(self):
        bfh = BreslowFlemingHarringtonFitter()
        births = np.array([51, 58, 55, 28, 21, 19, 25, 48, 47, 25, 31, 24, 25, 30, 33, 36, 30, \
           41, 43, 45, 35, 29, 35, 32, 36, 32, 10])
        observations = np.array([ 1,  1,  2, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31, \
           22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        bfh.fit(observations, entry=births)
        return 

    def test_bayesian_fitter_low_data(self):
        bf = BayesianFitter(samples=15)
        bf.fit(waltonT1)
        ax = bf.plot(alpha=.2)
        bf.fit(waltonT2)
        bf.plot(ax=ax, alpha=0.2, c='k')
        plt.show()
        return

    def test_bayesian_fitter_large_data(self):
        bf = BayesianFitter()
        bf.fit(np.random.exponential(10,size=1000))
        bf.plot()
        plt.show()
        return

    def test_kmf_left_censorship_plots(self):
        kmf = KaplanMeierFitter()
        kmf.fit(lcd_dataset['alluvial_fan']['T'], lcd_dataset['alluvial_fan']['C'], left_censorship=True, label='alluvial_fan')
        ax = kmf.plot()

        kmf.fit(lcd_dataset['basin_trough']['T'], lcd_dataset['basin_trough']['C'], left_censorship=True, label='basin_trough')
        ax = kmf.plot(ax=ax)
        plt.show()
        return

    def test_kmf_left_censorship_stats(self):
        T = [3, 5, 5, 5, 6, 6, 10, 12]
        C = [1,0,0,1,1,1,0,1]
        kmf = KaplanMeierFitter()
        kmf.fit(T,C, left_censorship=True)



    def kaplan_meier(self, censor=False):
        km = np.zeros((len(self.lifetimes.keys()), 1))
        ordered_lifetimes = np.sort(self.lifetimes.keys())
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
        na = np.zeros((len(self.lifetimes.keys()), 1))
        ordered_lifetimes = np.sort(self.lifetimes.keys())
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


class AalenAdditiveModelTests(unittest.TestCase):

    def setUp(self):
        self.aaf = AalenAdditiveFitter(penalizer=0.1, fit_intercept=False)

    def test_large_dimensions_for_recursion_error(self):
        n = 500
        d = 50
        X = pd.DataFrame(np.random.randn(n,d))
        T = np.random.exponential(size=n)
        X['T'] = T
        X['E'] = 1
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        return True

    def test_tall_data_points(self):
        n = 20000
        d = 2
        X = pd.DataFrame(np.random.randn(n,d))
        T = np.random.exponential(size=n)
        X['T'] = T
        X['E'] = 1
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        return True
    
    def test_aaf_panel_dataset(self):
        aaf = AalenAdditiveFitter()
        aaf.fit(panel_dataset, id_col='id',duration_col='t', event_col='E')
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
        T_pred = aaf.predict_median(X[range(6)])
        self.assertTrue(abs((T_pred.values > T).mean() - 0.5) < 0.05)


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
        X['E'] = np.random.binomial(1,1,n)
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        for i in range(d+1):
            ax = plt.subplot(d+1,1,i+1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].ix[:15].plot(legend=False,ax=ax)
            ax = aaf.plot(ix=slice(0,15),ax=ax, columns=[col], legend=False)
        plt.show()
        return

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
        X['E'] = np.random.binomial(1,0.99,n)
        aaf = AalenAdditiveFitter(penalizer=1., fit_intercept=False)
        aaf.fit(X)

        for i in range(d+1):
            ax = plt.subplot(d+1,1,i+1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].ix[:15].plot(legend=False,ax=ax)
            ax = aaf.plot(ix=slice(0,15),ax=ax, columns=[col], legend=False)
        plt.show()
        return

    def test_dataframe_input_with_nonstandard_index(self):

        df = pd.DataFrame([(16,True,True), (1,True,True), (4,False,True)], 
                          columns=['duration', 'done_feeding', 'white'],
                          index=['a','b','c'])
        aaf = AalenAdditiveFitter()
        aaf.fit(df, duration_col='duration', event_col='done_feeding')
        return 


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
        naf = NelsonAalenFitter().fit(waltonT)
        self.assertTrue(naf.cumulative_hazard_.ix[0:10].shape[0] == 4)
        return

    def test_iloc_slicing(self):
        naf = NelsonAalenFitter().fit(waltonT)
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
        ax = kmf.plot(flat=True, censor_styles={'marker':'+', 'mew':2, 'ms':7})
        kmf.fit(data2, C2, label='test label 2')
        kmf.plot(ax=ax, censor_styles={'marker':'o', 'ms':7}, flat=True)
        plt.title("testing kmf flat styling + marker")
        return

    def test_flat_style_no_censor(self):
        data1 = np.random.exponential(10, size=200)
        kmf = KaplanMeierFitter()
        kmf.fit(data1,label='test label 1')
        ax = kmf.plot(flat=True, censor_styles={'marker':'+', 'mew':2, 'ms':7})
        return


# some data
LIFETIMES = np.array([2, 4, 4, 4, 5, 7, 10, 11, 11, 12])
OBSERVED = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0])
N = len(LIFETIMES)

waltonT1 = np.array([6.,13.,13.,13.,19.,19.,19.,26.,26.,26.,26.,26.,33.,33.,47.,62.,62.,9.,9.,9.,15.,15.,22.,22.,22.,22.,29.,29.,29.,29.,29.,36.,36.,43.])
waltonT2 = np.array([33.,54.,54.,61.,61.,61.,61.,61.,61.,61.,61.,61.,61.,61.,69.,69.,69.,69.,69.,69.,69.,69.,69.,69.,69.,32.,53.,53.,60.,60.,60.,60.,60.,
                        68.,68.,68.,68.,68.,68.,68.,68.,68.,68.,75.,17.,51.,51.,51.,58.,58.,58.,58.,66.,66.,7.,7.,41.,41.,41.,41.,41.,41.,41.,48.,48.,48.,
                        48.,48.,48.,48.,48.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,56.,63.,63.,63.,63.,63.,63.,63.,63.,63.,69.,
                        69.,38.,38.,45.,45.,45.,45.,45.,45.,45.,45.,45.,45.,53.,53.,53.,53.,53.,60.,60.,60.,60.,60.,60.,60.,60.,60.,60.,60.,66.])

waltonG = np.array(['miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                    'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                    'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                    'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                    'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137', 'miR-137',
                    'miR-137', 'miR-137', 'miR-137', 'miR-137', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control', 'control', 'control', 'control', 'control', 'control',
                    'control'], dtype=object)

waltonT = np.array([6., 13., 13., 13., 19., 19., 19., 26., 26., 26., 26.,
                    26., 33., 33., 47., 62., 62., 9., 9., 9., 15., 15.,
                    22., 22., 22., 22., 29., 29., 29., 29., 29., 36., 36.,
                    43., 33., 54., 54., 61., 61., 61., 61., 61., 61., 61.,
                    61., 61., 61., 61., 69., 69., 69., 69., 69., 69., 69.,
                    69., 69., 69., 69., 32., 53., 53., 60., 60., 60., 60.,
                    60., 68., 68., 68., 68., 68., 68., 68., 68., 68., 68.,
                    75., 17., 51., 51., 51., 58., 58., 58., 58., 66., 66.,
                    7., 7., 41., 41., 41., 41., 41., 41., 41., 48., 48.,
                    48., 48., 48., 48., 48., 48., 56., 56., 56., 56., 56.,
                    56., 56., 56., 56., 56., 56., 56., 56., 56., 56., 56.,
                    56., 56., 63., 63., 63., 63., 63., 63., 63., 63., 63.,
                    69., 69., 38., 38., 45., 45., 45., 45., 45., 45., 45.,
                    45., 45., 45., 53., 53., 53., 53., 53., 60., 60., 60.,
                    60., 60., 60., 60., 60., 60., 60., 60., 66.])



panel_dataset = pd.read_csv(
    StringIO("""id,t,E,var1,var2
1,1,0,0,1
1,2,0,0,1
1,3,0,4,3
1,4,1,8,4
2,1,0,1.2,1
2,2,0,1.2,2
2,3,0,1.2,2
3,1,0,0,1
3,2,1,1,2
4,1,0,0,1
4,2,0,1,2
4,3,0,1,3
4,4,0,2,4
4,5,1,2,5
5,1,0,1,-1
5,2,0,2,-1
5,3,0,3,-1
6,1,1,3,0
7,1,0,1,0
7,2,0,2,1
7,3,0,3,0
7,4,0,3,1
7,5,0,3,0
7,6,1,3,1
8,1,0,-1,0
8,2,1,1,0
9,1,0,1,1
9,2,0,2,2
"""))



if __name__ == '__main__':
    unittest.main()
