from __future__ import print_function
from __future__ import division

from collections import Counter, Iterable
import os
import warnings
from itertools import combinations

try:
    from StringIO import StringIO as stringio, StringIO
except ImportError:
    from io import StringIO, BytesIO as stringio

import numpy as np
import pandas as pd
import pytest

from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy.testing as npt
from numpy.linalg.linalg import LinAlgError

from lifelines.utils import k_fold_cross_validation, StatError
from lifelines.estimation import CoxPHFitter, AalenAdditiveFitter, KaplanMeierFitter, \
    NelsonAalenFitter, BreslowFlemingHarringtonFitter, ExponentialFitter, \
    WeibullFitter, BaseFitter, CoxTimeVaryingFitter, BayesianFitter
from lifelines.datasets import load_larynx, load_waltons, load_kidney_transplant, load_rossi,\
    load_panel_test, load_g3, load_holly_molly_polly, load_regression_dataset,\
    load_stanford_heart_transplants
from lifelines.generate_datasets import generate_hazard_rates, generate_random_lifetimes
from lifelines.utils import concordance_index


@pytest.fixture
def sample_lifetimes():
    N = 30
    return (np.random.randint(20, size=N), np.random.randint(2, size=N))


@pytest.fixture
def positive_sample_lifetimes():
    N = 30
    return (np.random.randint(1, 20, size=N), np.random.randint(2, size=N))


@pytest.fixture
def waltons_dataset():
    return load_waltons()


@pytest.fixture
def data_pred1():
    N = 150
    data_pred1 = pd.DataFrame()
    data_pred1['x1'] = np.random.uniform(size=N)
    data_pred1['t'] = 1 + data_pred1['x1'] + np.random.normal(0, 0.05, size=N)
    data_pred1['E'] = True
    return data_pred1


@pytest.fixture
def univariate_fitters():
    return [KaplanMeierFitter, NelsonAalenFitter, BreslowFlemingHarringtonFitter,
            ExponentialFitter, WeibullFitter]


@pytest.fixture
def data_pred2():
    N = 150
    data_pred2 = pd.DataFrame()
    data_pred2['x1'] = np.random.uniform(size=N)
    data_pred2['x2'] = np.random.uniform(size=N)
    data_pred2['t'] = (1 + data_pred2['x1'] + data_pred2['x2'] +
                       np.random.normal(0, 0.05, size=N))
    data_pred2['E'] = True
    return data_pred2


@pytest.fixture
def data_nus():
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
    return data_nus


@pytest.fixture
def rossi():
    return load_rossi()


@pytest.fixture
def regression_dataset():
    return load_regression_dataset()


class TestBaseFitter():

    def test_repr_without_fitter(self):
        bf = BaseFitter()
        assert bf.__repr__() == '<lifelines.BaseFitter>'

    def test_repr_with_fitter(self, sample_lifetimes):
        T, C = sample_lifetimes
        bf = BaseFitter()
        bf.event_observed = C
        assert bf.__repr__() == '<lifelines.BaseFitter: fitted with %d observations, %d censored>' % (C.shape[0], C.shape[0] - C.sum())


class TestUnivariateFitters():

    def test_univarite_fitters_with_survival_function_have_conditional_time_to(self, positive_sample_lifetimes, univariate_fitters):
        for fitter in univariate_fitters:
            f = fitter().fit(positive_sample_lifetimes[0])
            if hasattr(f, 'survival_function_'):
                assert all(f.conditional_time_to_event_.index == f.survival_function_.index)

    def test_univariate_fitters_allows_one_to_change_alpha_at_fit_time(self, positive_sample_lifetimes, univariate_fitters):
        alpha = 0.9
        alpha_fit = 0.95
        for f in univariate_fitters:
            fitter = f(alpha=alpha)
            fitter.fit(positive_sample_lifetimes[0], alpha=alpha_fit)
            assert str(alpha_fit) in fitter.confidence_interval_.columns[0]

            fitter.fit(positive_sample_lifetimes[0])
            assert str(alpha) in fitter.confidence_interval_.columns[0]

    def test_univariate_fitters_have_a_plot_method(self, positive_sample_lifetimes, univariate_fitters):
        T = positive_sample_lifetimes[0]
        for f in univariate_fitters:
            fitter = f()
            fitter.fit(T)
            assert hasattr(fitter, 'plot')

    def test_predict_methods_returns_a_scalar_or_a_array_depending_on_input(self, positive_sample_lifetimes, univariate_fitters):
        T = positive_sample_lifetimes[0]
        for f in univariate_fitters:
            fitter = f()
            fitter.fit(T)
            assert not isinstance(fitter.predict(1), Iterable)
            assert isinstance(fitter.predict([1, 2]), Iterable)

    def test_predict_method_returns_exact_value_if_given_an_observed_time(self):
        T = [1, 2, 3]
        kmf = KaplanMeierFitter()
        kmf.fit(T)
        time = 1
        assert abs(kmf.predict(time) - kmf.survival_function_.iloc[time].values) < 10e-8

    def test_predict_method_returns_an_approximation_if_not_in_the_index(self):
        T = [1, 2, 3]
        kmf = KaplanMeierFitter()
        kmf.fit(T)
        assert abs(kmf.predict(0.5) - 5 / 6.) < 10e-8
        assert abs(kmf.predict(1.9999) - 0.3333666666) < 10e-8

    def test_custom_timeline_can_be_list_or_array(self, positive_sample_lifetimes, univariate_fitters):
        T, C = positive_sample_lifetimes
        timeline = [2, 3, 4., 1., 6, 5.]
        for f in univariate_fitters:
            fitter = f()
            fitter.fit(T, C, timeline=timeline)
            if hasattr(fitter, 'survival_function_'):
                with_list = fitter.survival_function_.values
                with_array = fitter.fit(T, C, timeline=np.array(timeline)).survival_function_.values
                npt.assert_array_equal(with_list, with_array)
            elif hasattr(fitter, 'cumulative_hazard_'):
                with_list = fitter.cumulative_hazard_.values
                with_array = fitter.fit(T, C, timeline=np.array(timeline)).cumulative_hazard_.values
                npt.assert_array_equal(with_list, with_array)

    def test_custom_timeline(self, positive_sample_lifetimes, univariate_fitters):
        T, C = positive_sample_lifetimes
        timeline = [2, 3, 4., 1., 6, 5.]
        for f in univariate_fitters:
            fitter = f()
            fitter.fit(T, C, timeline=timeline)
            if hasattr(fitter, 'survival_function_'):
                assert sorted(timeline) == list(fitter.survival_function_.index.values)
            elif hasattr(fitter, 'cumulative_hazard_'):
                assert sorted(timeline) == list(fitter.cumulative_hazard_.index.values)

    def test_label_is_a_property(self, positive_sample_lifetimes, univariate_fitters):
        label = 'Test Label'
        for f in univariate_fitters:
            fitter = f()
            fitter.fit(positive_sample_lifetimes[0], label=label)
            assert fitter._label == label
            assert fitter.confidence_interval_.columns[0] == '%s_upper_0.95' % label
            assert fitter.confidence_interval_.columns[1] == '%s_lower_0.95' % label

    def test_ci_labels(self, positive_sample_lifetimes, univariate_fitters):
        expected = ['upper', 'lower']
        for f in univariate_fitters:
            fitter = f()
            fitter.fit(positive_sample_lifetimes[0], ci_labels=expected)
            npt.assert_array_equal(fitter.confidence_interval_.columns, expected)

    def test_lists_as_input(self, positive_sample_lifetimes, univariate_fitters):
        T, C = positive_sample_lifetimes
        for f in univariate_fitters:
            fitter = f()
            if hasattr(fitter, 'survival_function_'):
                with_array = fitter.fit(T, C).survival_function_
                with_list = fitter.fit(list(T), list(C)).survival_function_
                assert_frame_equal(with_list, with_array)
            if hasattr(fitter, 'cumulative_hazard_'):
                with_array = fitter.fit(T, C).cumulative_hazard_
                with_list = fitter.fit(list(T), list(C)).cumulative_hazard_
                assert_frame_equal(with_list, with_array)

    def test_subtraction_function(self, positive_sample_lifetimes, univariate_fitters):
        T2 = np.arange(1, 50)
        for fitter in univariate_fitters:
            f1 = fitter()
            f2 = fitter()

            f1.fit(positive_sample_lifetimes[0])
            f2.fit(T2)

            result = f1.subtract(f2)
            assert result.shape[0] == (np.unique(np.concatenate((f1.timeline, f2.timeline))).shape[0])

            npt.assert_array_almost_equal(f1.subtract(f1).sum().values, 0.0)

    def test_divide_function(self, positive_sample_lifetimes, univariate_fitters):
        T2 = np.arange(1, 50)
        for fitter in univariate_fitters:
            f1 = fitter()
            f2 = fitter()

            f1.fit(positive_sample_lifetimes[0])
            f2.fit(T2)

            result = f1.subtract(f2)
            assert result.shape[0] == (np.unique(np.concatenate((f1.timeline, f2.timeline))).shape[0])

            npt.assert_array_almost_equal(np.log(f1.divide(f1)).sum().values, 0.0)

    def test_valueerror_is_thrown_if_alpha_out_of_bounds(self, univariate_fitters):
        for fitter in univariate_fitters:
            with pytest.raises(ValueError):
                fitter(alpha=95)


class TestWeibullFitter():

    def test_weibull_fit_returns_integer_timelines(self):
        wf = WeibullFitter()
        T = np.linspace(0.1, 10)
        wf.fit(T)
        npt.assert_array_equal(wf.timeline, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        npt.assert_array_equal(wf.survival_function_.index.values, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test_weibull_model_does_not_except_negative_or_zero_values(self):
        wf = WeibullFitter()

        T = [0, 1, 2, 4, 5]
        with pytest.raises(ValueError):
            wf.fit(T)

        T[0] = -1
        with pytest.raises(ValueError):
            wf.fit(T)

    def test_exponential_data_produces_correct_inference_no_censorship(self):
        wf = WeibullFitter()
        N = 40000
        T = 5 * np.random.exponential(1, size=N) ** 2
        wf.fit(T)
        assert abs(wf.rho_ - 0.5) < 0.01
        assert abs(wf.lambda_ - 0.2) < 0.01
        assert abs(wf.median_ - 5 * np.log(2) ** 2) < 0.1  # worse convergence

    def test_exponential_data_produces_correct_inference_with_censorship(self):
        wf = WeibullFitter()
        N = 40000
        factor = 5
        T = factor * np.random.exponential(1, size=N)
        T_ = factor * np.random.exponential(1, size=N)
        wf.fit(np.minimum(T, T_), (T < T_))
        assert abs(wf.rho_ - 1.) < 0.05
        assert abs(wf.lambda_ - 1. / factor) < 0.05
        assert abs(wf.median_ - 5 * np.log(2)) < 0.1

    def test_convergence_completes_for_ever_increasing_data_sizes(self):
        wf = WeibullFitter()
        rho = 5
        lambda_ = 1. / 2
        for N in [10, 50, 500, 5000, 50000]:
            T = np.random.weibull(rho, size=N) / lambda_
            wf.fit(T)
            assert abs(1 - wf.rho_ / rho) < 5 / np.sqrt(N)
            assert abs(1 - wf.lambda_ / lambda_) < 5 / np.sqrt(N)


class TestExponentialFitter():

    def test_fit_computes_correct_lambda_(self):
        T = np.array([10, 10, 10, 10], dtype=float)
        E = np.array([1, 0, 0, 0], dtype=float)
        enf = ExponentialFitter()
        enf.fit(T, E)
        assert abs(enf.lambda_ - (E.sum() / T.sum())) < 10e-6


class TestKaplanMeierFitter():

    def kaplan_meier(self, lifetimes, observed=None):
        lifetimes_counter = Counter(lifetimes)
        km = np.zeros((len(list(lifetimes_counter.keys())), 1))
        ordered_lifetimes = np.sort(list(lifetimes_counter.keys()))
        N = len(lifetimes)
        v = 1.
        n = N * 1.0
        for i, t in enumerate(ordered_lifetimes):
            if (observed is not None):
                ix = lifetimes == t
                c = sum(1 - observed[ix])
                if n != 0:
                    v *= (1 - (lifetimes_counter.get(t) - c) / n)
                n -= lifetimes_counter.get(t)
            else:
                v *= (1 - lifetimes_counter.get(t) / n)
                n -= lifetimes_counter.get(t)
            km[i] = v
        if lifetimes_counter.get(0) is None:
            km = np.insert(km, 0, 1.)
        return km.reshape(len(km), 1)

    def test_kaplan_meier_no_censorship(self, sample_lifetimes):
        T, _ = sample_lifetimes
        kmf = KaplanMeierFitter()
        kmf.fit(T)
        npt.assert_almost_equal(kmf.survival_function_.values, self.kaplan_meier(T))

    def test_kaplan_meier_with_censorship(self, sample_lifetimes):
        T, C = sample_lifetimes
        kmf = KaplanMeierFitter()
        kmf.fit(T, C)
        npt.assert_almost_equal(kmf.survival_function_.values, self.kaplan_meier(T, C))

    def test_stat_error_is_raised_if_too_few_early_deaths(self):
        observations = np.array([1,  1,  1, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31,
                                 22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        births = observations - 1
        kmf = KaplanMeierFitter()
        with pytest.raises(StatError):
            kmf.fit(observations, entry=births)

    def test_sort_doesnt_affect_kmf(self, sample_lifetimes):
        T, _ = sample_lifetimes
        kmf = KaplanMeierFitter()
        assert_frame_equal(kmf.fit(T).survival_function_, kmf.fit(sorted(T)).survival_function_)

    def test_passing_in_left_censorship_creates_a_cumulative_density(self, sample_lifetimes):
        T, C = sample_lifetimes
        kmf = KaplanMeierFitter()
        kmf.fit(T, C, left_censorship=True)
        assert hasattr(kmf, 'cumulative_density_')
        assert hasattr(kmf, 'plot_cumulative_density_')
        assert not hasattr(kmf, 'survival_function_')

    def test_kmf_left_censorship_stats(self):
        # from http://www.public.iastate.edu/~pdixon/stat505/Chapter%2011.pdf
        T = [3, 5, 5, 5, 6, 6, 10, 12]
        C = [1, 0, 0, 1, 1, 1,  0,  1]
        kmf = KaplanMeierFitter()
        kmf.fit(T, C, left_censorship=True)

        actual = kmf.cumulative_density_[kmf._label].values
        npt.assert_almost_equal(actual, np.array([0, 0.437500, 0.5833333, 0.875, 0.875, 1]))

    def test_shifting_durations_doesnt_affect_survival_function_values(self):
        T = np.random.exponential(10, size=100)
        kmf = KaplanMeierFitter()
        expected = kmf.fit(T).survival_function_.values

        T_shifted = T + 100
        npt.assert_almost_equal(expected, kmf.fit(T_shifted).survival_function_.values)

        T_shifted = T - 50
        npt.assert_almost_equal(expected[1:], kmf.fit(T_shifted).survival_function_.values)

        T_shifted = T - 200
        npt.assert_almost_equal(expected[1:], kmf.fit(T_shifted).survival_function_.values)

    def test_kmf_survival_curve_output_against_R(self):
        df = load_g3()
        ix = df['group'] == 'RIT'
        kmf = KaplanMeierFitter()

        expected = np.array([[0.909, 0.779]]).T
        kmf.fit(df.loc[ix]['time'], df.loc[ix]['event'], timeline=[25, 53])
        npt.assert_array_almost_equal(kmf.survival_function_.values, expected, decimal=3)

        expected = np.array([[0.833, 0.667, 0.5, 0.333]]).T
        kmf.fit(df.loc[~ix]['time'], df.loc[~ix]['event'], timeline=[9, 19, 32, 34])
        npt.assert_array_almost_equal(kmf.survival_function_.values, expected, decimal=3)

    @pytest.mark.xfail()
    def test_kmf_survival_curve_output_against_R_super_accurate(self):
        df = load_g3()
        ix = df['group'] == 'RIT'
        kmf = KaplanMeierFitter()

        expected = np.array([[0.909, 0.779]]).T
        kmf.fit(df.loc[ix]['time'], df.loc[ix]['event'], timeline=[25, 53])
        npt.assert_array_almost_equal(kmf.survival_function_.values, expected, decimal=4)

        expected = np.array([[0.833, 0.667, 0.5, 0.333]]).T
        kmf.fit(df.loc[~ix]['time'], df.loc[~ix]['event'], timeline=[9, 19, 32, 34])
        npt.assert_array_almost_equal(kmf.survival_function_.values, expected, decimal=4)

    def test_kmf_confidence_intervals_output_against_R(self):
        # this uses conf.type = 'log-log'
        df = load_g3()
        ix = df['group'] != 'RIT'
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[ix]['time'], df.loc[ix]['event'], timeline=[9, 19, 32, 34])

        expected_lower_bound = np.array([0.2731, 0.1946, 0.1109, 0.0461])
        npt.assert_array_almost_equal(kmf.confidence_interval_['KM_estimate_lower_0.95'].values,
                                      expected_lower_bound, decimal=3)

        expected_upper_bound = np.array([0.975, 0.904, 0.804, 0.676])
        npt.assert_array_almost_equal(kmf.confidence_interval_['KM_estimate_upper_0.95'].values,
                                      expected_upper_bound, decimal=3)

    def test_kmf_does_not_drop_to_zero_if_last_point_is_censored(self):
        T = np.arange(0, 50, 0.5)
        E = np.random.binomial(1, 0.7, 100)
        E[np.argmax(T)] = 0
        kmf = KaplanMeierFitter()
        kmf.fit(T, E)
        assert kmf.survival_function_['KM_estimate'].iloc[-1] > 0


class TestNelsonAalenFitter():

    def nelson_aalen(self, lifetimes, observed=None):
        lifetimes_counter = Counter(lifetimes)
        na = np.zeros((len(list(lifetimes_counter.keys())), 1))
        ordered_lifetimes = np.sort(list(lifetimes_counter.keys()))
        N = len(lifetimes)
        v = 0.
        n = N * 1.0
        for i, t in enumerate(ordered_lifetimes):
            if (observed is not None):
                ix = lifetimes == t
                c = sum(1 - observed[ix])
                if n != 0:
                    v += ((lifetimes_counter.get(t) - c) / n)
                n -= lifetimes_counter.get(t)
            else:
                v += (lifetimes_counter.get(t) / n)
                n -= lifetimes_counter.get(t)
            na[i] = v
        if lifetimes_counter.get(0) is None:
            na = np.insert(na, 0, 0.)
        return na.reshape(len(na), 1)

    def test_nelson_aalen_no_censorship(self, sample_lifetimes):
        T, _ = sample_lifetimes
        naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
        naf.fit(T)
        npt.assert_almost_equal(naf.cumulative_hazard_.values, self.nelson_aalen(T))

    def test_censor_nelson_aalen(self, sample_lifetimes):
        T, C = sample_lifetimes
        naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
        naf.fit(T, C)
        npt.assert_almost_equal(naf.cumulative_hazard_.values, self.nelson_aalen(T, C))

    def test_loc_slicing(self, waltons_dataset):
        naf = NelsonAalenFitter().fit(waltons_dataset['T'])
        assert naf.cumulative_hazard_.loc[0:10].shape[0] == 4

    def test_iloc_slicing(self, waltons_dataset):
        naf = NelsonAalenFitter().fit(waltons_dataset['T'])
        assert naf.cumulative_hazard_.iloc[0:10].shape[0] == 10
        assert naf.cumulative_hazard_.iloc[0:-1].shape[0] == 32

    def test_smoothing_hazard_ties(self):
        T = np.random.binomial(20, 0.7, size=300)
        C = np.random.binomial(1, 0.8, size=300)
        naf = NelsonAalenFitter()
        naf.fit(T, C)
        naf.smoothed_hazard_(1.)

    def test_smoothing_hazard_nontied(self):
        T = np.random.exponential(20, size=300) ** 2
        C = np.random.binomial(1, 0.8, size=300)
        naf = NelsonAalenFitter()
        naf.fit(T, C)
        naf.smoothed_hazard_(1.)
        naf.fit(T)
        naf.smoothed_hazard_(1.)

    def test_smoothing_hazard_ties_all_events_observed(self):
        T = np.random.binomial(20, 0.7, size=300)
        naf = NelsonAalenFitter()
        naf.fit(T)
        naf.smoothed_hazard_(1.)

    def test_smoothing_hazard_with_spike_at_time_0(self):
        T = np.random.binomial(20, 0.7, size=300)
        T[np.random.binomial(1, 0.3, size=300).astype(bool)] = 0
        naf = NelsonAalenFitter()
        naf.fit(T)
        df = naf.smoothed_hazard_(bandwidth=0.1)
        assert df.iloc[0].values[0] > df.iloc[1].values[0]

    def test_nelson_aalen_smoothing(self):
        # this test was included because I was refactoring the estimators.
        np.random.seed(1)
        N = 10**4
        t = np.random.exponential(1, size=N)
        c = np.random.binomial(1, 0.9, size=N)
        naf = NelsonAalenFitter(nelson_aalen_smoothing=True)
        naf.fit(t, c)
        assert abs(naf.cumulative_hazard_['NA_estimate'].iloc[-1] - 8.545665) < 1e-6
        assert abs(naf.confidence_interval_['NA_estimate_upper_0.95'].iloc[-1] - 11.315662) < 1e-6
        assert abs(naf.confidence_interval_['NA_estimate_lower_0.95'].iloc[-1] - 6.4537448) < 1e-6


class TestBreslowFlemingHarringtonFitter():

    def test_BHF_fit(self):
        bfh = BreslowFlemingHarringtonFitter()

        observations = np.array([1,  1,  2, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31,
                                 22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        births = observations - 1
        bfh.fit(observations, entry=births)


class TestRegressionFitters():

    @pytest.fixture
    def regression_models(self):
        return [CoxPHFitter(), AalenAdditiveFitter(), CoxPHFitter(strata=['race', 'paro', 'mar', 'wexp'])]

    def test_pickle(self, rossi, regression_models):
        from pickle import dump
        for fitter in regression_models:
            output = stringio()
            f = fitter.fit(rossi, 'week', 'arrest')
            dump(f, output)

    def test_fit_methods_require_duration_col(self, rossi, regression_models):
        for fitter in regression_models:
            with pytest.raises(TypeError):
                fitter.fit(rossi)

    def test_fit_methods_can_accept_optional_event_col_param(self, regression_models, rossi):
        for model in regression_models:
            model.fit(rossi, 'week', event_col='arrest')
            assert_series_equal(model.event_observed.sort_index(), rossi['arrest'].astype(bool), check_names=False)

            model.fit(rossi, 'week')
            npt.assert_array_equal(model.event_observed.values, np.ones(rossi.shape[0]))

    def test_predict_methods_in_regression_return_same_types(self, regression_models, rossi):

        fitted_regression_models = map(lambda model: model.fit(rossi, duration_col='week', event_col='arrest'), regression_models)

        for fit_method in ['predict_percentile', 'predict_median', 'predict_expectation', 'predict_survival_function', 'predict_cumulative_hazard']:
            for fitter1, fitter2 in combinations(fitted_regression_models, 2):
                assert isinstance(getattr(fitter1, fit_method)(rossi), type(getattr(fitter2, fit_method)(rossi)))

    def test_duration_vector_can_be_normalized(self, regression_models, rossi):
        t = rossi['week']
        normalized_rossi = rossi.copy()
        normalized_rossi['week'] = (normalized_rossi['week'] - t.mean()) / t.std()

        for fitter in regression_models:
            # we drop indexs since aaf will have a different "time" index.
            hazards = fitter.fit(rossi, duration_col='week', event_col='arrest').hazards_.reset_index(drop=True)
            hazards_norm = fitter.fit(normalized_rossi, duration_col='week', event_col='arrest').hazards_.reset_index(drop=True)
            assert_frame_equal(hazards, hazards_norm)

    def test_prediction_methods_respect_index(self, regression_models, rossi):
        X = rossi.iloc[:4].sort_index(ascending=False)
        expected_index = pd.Index(np.array([3, 2, 1, 0]))

        for fitter in regression_models:
            fitter.fit(rossi, duration_col='week', event_col='arrest')
            npt.assert_array_equal(fitter.predict_percentile(X).index, expected_index)
            npt.assert_array_equal(fitter.predict_expectation(X).index, expected_index)
            try:
                npt.assert_array_equal(fitter.predict_partial_hazard(X).index, expected_index)
            except AttributeError:
                pass

    def test_error_is_raised_if_using_non_numeric_data(self, regression_models):
        df = pd.DataFrame.from_dict({
            't': [1., 2., 3.],
            'bool_': [True, True, False],
            'int_': [1, -1, 0],
            'uint8_': pd.Series([1, -1, 0], dtype="uint8"),
            'string_': ['test', 'a', '2.5'],
            'float_': [1.2, -0.5, 0.0],
            'categorya_': pd.Series([1, 2, 3], dtype='category'),
            'categoryb_': pd.Series(['a', 'b', 'a'], dtype='category')
        })

        for fitter in [CoxPHFitter(), AalenAdditiveFitter()]:
            for subset in [
                ['t', 'categorya_'],
                ['t', 'categoryb_'],
                ['t', 'string_'],
            ]:
                with pytest.raises(TypeError):
                    fitter.fit(df[subset], duration_col='t')

            for subset in [
                ['t', 'bool_'],
                ['t', 'int_'],
                ['t', 'float_'],
                ['t', 'uint8_'],
            ]:
                fitter.fit(df[subset], duration_col='t')

    def test_regression_model_has_score_(self, regression_models, rossi):

        for fitter in regression_models:
            assert not hasattr(fitter, 'score_')
            fitter.fit(rossi, duration_col='week', event_col='arrest')
            assert hasattr(fitter, 'score_')


class TestCoxPHFitter():

    def test_summary(self, rossi):
        cp = CoxPHFitter()
        cp.fit(rossi, duration_col='week', event_col='arrest')
        summDf = cp.summary
        expectedColumns = ['coef',
                           'exp(coef)',
                           'se(coef)',
                           'z',
                           'p',
                           'lower 0.95',
                           'upper 0.95']
        assert all([col in summDf.columns for col in expectedColumns])

    def test_print_summary(self, rossi):

        import sys
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out

            cp = CoxPHFitter()
            cp.fit(rossi, duration_col='week', event_col='arrest')
            cp.print_summary()
            output = out.getvalue().strip().split()
            expected = """n=432, number of events=114

           coef  exp(coef)  se(coef)          z         p  lower 0.95  upper 0.95
fin  -1.897e-01  8.272e-01 9.579e-02 -1.981e+00 4.763e-02  -3.775e-01  -1.938e-03   *
age  -3.500e-01  7.047e-01 1.344e-01 -2.604e+00 9.210e-03  -6.134e-01  -8.651e-02  **
race  1.032e-01  1.109e+00 1.012e-01  1.020e+00 3.078e-01  -9.516e-02   3.015e-01
wexp -7.486e-02  9.279e-01 1.051e-01 -7.124e-01 4.762e-01  -2.809e-01   1.311e-01
mar  -1.421e-01  8.675e-01 1.254e-01 -1.134e+00 2.570e-01  -3.880e-01   1.037e-01
paro -4.134e-02  9.595e-01 9.522e-02 -4.341e-01 6.642e-01  -2.280e-01   1.453e-01
prio  2.639e-01  1.302e+00 8.291e-02  3.182e+00 1.460e-03   1.013e-01   4.264e-01  **
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Concordance = 0.640""".strip().split()
            for i in [0, 1, 2, -2, -1]:
                assert output[i] == expected[i]
        finally:
            sys.stdout = saved_stdout

    def test_log_likelihood_is_available_in_output(self, data_nus):
        cox = CoxPHFitter()
        cox.fit(data_nus, duration_col='t', event_col='E')
        assert abs(cox._log_likelihood - -12.7601409152) < 0.001

    def test_efron_computed_by_hand_examples(self, data_nus):
        cox = CoxPHFitter()

        X = data_nus['x'][:, None]
        T = data_nus['t']
        E = data_nus['E']
        weights = np.ones_like(T)

        # Enforce numpy arrays
        X = np.array(X)
        T = np.array(T)
        E = np.array(E)

        # Want as bools
        E = E.astype(bool)

        # tests from http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
        beta = np.array([[0]])

        l, u, _ = cox._get_efron_values(X, beta, T, E, weights)
        l = -l

        assert np.abs(l[0][0] - 77.13) < 0.05
        assert np.abs(u[0] - -2.51) < 0.05
        beta = beta + u / l
        assert np.abs(beta - -0.0326) < 0.05

        l, u, _ = cox._get_efron_values(X, beta, T, E, weights)
        l = -l

        assert np.abs(l[0][0] - 72.83) < 0.05
        assert np.abs(u[0] - -0.069) < 0.05
        beta = beta + u / l
        assert np.abs(beta - -0.0325) < 0.01

        l, u, _ = cox._get_efron_values(X, beta, T, E, weights)
        l = -l

        assert np.abs(l[0][0] - 72.70) < 0.01
        assert np.abs(u[0] - -0.000061) < 0.01
        beta = beta + u / l
        assert np.abs(beta - -0.0335) < 0.01

    def test_efron_newtons_method(self, data_nus):
        newton = CoxPHFitter()._newton_rhaphson
        X, T, E, W = data_nus[['x']], data_nus['t'], data_nus['E'], np.ones_like(data_nus['t'])
        assert np.abs(newton(X, T, E, W)[0][0] - -0.0335) < 0.0001

    def test_fit_method(self, data_nus):
        cf = CoxPHFitter()
        cf.fit(data_nus, duration_col='t', event_col='E')
        assert np.abs(cf.hazards_.iloc[0][0] - -0.0335) < 0.0001

    def test_using_dataframes_vs_numpy_arrays(self, data_pred2):
        cf = CoxPHFitter()
        cf.fit(data_pred2, 't', 'E')

        X = data_pred2[data_pred2.columns.difference(['t', 'E'])]
        assert_frame_equal(
            cf.predict_partial_hazard(np.array(X)),
            cf.predict_partial_hazard(X)
        )

    def test_prediction_methods_will_accept_a_times_arg_to_reindex_the_predictions(self, data_pred2):
        cf = CoxPHFitter()
        cf.fit(data_pred2, duration_col='t', event_col='E')
        times_of_interest = np.arange(0, 10, 0.5)

        actual_index = cf.predict_survival_function(data_pred2.drop(["t", "E"], axis=1), times=times_of_interest).index
        np.testing.assert_allclose(actual_index.values, times_of_interest)

        actual_index = cf.predict_cumulative_hazard(data_pred2.drop(["t", "E"], axis=1), times=times_of_interest).index
        np.testing.assert_allclose(actual_index.values, times_of_interest)

    def test_data_normalization(self, data_pred2):
        # During fit, CoxPH copies the training data and normalizes it.
        # Future calls should be normalized in the same way and

        cf = CoxPHFitter()
        cf.fit(data_pred2, duration_col='t', event_col='E')

        # Internal training set
        ci_trn = cf.score_
        # New data should normalize in the exact same way
        ci_org = concordance_index(data_pred2['t'],
                                   -cf.predict_partial_hazard(data_pred2[['x1', 'x2']]).values,
                                   data_pred2['E'])

        assert ci_org == ci_trn

    @pytest.mark.xfail
    def test_cox_ph_prediction_monotonicity(self, data_pred2):
        # Concordance wise, all prediction methods should be monotonic versions
        # of one-another, unless numerical factors screw it up.
        t = data_pred2['t']
        e = data_pred2['E']
        X = data_pred2[['x1', 'x2']]

        cf = CoxPHFitter()
        cf.fit(data_pred2, duration_col='t', event_col='E')

        # Base comparison is partial_hazards
        ci_ph = concordance_index(t, -cf.predict_partial_hazard(X).values, e)

        ci_med = concordance_index(t, cf.predict_median(X).ravel(), e)
        assert ci_ph == ci_med

        ci_exp = concordance_index(t, cf.predict_expectation(X).ravel(), e)
        assert ci_ph == ci_exp

    def test_crossval_for_cox_ph_with_normalizing_times(self, data_pred2, data_pred1):
        cf = CoxPHFitter()

        for data_pred in [data_pred1, data_pred2]:

            # why does this
            data_norm = data_pred.copy()
            times = data_norm['t']
            # Normalize to mean = 0 and standard deviation = 1
            times -= np.mean(times)
            times /= np.std(times)
            data_norm['t'] = times

            scores = k_fold_cross_validation(cf, data_norm,
                                             duration_col='t',
                                             event_col='E', k=3,
                                             predictor='predict_partial_hazard')

            mean_score = 1 - np.mean(scores)

            expected = 0.9
            msg = "Expected min-mean c-index {:.2f} < {:.2f}"
            assert mean_score > expected, msg.format(expected, mean_score)

    def test_crossval_for_cox_ph(self, data_pred2, data_pred1):
        cf = CoxPHFitter()

        for data_pred in [data_pred1, data_pred2]:
            scores = k_fold_cross_validation(cf, data_pred,
                                             duration_col='t',
                                             event_col='E', k=3,
                                             predictor='predict_partial_hazard')

            mean_score = 1 - np.mean(scores)  # this is because we are using predict_partial_hazard

            expected = 0.9
            msg = "Expected min-mean c-index {:.2f} < {:.2f}"
            assert mean_score > expected, msg.format(expected, mean_score)

    def test_crossval_for_cox_ph_normalized(self, data_pred2, data_pred1):
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
                                             event_col='E', k=3,
                                             predictor='predict_partial_hazard')

            mean_score = 1 - np.mean(scores)  # this is because we are using predict_partial_hazard
            expected = 0.9
            msg = "Expected min-mean c-index {:.2f} < {:.2f}"
            assert mean_score > expected, msg.format(expected, mean_score)

    def test_coef_output_against_R_super_accurate(self, rossi):
        """
        from http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-cox-regression.pdf
        Link is now broken, but this is the code:

        library(survival)
        rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        mod.allison <- coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar + paro + prio,
            data=rossi)
        cat(round(mod.allison$coefficients, 4), sep=", ")
        """
        expected = np.array([[-0.3794, -0.0574, 0.3139, -0.1498, -0.4337, -0.0849,  0.0915]])
        cf = CoxPHFitter()
        cf.fit(rossi, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=4)

    def test_coef_output_against_R_using_weights(self, rossi):
        rossi_ = rossi.copy()
        rossi_['weights'] = 1.
        rossi_ = rossi_.groupby(rossi.columns.tolist())['weights'].sum()\
                       .reset_index()

        expected = np.array([[-0.3794, -0.0574, 0.3139, -0.1498, -0.4337, -0.0849,  0.0915]])
        cf = CoxPHFitter()
        cf.fit(rossi_, duration_col='week', event_col='arrest', weights_col='weights')
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=4)

    def test_standard_error_coef_output_against_R(self, rossi):
        """
        from http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-cox-regression.pdf
        Link is now broken, but this is the code:

        library(survival)
        rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        mod.allison <- coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar + paro + prio,
            data=rossi)
        summary(mod.allison)
        """
        expected = np.array([0.19138, 0.02200, 0.30799, 0.21222, 0.38187, 0.19576, 0.02865])
        cf = CoxPHFitter()
        cf.fit(rossi, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.summary['se(coef)'].values, expected, decimal=4)

    def test_z_value_output_against_R_to_3_decimal_places(self, rossi):
        """
        from http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-cox-regression.pdf
        Link is now broken, but this is the code:

        library(survival)
        rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        mod.allison <- coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar + paro + prio,
            data=rossi)
        summary(mod.allison)
        """
        expected = np.array([-1.983, -2.611, 1.019, -0.706, -1.136, -0.434, 3.194])
        cf = CoxPHFitter()
        cf.fit(rossi, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.summary['z'].values, expected, decimal=3)

    def test_output_with_strata_against_R(self, rossi):
        """
        rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        r = coxph(formula = Surv(week, arrest) ~ fin + age + strata(race,
                    paro, mar, wexp) + prio, data = rossi)
        """
        expected = np.array([[-0.3355, -0.0590, 0.1002]])
        cf = CoxPHFitter()
        cf.fit(rossi, duration_col='week', event_col='arrest', strata=['race', 'paro', 'mar', 'wexp'], show_progress=True)
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=4)

    def test_penalized_output_against_R(self, rossi):
        # R code:
        #
        # rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        # mod.allison <- coxph(Surv(week, arrest) ~ ridge(fin, age, race, wexp, mar, paro, prio,
        #                                                 theta=1.0, scale=TRUE), data=rossi)
        # cat(round(mod.allison$coefficients, 4), sep=", ")
        expected = np.array([[-0.3761, -0.0565, 0.3099, -0.1532, -0.4295, -0.0837, 0.0909]])
        cf = CoxPHFitter(penalizer=1.0)
        cf.fit(rossi, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=4)

    def test_coef_output_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see example 8.3 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = load_kidney_transplant(usecols=['time', 'death',
                                             'black_male', 'white_male',
                                             'black_female'])
        cf = CoxPHFitter()
        cf.fit(df, duration_col='time', event_col='death')

        # coefs
        actual_coefs = cf.hazards_.values
        expected_coefs = np.array([[0.1596, 0.2484, 0.6567]])
        npt.assert_array_almost_equal(actual_coefs, expected_coefs, decimal=3)

    def test_se_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see table 8.1 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = load_larynx()
        cf = CoxPHFitter()
        cf.fit(df, duration_col='time', event_col='death')

        # standard errors
        actual_se = cf._compute_standard_errors().values
        expected_se = np.array([[0.0143,  0.4623,  0.3561,  0.4222]])
        npt.assert_array_almost_equal(actual_se, expected_se, decimal=3)

    def test_p_value_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see table 8.1 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = load_larynx()
        cf = CoxPHFitter()
        cf.fit(df, duration_col='time', event_col='death')

        # p-values
        actual_p = cf._compute_p_values()
        expected_p = np.array([0.1847, 0.7644,  0.0730, 0.00])
        npt.assert_array_almost_equal(actual_p, expected_p, decimal=2)

    def test_input_column_order_is_equal_to_output_hazards_order(self, rossi):
        cp = CoxPHFitter()
        expected = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
        cp.fit(rossi, event_col='week', duration_col='arrest')
        assert list(cp.hazards_.columns) == expected

    def test_strata_removes_variable_from_summary_output(self, rossi):
        cp = CoxPHFitter()
        cp.fit(rossi, 'week', 'arrest', strata=['race'])
        assert 'race' not in cp.summary.index

    def test_strata_works_if_only_a_single_element_is_in_the_strata(self):
        df = load_holly_molly_polly()
        del df['Start(days)']
        del df['Stop(days)']
        del df['ID']
        cp = CoxPHFitter()
        cp.fit(df, 'T', 'Status', strata=['Stratum'])
        assert True

    def test_coxph_throws_a_explainable_error_when_predict_sees_a_strata_it_hasnt_seen(self):
        training_df = pd.DataFrame.from_records([
            {'t': 1, 'e': 1, 's1': 0, 's2': 0, 'v': 1.},
            {'t': 2, 'e': 1, 's1': 0, 's2': 0, 'v': 1.5},
            {'t': 3, 'e': 1, 's1': 0, 's2': 0, 'v': 2.5},

            {'t': 3, 'e': 1, 's1': 0, 's2': 1, 'v': 2.5},
            {'t': 4, 'e': 1, 's1': 0, 's2': 1, 'v': 2.5},
            {'t': 3, 'e': 1, 's1': 0, 's2': 1, 'v': 4.5},
        ])

        cp = CoxPHFitter()
        cp.fit(training_df, 't', 'e', strata=['s1', 's2'])

        testing_df = pd.DataFrame.from_records([
            {'t': 1, 'e': 1, 's1': 1, 's2': 0, 'v': 0.},
            {'t': 2, 'e': 1, 's1': 1, 's2': 0, 'v': 0.5},
            {'t': 3, 'e': 1, 's1': 1, 's2': 0, 'v': -0.5},
        ])

        with pytest.raises(StatError):
            cp.predict_median(testing_df)

    def test_strata_against_R_output(self, rossi):
        """
        > library(survival)
        > rossi = read.csv('.../lifelines/datasets/rossi.csv')
        > r = coxph(formula = Surv(week, arrest) ~ fin + age + strata(race,
            paro, mar, wexp) + prio, data = rossi)
        > r$loglik
        """

        cp = CoxPHFitter()
        cp.fit(rossi, 'week', 'arrest', strata=['race', 'paro', 'mar', 'wexp'])

        npt.assert_almost_equal(cp.summary['coef'].values, [-0.335, -0.059, 0.100], decimal=3)
        assert abs(cp._log_likelihood - -436.9339) / 436.9339 < 0.01

    def test_hazard_works_as_intended_with_strata_against_R_output(self, rossi):
        """
        > library(survival)
        > rossi = read.csv('.../lifelines/datasets/rossi.csv')
        > r = coxph(formula = Surv(week, arrest) ~ fin + age + strata(race,
            paro, mar, wexp) + prio, data = rossi)
        > basehaz(r, centered=TRUE)
        """
        cp = CoxPHFitter()
        cp.fit(rossi, 'week', 'arrest', strata=['race', 'paro', 'mar', 'wexp'])
        npt.assert_almost_equal(cp.baseline_cumulative_hazard_[(0, 0, 0, 0)].loc[[14, 35, 37, 43, 52]].values, [0.076600555, 0.169748261, 0.272088807, 0.396562717, 0.396562717], decimal=4)
        npt.assert_almost_equal(cp.baseline_cumulative_hazard_[(0, 0, 0, 1)].loc[[27, 43, 48, 52]].values, [0.095499001, 0.204196905, 0.338393113, 0.338393113], decimal=4)

    def test_strata_from_init_is_used_in_fit_later(self, rossi):
        strata = ['race', 'paro', 'mar']
        cp_with_strata_in_init = CoxPHFitter(strata=strata)
        cp_with_strata_in_init.fit(rossi, 'week', 'arrest')
        assert cp_with_strata_in_init.strata == strata

        cp_with_strata_in_fit = CoxPHFitter()
        cp_with_strata_in_fit.fit(rossi, 'week', 'arrest', strata=strata)
        assert cp_with_strata_in_fit.strata == strata

        assert cp_with_strata_in_init._log_likelihood == cp_with_strata_in_fit._log_likelihood

    def test_baseline_survival_is_the_same_indp_of_location(self, regression_dataset):
        df = regression_dataset.copy()
        cp1 = CoxPHFitter()
        cp1.fit(df, event_col='E', duration_col='T')

        df_demeaned = regression_dataset.copy()
        df_demeaned[['var1', 'var2', 'var3']] = df_demeaned[['var1', 'var2', 'var3']] - df_demeaned[['var1', 'var2', 'var3']].mean()
        cp2 = CoxPHFitter()
        cp2.fit(df_demeaned, event_col='E', duration_col='T')
        assert_frame_equal(cp2.baseline_survival_, cp1.baseline_survival_)

    def test_baseline_cumulative_hazard_is_the_same_indp_of_location(self, regression_dataset):
        df = regression_dataset.copy()
        cp1 = CoxPHFitter()
        cp1.fit(df, event_col='E', duration_col='T')

        df_demeaned = regression_dataset.copy()
        df_demeaned[['var1', 'var2', 'var3']] = df_demeaned[['var1', 'var2', 'var3']] - df_demeaned[['var1', 'var2', 'var3']].mean()
        cp2 = CoxPHFitter()
        cp2.fit(df_demeaned, event_col='E', duration_col='T')
        assert_frame_equal(cp2.baseline_cumulative_hazard_, cp1.baseline_cumulative_hazard_)

    def test_survival_prediction_is_the_same_indp_of_location(self, regression_dataset):
        df = regression_dataset.copy()

        df_demeaned = regression_dataset.copy()
        mean = df_demeaned[['var1', 'var2', 'var3']].mean()
        df_demeaned[['var1', 'var2', 'var3']] = df_demeaned[['var1', 'var2', 'var3']] - mean

        cp1 = CoxPHFitter()
        cp1.fit(df, event_col='E', duration_col='T')

        cp2 = CoxPHFitter()
        cp2.fit(df_demeaned, event_col='E', duration_col='T')

        assert_frame_equal(
            cp1.predict_survival_function(df.iloc[[0]][['var1', 'var2', 'var3']]),
            cp2.predict_survival_function(df_demeaned.iloc[[0]][['var1', 'var2', 'var3']])
        )

    def test_baseline_survival_is_the_same_indp_of_scale(self, regression_dataset):
        df = regression_dataset.copy()
        cp1 = CoxPHFitter()
        cp1.fit(df, event_col='E', duration_col='T')

        df_descaled = regression_dataset.copy()
        df_descaled[['var1', 'var2', 'var3']] = df_descaled[['var1', 'var2', 'var3']] / df_descaled[['var1', 'var2', 'var3']].std()
        cp2 = CoxPHFitter()
        cp2.fit(df_descaled, event_col='E', duration_col='T')
        assert_frame_equal(cp2.baseline_survival_, cp1.baseline_survival_)

    def test_survival_prediction_is_the_same_indp_of_scale(self, regression_dataset):
        df = regression_dataset.copy()

        df_scaled = regression_dataset.copy()
        df_scaled[['var1', 'var2', 'var3']] = df_scaled[['var1', 'var2', 'var3']] * 10.0

        cp1 = CoxPHFitter()
        cp1.fit(df, event_col='E', duration_col='T')

        cp2 = CoxPHFitter()
        cp2.fit(df_scaled, event_col='E', duration_col='T')

        assert_frame_equal(
            cp1.predict_survival_function(df.iloc[[0]][['var1', 'var2', 'var3']]),
            cp2.predict_survival_function(df_scaled.iloc[[0]][['var1', 'var2', 'var3']])
        )

    def test_predict_log_hazard_relative_to_mean(self, rossi):
        cox = CoxPHFitter()
        cox.fit(rossi, 'week', 'arrest')
        log_relative_hazards = cox.predict_log_hazard_relative_to_mean(rossi)
        means = rossi.mean(0).to_frame().T
        assert_frame_equal(log_relative_hazards, np.log(cox.predict_partial_hazard(rossi) / cox.predict_partial_hazard(means).squeeze()))

    def test_warning_is_raised_if_df_has_a_near_constant_column(self, rossi):
        cox = CoxPHFitter()
        rossi['constant'] = 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cox.fit(rossi, 'week', 'arrest')
            except (LinAlgError, ValueError):
                pass
            assert len(w) == 2
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "variance" in str(w[0].message)

    def test_warning_is_raised_if_df_has_a_near_constant_column_in_one_seperation(self, rossi):
        # check for a warning if we have complete seperation
        cox = CoxPHFitter()
        ix = rossi['arrest'] == 1
        rossi.loc[ix, 'paro'] = 1
        rossi.loc[~ix, 'paro'] = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cox.fit(rossi, 'week', 'arrest')
            except LinAlgError:
                pass
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "complete separation" in str(w[-1].message)

    @pytest.mark.xfail
    def test_what_happens_when_column_is_constant_for_all_non_deaths(self, rossi):
        # this is known as complete seperation: https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/
        cp = CoxPHFitter()
        ix = rossi['arrest'] == 1
        rossi.loc[ix, 'paro'] = 1
        cp.fit(rossi, 'week', 'arrest', show_progress=True)
        assert cp.summary.loc['paro', 'exp(coef)'] < 100

    @pytest.mark.xfail
    def test_what_happens_with_colinear_inputs(self, rossi):
        cp = CoxPHFitter()
        rossi['duped'] = rossi['paro'] + rossi['prio']
        cp.fit(rossi, 'week', 'arrest', show_progress=True)
        assert cp.summary.loc['duped', 'se(coef)'] < 100

    def test_durations_of_zero_are_okay(self, rossi):
        cp = CoxPHFitter()
        rossi.loc[range(10), 'week'] = 0
        cp.fit(rossi, 'week', 'arrest')


class TestAalenAdditiveFitter():

    def test_nn_cumulative_hazard_will_set_cum_hazards_to_0(self, rossi):
        aaf = AalenAdditiveFitter(nn_cumulative_hazard=False)
        aaf.fit(rossi, event_col='arrest', duration_col='week')
        cum_hazards = aaf.predict_cumulative_hazard(rossi)
        assert (cum_hazards < 0).stack().mean() > 0

        aaf = AalenAdditiveFitter(nn_cumulative_hazard=True)
        aaf.fit(rossi, event_col='arrest', duration_col='week')
        cum_hazards = aaf.predict_cumulative_hazard(rossi)
        assert (cum_hazards < 0).stack().mean() == 0

    def test_using_a_custom_timeline_in_static_fitting(self, rossi):
        aaf = AalenAdditiveFitter()
        timeline = np.arange(10)
        aaf.fit(rossi, event_col='arrest', duration_col='week', timeline=timeline)
        npt.assert_array_equal(aaf.hazards_.index.values, timeline)
        npt.assert_array_equal(aaf.cumulative_hazards_.index.values, timeline)
        npt.assert_array_equal(aaf.variance_.index.values, timeline)
        npt.assert_array_equal(aaf.timeline, timeline)

    def test_using_a_custom_timeline_in_varying_fitting(self):
        panel_dataset = load_panel_test()
        aaf = AalenAdditiveFitter()
        timeline = np.arange(10)
        aaf.fit(panel_dataset, id_col='id', duration_col='t', timeline=timeline)
        npt.assert_array_equal(aaf.hazards_.index.values, timeline)
        npt.assert_array_equal(aaf.cumulative_hazards_.index.values, timeline)
        npt.assert_array_equal(aaf.variance_.index.values, timeline)
        npt.assert_array_equal(aaf.timeline, timeline)

    def test_penalizer_reduces_norm_of_hazards(self, rossi):
        from numpy.linalg import norm

        aaf_without_penalizer = AalenAdditiveFitter(coef_penalizer=0., smoothing_penalizer=0.)
        assert aaf_without_penalizer.coef_penalizer == aaf_without_penalizer.smoothing_penalizer == 0.0
        aaf_without_penalizer.fit(rossi, event_col='arrest', duration_col='week')

        aaf_with_penalizer = AalenAdditiveFitter(coef_penalizer=10., smoothing_penalizer=10.)
        aaf_with_penalizer.fit(rossi, event_col='arrest', duration_col='week')
        assert norm(aaf_with_penalizer.cumulative_hazards_) <= norm(aaf_without_penalizer.cumulative_hazards_)

    def test_input_column_order_is_equal_to_output_hazards_order(self, rossi):
        aaf = AalenAdditiveFitter()
        expected = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
        aaf.fit(rossi, event_col='arrest', duration_col='week')
        assert list(aaf.cumulative_hazards_.columns.drop('baseline')) == expected

        aaf = AalenAdditiveFitter(fit_intercept=False)
        expected = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
        aaf.fit(rossi, event_col='arrest', duration_col='week')
        assert list(aaf.cumulative_hazards_.columns) == expected

    def test_swapping_order_of_columns_in_a_df_is_okay(self, rossi):
        aaf = AalenAdditiveFitter()
        aaf.fit(rossi, event_col='arrest', duration_col='week')

        misorder = ['age', 'race', 'wexp', 'mar', 'paro', 'prio', 'fin']
        natural_order = rossi.columns.drop(['week', 'arrest'])
        deleted_order = rossi.columns.difference(['week', 'arrest'])
        assert_frame_equal(aaf.predict_median(rossi[natural_order]), aaf.predict_median(rossi[misorder]))
        assert_frame_equal(aaf.predict_median(rossi[natural_order]), aaf.predict_median(rossi[deleted_order]))

        aaf = AalenAdditiveFitter(fit_intercept=False)
        aaf.fit(rossi, event_col='arrest', duration_col='week')
        assert_frame_equal(aaf.predict_median(rossi[natural_order]), aaf.predict_median(rossi[misorder]))
        assert_frame_equal(aaf.predict_median(rossi[natural_order]), aaf.predict_median(rossi[deleted_order]))

    def test_large_dimensions_for_recursion_error(self):
        n = 500
        d = 50
        X = pd.DataFrame(np.random.randn(n, d))
        T = np.random.exponential(size=n)
        X['T'] = T
        aaf = AalenAdditiveFitter()
        aaf.fit(X, duration_col='T')

    def test_aaf_panel_dataset_with_no_censorship(self):
        panel_dataset = load_panel_test()
        aaf = AalenAdditiveFitter()
        aaf.fit(panel_dataset, id_col='id', duration_col='t')
        expected = pd.Series([True] * 9, index=range(1, 10))
        expected.index.name = 'id'
        assert_series_equal(aaf.event_observed, expected)

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
        # fit it to Aalen's model
        aaf = AalenAdditiveFitter()
        aaf.fit(X, 'T')

        # predictions
        T_pred = aaf.predict_median(X[list(range(6))])
        assert abs((T_pred.values > T).mean() - 0.5) < 0.05

    def test_dataframe_input_with_nonstandard_index(self):
        aaf = AalenAdditiveFitter()
        df = pd.DataFrame([(16, True, True), (1, True, True), (4, False, True)],
                          columns=['duration', 'done_feeding', 'white'],
                          index=['a', 'b', 'c'])
        aaf.fit(df, duration_col='duration', event_col='done_feeding')

    def test_crossval_for_aalen_add(self, data_pred2, data_pred1):
        aaf = AalenAdditiveFitter()
        for data_pred in [data_pred1, data_pred2]:
            mean_scores = []
            for repeat in range(20):
                scores = k_fold_cross_validation(aaf, data_pred,
                                                 duration_col='t',
                                                 event_col='E', k=3)
                mean_scores.append(np.mean(scores))

            expected = 0.90
            msg = "Expected min-mean c-index {:.2f} < {:.2f}"
            assert np.mean(mean_scores) > expected, msg.format(expected, np.mean(scores))

    def test_predict_cumulative_hazard_inputs(self, data_pred1):
        aaf = AalenAdditiveFitter()
        aaf.fit(data_pred1, duration_col='t', event_col='E',)
        x = data_pred1.iloc[:5].drop(['t', 'E'], axis=1)
        y_df = aaf.predict_cumulative_hazard(x)
        y_np = aaf.predict_cumulative_hazard(x.values)
        assert_frame_equal(y_df, y_np)


class TestCoxTimeVaryingFitter():

    @pytest.fixture()
    def ctv(self):
        return CoxTimeVaryingFitter()

    @pytest.fixture()
    def dfcv(self):
        from lifelines.datasets import load_dfcv
        return load_dfcv()

    @pytest.fixture()
    def heart(self):
        return load_stanford_heart_transplants()

    def test_inference_against_known_R_output(self, ctv, dfcv):
        # from http://www.math.ucsd.edu/~rxu/math284/slect7.pdf
        ctv.fit(dfcv, id_col="id", start_col="start", stop_col="stop", event_col="event")
        npt.assert_almost_equal(ctv.summary['coef'].values, [1.826757, 0.705963], decimal=4)
        npt.assert_almost_equal(ctv.summary['se(coef)'].values, [1.229, 1.206], decimal=3)
        npt.assert_almost_equal(ctv.summary['p'].values, [0.14, 0.56], decimal=2)

    @pytest.mark.xfail()
    def test_fitter_will_raise_an_error_if_overlapping_intervals(self, ctv):
        df = pd.DataFrame.from_records([
            {'id': 1, 'start': 0, 'stop': 10, 'var': 1., 'event': 0},
            {'id': 1, 'start': 5, 'stop': 10, 'var': 1., 'event': 0},
        ])

        with warnings.catch_warnings(record=True):
            with pytest.raises(ValueError):
                ctv.fit(df, id_col="id", start_col="start", stop_col="stop", event_col="event")

    def test_warning_is_raised_if_df_has_a_near_constant_column(self, ctv, dfcv):
        dfcv['constant'] = 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                ctv.fit(dfcv, id_col="id", start_col="start", stop_col="stop", event_col="event")
            except (LinAlgError, ValueError):
                pass
            assert len(w) == 2
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "variance" in str(w[0].message)

    def test_warning_is_raised_if_df_has_a_near_constant_column_in_one_seperation(self, ctv, dfcv):
        # check for a warning if we have complete seperation
        ix = dfcv['event']
        dfcv.loc[ix, 'var3'] = 1
        dfcv.loc[~ix, 'var3'] = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                ctv.fit(dfcv, id_col="id", start_col="start", stop_col="stop", event_col="event")
            except (LinAlgError, ValueError):
                pass
            assert len(w) == 2
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "complete separation" in str(w[-1].message)

    def test_output_versus_Rs_against_standford_heart_transplant(self, ctv, heart):
        """
        library(survival)
        data(heart)
        coxph(Surv(start, stop, event) ~ age + transplant + surgery + year, data= heart)
        """
        ctv.fit(heart, id_col='id', event_col='event')
        npt.assert_almost_equal(ctv.summary['coef'].values, [0.0272, -0.1463, -0.6372, -0.0103], decimal=3)
        npt.assert_almost_equal(ctv.summary['se(coef)'].values, [0.0137, 0.0705, 0.3672, 0.3138], decimal=3)
        npt.assert_almost_equal(ctv.summary['p'].values, [0.048, 0.038, 0.083, 0.974], decimal=3)

class TestBayesianFitter():
    @pytest.mark.plottest
    @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
    def test_bayesian_fitter_low_data(self):
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt
        waltons_dataset = load_waltons()
        bf = BayesianFitter(samples=15)
        ix = waltons_dataset['group'] == 'miR-137'
        waltonT1 = waltons_dataset.loc[ix]['T']
        waltonT2 = waltons_dataset.loc[~ix]['T']
        bf.fit(waltonT1)
        ax = bf.plot(alpha=.2)
        bf.fit(waltonT2)
        bf.plot(ax=ax, alpha=0.2, c='k')
        plt.show()
        return

    def test_bayesian_fitter_large_data(self):
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt
        bf = BayesianFitter()
        bf.fit(np.random.exponential(10,size=1000))
        bf.plot()
        plt.show()
        return
