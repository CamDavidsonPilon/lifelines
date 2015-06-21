from __future__ import print_function
from collections import Counter, Iterable
import os

import numpy as np
import pandas as pd
import pytest

from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy.testing as npt

from ..utils import k_fold_cross_validation, StatError
from ..estimation import CoxPHFitter, AalenAdditiveFitter, KaplanMeierFitter, \
    NelsonAalenFitter, BreslowFlemingHarringtonFitter, ExponentialFitter, \
    WeibullFitter, BaseFitter
from ..datasets import load_regression_dataset, load_larynx, load_waltons, load_kidney_transplant, load_rossi,\
    load_lcd, load_panel_test, load_g3, load_holly_molly_polly
from ..generate_datasets import generate_hazard_rates, generate_random_lifetimes, cumulative_integral
from ..utils import concordance_index


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

    def test_predict_methods_returns_a_scalar_or_a_array_depending_on_input(self, sample_lifetimes):
        kmf = KaplanMeierFitter()
        kmf.fit(sample_lifetimes[0])
        assert not isinstance(kmf.predict(1), Iterable)
        assert isinstance(kmf.predict([1, 2]), Iterable)

    def test_predict_method_returns_exact_value_if_given_an_observed_time(self):
        T = [1, 2, 3]
        kmf = KaplanMeierFitter()
        kmf.fit(T)
        time = 1
        assert abs(kmf.predict(time) - kmf.survival_function_.ix[time].values) < 10e-8

    def test_predict_method_returns_gives_values_prior_to_the_value_in_the_survival_function(self):
        T = [1, 2, 3]
        kmf = KaplanMeierFitter()
        kmf.fit(T)
        assert abs(kmf.predict(0.5) - kmf.survival_function_.ix[0].values) < 10e-8
        assert abs(kmf.predict(1.9999) - kmf.survival_function_.ix[1].values) < 10e-8

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
        T2 = np.arange(1, 50)
        for fitter in univariate_fitters:
            with pytest.raises(ValueError):
                f = fitter(alpha=95)


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
        T = [3, 5, 5, 5, 6, 6, 10, 12]
        C = [1, 0, 0, 1, 1, 1, 0, 1]
        kmf = KaplanMeierFitter()
        kmf.fit(T, C, left_censorship=True)
        assert kmf.cumulative_density_[kmf._label].ix[0] == 0.0
        assert kmf.cumulative_density_[kmf._label].ix[12] == 1.0

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

    @pytest.mark.plottest
    @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
    def test_kmf_left_censorship_plots(self):
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt

        kmf = KaplanMeierFitter()
        lcd_dataset = load_lcd()
        alluvial_fan = lcd_dataset.ix[lcd_dataset['group'] == 'alluvial_fan']
        basin_trough = lcd_dataset.ix[lcd_dataset['group'] == 'basin_trough']
        kmf.fit(alluvial_fan['T'], alluvial_fan['C'], left_censorship=True, label='alluvial_fan')
        ax = kmf.plot()

        kmf.fit(basin_trough['T'], basin_trough['C'], left_censorship=True, label='basin_trough')
        ax = kmf.plot(ax=ax)
        plt.show()
        return

    def test_kmf_survival_curve_output_against_R(self):
        df = load_g3()
        ix = df['group'] == 'RIT'
        kmf = KaplanMeierFitter()

        expected = np.array([[0.909, 0.779]]).T
        kmf.fit(df.ix[ix]['time'], df.ix[ix]['event'], timeline=[25, 53])
        npt.assert_array_almost_equal(kmf.survival_function_.values, expected, decimal=3)

        expected = np.array([[0.833, 0.667, 0.5, 0.333]]).T
        kmf.fit(df.ix[~ix]['time'], df.ix[~ix]['event'], timeline=[9, 19, 32, 34])
        npt.assert_array_almost_equal(kmf.survival_function_.values, expected, decimal=3)

    def test_kmf_confidence_intervals_output_against_R(self):
        # this uses conf.type = 'log-log'
        df = load_g3()
        ix = df['group'] != 'RIT'
        kmf = KaplanMeierFitter()
        kmf.fit(df.ix[ix]['time'], df.ix[ix]['event'], timeline=[9, 19, 32, 34])

        expected_lower_bound = np.array([0.2731, 0.1946, 0.1109, 0.0461])
        npt.assert_array_almost_equal(kmf.confidence_interval_['KM_estimate_lower_0.95'].values,
                                      expected_lower_bound, decimal=3)

        expected_upper_bound = np.array([0.975, 0.904, 0.804, 0.676])
        npt.assert_array_almost_equal(kmf.confidence_interval_['KM_estimate_upper_0.95'].values,
                                      expected_upper_bound, decimal=3)


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

    def test_ix_slicing(self, waltons_dataset):
        naf = NelsonAalenFitter().fit(waltons_dataset['T'])
        assert naf.cumulative_hazard_.ix[0:10].shape[0] == 4

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


class TestBreslowFlemingHarringtonFitter():

    def test_BHF_fit(self):
        bfh = BreslowFlemingHarringtonFitter()

        observations = np.array([1,  1,  2, 22, 30, 28, 32, 11, 14, 36, 31, 33, 33, 37, 35, 25, 31,
                                 22, 26, 24, 35, 34, 30, 35, 40, 39,  2])
        births = observations - 1
        bfh.fit(observations, entry=births)


class TestRegressionFitters():

    def test_fit_methods_require_duration_col(self):
        X = load_regression_dataset()

        aaf = AalenAdditiveFitter()
        cph = CoxPHFitter()

        with pytest.raises(TypeError):
            aaf.fit(X)
        with pytest.raises(TypeError):
            cph.fit(X)

    def test_fit_methods_can_accept_optional_event_col_param(self):
        X = load_regression_dataset()

        aaf = AalenAdditiveFitter()
        aaf.fit(X, 'T', event_col='E')
        assert_series_equal(aaf.event_observed.sort_index(), X['E'].astype(bool), check_names=False)

        aaf.fit(X, 'T')
        npt.assert_array_equal(aaf.event_observed.values, np.ones(X.shape[0]))

        cph = CoxPHFitter()
        cph.fit(X, 'T', event_col='E')
        assert_series_equal(cph.event_observed.sort_index(), X['E'].astype(bool), check_names=False)

        cph.fit(X, 'T')
        npt.assert_array_equal(cph.event_observed.values, np.ones(X.shape[0]))

    def test_predict_methods_in_regression_return_same_types(self):
        X = load_regression_dataset()
        x = X[X.columns - ['T', 'E']]

        aaf = AalenAdditiveFitter()
        cph = CoxPHFitter()

        aaf.fit(X, duration_col='T', event_col='E')
        cph.fit(X, duration_col='T', event_col='E')

        for fit_method in ['predict_percentile', 'predict_median', 'predict_expectation', 'predict_survival_function', 'predict', 'predict_cumulative_hazard']:
            assert isinstance(getattr(aaf, fit_method)(x), type(getattr(cph, fit_method)(x)))

    def test_duration_vector_can_be_normalized(self):
        df = load_kidney_transplant()
        t = df['time']
        normalized_df = df.copy()
        normalized_df['time'] = (normalized_df['time'] - t.mean()) / t.std()

        for fitter in [CoxPHFitter(), AalenAdditiveFitter()]:
            # we drop indexs since aaf will have a different "time" index.
            hazards = fitter.fit(df, duration_col='time', event_col='death').hazards_.reset_index(drop=True)
            hazards_norm = fitter.fit(normalized_df, duration_col='time', event_col='death').hazards_.reset_index(drop=True)
            assert_frame_equal(hazards, hazards_norm)

    def test_prediction_methods_respect_index(self, data_pred2):
        x = data_pred2[['x1', 'x2']].ix[:3].sort_index(ascending=False)
        expected_index = pd.Index(np.array([3, 2, 1, 0]))

        cph = CoxPHFitter()
        cph.fit(data_pred2, duration_col='t', event_col='E')
        npt.assert_array_equal(cph.predict_partial_hazard(x).index, expected_index)
        npt.assert_array_equal(cph.predict_percentile(x).index, expected_index)
        npt.assert_array_equal(cph.predict(x).index, expected_index)
        npt.assert_array_equal(cph.predict_expectation(x).index, expected_index)

        aaf = AalenAdditiveFitter()
        aaf.fit(data_pred2, duration_col='t', event_col='E')
        npt.assert_array_equal(aaf.predict_percentile(x).index, expected_index)
        npt.assert_array_equal(aaf.predict(x).index, expected_index)
        npt.assert_array_equal(aaf.predict_expectation(x).index, expected_index)


class TestCoxPHFitter():

    def test_summary(self):

        cp = CoxPHFitter()
        df = load_rossi()
        cp.fit(df, duration_col='week', event_col='arrest')
        summDf = cp.summary
        expectedColumns = ['coef',
                           'exp(coef)',
                           'se(coef)',
                           'z',
                           'p',
                           'lower 0.95',
                           'upper 0.95']
        assert all([col in summDf.columns for col in expectedColumns])

    def test_print_summary(self):

        import sys
        try:
            from StringIO import StringIO
        except:
            from io import StringIO

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out

            cp = CoxPHFitter()
            df = load_rossi()
            cp.fit(df, duration_col='week', event_col='arrest')
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
        cox.fit(data_nus, duration_col='t', event_col='E', include_likelihood=True)
        assert abs(cox._log_likelihood - -12.7601409152) < 0.001

    def test_efron_computed_by_hand_examples(self, data_nus):
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

    def test_efron_newtons_method(self, data_nus):
        newton = CoxPHFitter()._newton_rhaphson
        X, T, E = data_nus[['x']], data_nus['t'], data_nus['E']
        assert np.abs(newton(X, T, E)[0][0] - -0.0335) < 0.0001

    def test_fit_method(self, data_nus):
        cf = CoxPHFitter(normalize=False)
        cf.fit(data_nus, duration_col='t', event_col='E')
        assert np.abs(cf.hazards_.ix[0][0] - -0.0335) < 0.0001

    def test_using_dataframes_vs_numpy_arrays(self, data_pred2):
        # First without normalization
        cf = CoxPHFitter(normalize=False)
        cf.fit(data_pred2, 't', 'E')

        X = data_pred2[cf.data.columns]
        hazards = cf.predict_partial_hazard(X)

        # A Numpy array should return the same result
        hazards_n = cf.predict_partial_hazard(np.array(X))
        assert np.all(hazards == hazards_n)

        # Now with normalization
        cf = CoxPHFitter(normalize=True)
        cf.fit(data_pred2, 't', 'E')

        hazards = cf.predict_partial_hazard(X)

        # Compare with array argument
        hazards_n = cf.predict_partial_hazard(np.array(X))
        assert np.all(hazards == hazards_n)

    def test_data_normalization(self, data_pred2):
        # During fit, CoxPH copies the training data and normalizes it.
        # Future calls should be normalized in the same way and
        # internal training set should not be saved in a normalized state.

        cf = CoxPHFitter(normalize=True)
        cf.fit(data_pred2, duration_col='t', event_col='E')

        # Internal training set
        ci_trn = concordance_index(cf.durations,
                                   -cf.predict_partial_hazard(cf.data).values,
                                   cf.event_observed)
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

        for normalize in [True, False]:
            msg = ("Predict methods should get the same concordance" +
                   " when {}normalizing".format('' if normalize else 'not '))
            cf = CoxPHFitter(normalize=normalize)
            cf.fit(data_pred2, duration_col='t', event_col='E')

            # Base comparison is partial_hazards
            ci_ph = concordance_index(t, -cf.predict_partial_hazard(X).values, e)

            ci_med = concordance_index(t, cf.predict_median(X).ravel(), e)
            assert ci_ph == ci_med, msg

            ci_exp = concordance_index(t, cf.predict_expectation(X).ravel(), e)
            assert ci_ph == ci_exp, msg

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

    def test_output_against_R(self):
        # from http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-cox-regression.pdf
        # Link is now broken, but this is the code:
        #
        # rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        # mod.allison <- coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar + paro + prio,
        #     data=rossi)
        # cat(round(mod.allison$coefficients, 4), sep=", ")
        expected = np.array([[-0.3794, -0.0574, 0.3139, -0.1498, -0.4337, -0.0849,  0.0915]])
        df = load_rossi()
        cf = CoxPHFitter(normalize=False)
        cf.fit(df, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=3)

    def test_penalized_output_against_R(self):
        # R code:
        #
        # rossi <- read.csv('.../lifelines/datasets/rossi.csv')
        # mod.allison <- coxph(Surv(week, arrest) ~ ridge(fin, age, race, wexp, mar, paro, prio,
        #                                                 theta=1.0, scale=FALSE), data=rossi)
        # cat(round(mod.allison$coefficients, 4), sep=", ")
        expected = np.array([[-0.3641, -0.0580, 0.2894, -0.1496, -0.3837, -0.0822, 0.0913]])
        df = load_rossi()
        cf = CoxPHFitter(normalize=False, penalizer=1.0)
        cf.fit(df, duration_col='week', event_col='arrest')
        npt.assert_array_almost_equal(cf.hazards_.values, expected, decimal=3)

    def test_coef_output_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see example 8.3 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = load_kidney_transplant(usecols=['time', 'death',
                                             'black_male', 'white_male',
                                             'black_female'])
        cf = CoxPHFitter(normalize=False)
        cf.fit(df, duration_col='time', event_col='death')

        # coefs
        actual_coefs = cf.hazards_.values
        expected_coefs = np.array([[0.1596, 0.2484, 0.6567]])
        npt.assert_array_almost_equal(actual_coefs, expected_coefs, decimal=4)

    def test_se_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see table 8.1 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = load_larynx()
        cf = CoxPHFitter(normalize=False)
        cf.fit(df, duration_col='time', event_col='death')

        # standard errors
        actual_se = cf._compute_standard_errors().values
        expected_se = np.array([[0.0143,  0.4623,  0.3561,  0.4222]])
        npt.assert_array_almost_equal(actual_se, expected_se, decimal=2)

    def test_p_value_against_Survival_Analysis_by_John_Klein_and_Melvin_Moeschberger(self):
        # see table 8.1 in Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition
        df = load_larynx()
        cf = CoxPHFitter()
        cf.fit(df, duration_col='time', event_col='death')

        # p-values
        actual_p = cf._compute_p_values()
        expected_p = np.array([0.1847, 0.7644,  0.0730, 0.00])
        npt.assert_array_almost_equal(actual_p, expected_p, decimal=2)

    def test_input_column_order_is_equal_to_output_hazards_order(self):
        rossi = load_rossi()
        cp = CoxPHFitter()
        expected = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
        cp.fit(rossi, event_col='week', duration_col='arrest')
        assert list(cp.hazards_.columns) == expected

    def test_strata_removes_variable_from_summary_output(self):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, 'week', 'arrest', strata=['race'])
        assert 'race' not in cp.summary.index

    def test_strata_works_if_only_a_single_element_is_in_the_strata(self):
        df = load_holly_molly_polly()
        del df['Start(days)']
        del df['Stop(days)']
        del df['ID']
        cp = CoxPHFitter()
        cp.fit(df, 'T', 'Status', strata=['Stratum'])
        assert True

    def test_strata_against_r_output(self):
        """
        > r = coxph(formula = Surv(week, arrest) ~ fin + age + strata(race,
            paro, mar, wexp) + prio, data = rossi)
        > r
        > r$loglik
        """

        df = load_rossi()
        cp = CoxPHFitter(normalize=False)
        cp.fit(df, 'week', 'arrest', strata=['race', 'paro', 'mar', 'wexp'], include_likelihood=True)

        npt.assert_almost_equal(cp.summary['coef'].values, [-0.335, -0.059, 0.100], decimal=3)
        assert abs(cp._log_likelihood - -436.9339) / 436.9339 < 0.01


class TestAalenAdditiveFitter():

    def test_using_a_custom_timeline_in_static_fitting(self):
        rossi = load_rossi()
        aaf = AalenAdditiveFitter()
        timeline = np.arange(10)
        aaf.fit(rossi, event_col='week', duration_col='arrest', timeline=timeline)
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

    def test_penalizer_reduces_norm_of_hazards(self):
        from numpy.linalg import norm
        rossi = load_rossi()

        aaf_without_penalizer = AalenAdditiveFitter(coef_penalizer=0., smoothing_penalizer=0.)
        assert aaf_without_penalizer.coef_penalizer == aaf_without_penalizer.smoothing_penalizer == 0.0
        aaf_without_penalizer.fit(rossi, event_col='week', duration_col='arrest')

        aaf_with_penalizer = AalenAdditiveFitter(coef_penalizer=10., smoothing_penalizer=10.)
        aaf_with_penalizer.fit(rossi, event_col='week', duration_col='arrest')
        assert norm(aaf_with_penalizer.cumulative_hazards_) <= norm(aaf_without_penalizer.cumulative_hazards_)

    def test_input_column_order_is_equal_to_output_hazards_order(self):
        rossi = load_rossi()
        aaf = AalenAdditiveFitter()
        expected = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
        aaf.fit(rossi, event_col='week', duration_col='arrest')
        assert list(aaf.cumulative_hazards_.columns.drop('baseline')) == expected

        aaf = AalenAdditiveFitter(fit_intercept=False)
        expected = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
        aaf.fit(rossi, event_col='week', duration_col='arrest')
        assert list(aaf.cumulative_hazards_.columns) == expected

    def test_swapping_order_of_columns_in_a_df_is_okay(self):
        rossi = load_rossi()
        aaf = AalenAdditiveFitter()
        aaf.fit(rossi, event_col='week', duration_col='arrest')

        misorder = ['age', 'race', 'wexp', 'mar', 'paro', 'prio', 'fin']
        natural_order = rossi.columns.drop(['week', 'arrest'])
        deleted_order = rossi.columns - ['week', 'arrest']
        assert_frame_equal(aaf.predict_median(rossi[natural_order]), aaf.predict_median(rossi[misorder]))
        assert_frame_equal(aaf.predict_median(rossi[natural_order]), aaf.predict_median(rossi[deleted_order]))

        aaf = AalenAdditiveFitter(fit_intercept=False)
        aaf.fit(rossi, event_col='week', duration_col='arrest')
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

    @pytest.mark.plottest
    @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
    def test_aaf_panel_dataset(self):
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt

        panel_dataset = load_panel_test()
        aaf = AalenAdditiveFitter()
        aaf.fit(panel_dataset, id_col='id', duration_col='t', event_col='E')
        aaf.plot()

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

    @pytest.mark.plottest
    @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
    def test_aalen_additive_fit_no_censor(self):
        # this is a visual test of the fitting the cumulative
        # hazards.
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt

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
        aaf = AalenAdditiveFitter()
        aaf.fit(X, 'T', 'E')

        for i in range(d + 1):
            ax = plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].ix[:15].plot(legend=False, ax=ax)
            ax = aaf.plot(ix=slice(0, 15), ax=ax, columns=[col], legend=False)
        plt.show()
        return

    @pytest.mark.plottest
    @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
    def test_aalen_additive_fit_with_censor(self):
        # this is a visual test of the fitting the cumulative
        # hazards.
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt

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

        aaf = AalenAdditiveFitter()
        aaf.fit(X, 'T', 'E')

        for i in range(d + 1):
            ax = plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].ix[:15].plot(legend=False, ax=ax)
            ax = aaf.plot(ix=slice(0, 15), ax=ax, columns=[col], legend=False)
        plt.show()

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
            assert np.mean(mean_scores) > expected, msg.format(expected, scores.mean())

    def test_predict_cumulative_hazard_inputs(self, data_pred1):
        aaf = AalenAdditiveFitter()
        aaf.fit(data_pred1, duration_col='t', event_col='E',)
        x = data_pred1.ix[:5].drop(['t', 'E'], axis=1)
        y_df = aaf.predict_cumulative_hazard(x)
        y_np = aaf.predict_cumulative_hazard(x.values)
        assert_frame_equal(y_df, y_np)
