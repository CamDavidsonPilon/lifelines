from __future__ import print_function

import os
import pytest
import numpy as np
from ..estimation import NelsonAalenFitter, KaplanMeierFitter, AalenAdditiveFitter
from ..generate_datasets import generate_random_lifetimes, generate_hazard_rates
from ..plotting import plot_lifetimes


@pytest.mark.plottest
@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
class TestPlotting():

    def setup_method(self, method):
        pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt
        self.plt = plt

    def test_negative_times_still_plots(self):
        n = 40
        T = np.linspace(-2, 3, n)
        C = np.random.randint(2, size=n)
        kmf = KaplanMeierFitter()
        kmf.fit(T, C)
        ax = kmf.plot()
        self.plt.title('test_negative_times_still_plots')
        self.plt.show()
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
        self.plt.title("test_kmf_plotting")
        self.plt.show()
        return

    def test_kmf_with_risk_counts(self):
        data1 = np.random.exponential(10, size=(100))
        kmf = KaplanMeierFitter()
        kmf.fit(data1)
        kmf.plot(at_risk_counts=True)
        self.plt.title("test_kmf_with_risk_counts")
        self.plt.show()

    def test_naf_plotting_with_custom_colours(self):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(500))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(color="r")
        naf.fit(data2)
        naf.plot(ax=ax, c="k")
        self.plt.title('test_naf_plotting_with_custom_coloirs')
        self.plt.show()
        return

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
        aaf = AalenAdditiveFitter(coef_penalizer=0.1, fit_intercept=False)

        aaf.fit(X, 'T', 'E')
        ax = aaf.plot(iloc=slice(0, aaf.cumulative_hazards_.shape[0] - 100))
        ax.set_xlabel("time")
        ax.set_title('test_aalen_additive_plot')
        self.plt.show()
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
        aaf = AalenAdditiveFitter(coef_penalizer=0.1, fit_intercept=False)
        aaf.fit(X, 'T', 'E')
        ax = aaf.smoothed_hazards_(1).iloc[0:aaf.cumulative_hazards_.shape[0] - 500].plot()
        ax.set_xlabel("time")
        ax.set_title('test_aalen_additive_smoothed_plot')
        self.plt.show()
        return

    def test_naf_plotting_slice(self):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(200, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(ix=slice(0, None))
        naf.fit(data2)
        naf.plot(ax=ax, ci_force_lines=True, iloc=slice(100, 180))
        self.plt.title('test_naf_plotting_slice')
        self.plt.show()
        return

    def test_plot_lifetimes_calendar(self):
        self.plt.figure()
        t = np.linspace(0, 20, 1000)
        hz, coef, covrt = generate_hazard_rates(1, 5, t)
        N = 20
        current = 10
        birthtimes = current * np.random.uniform(size=(N,))
        T, C = generate_random_lifetimes(hz, t, size=N, censor=current - birthtimes)
        plot_lifetimes(T, event_observed=C, birthtimes=birthtimes)

    def test_plot_lifetimes_relative(self):
        self.plt.figure()
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
        self.plt.title("I should have plotted the same thing, but different styles + color!")
        self.plt.show()
        return

    def test_naf_plot_cumulative_hazard_bandwidth_2(self):
        data1 = np.random.exponential(5, size=(2000, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=1., ix=slice(0, 7.))
        self.plt.title('test_naf_plot_cumulative_hazard_bandwidth_2')
        self.plt.show()
        return

    def test_naf_plot_cumulative_hazard_bandwith_1(self):
        data1 = np.random.exponential(5, size=(2000, 1)) ** 2
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=5., iloc=slice(0, 1700))
        self.plt.title('test_naf_plot_cumulative_hazard_bandwith_1')
        self.plt.show()
        return

    def test_show_censor_with_discrete_date(self):
        T = np.random.binomial(20, 0.1, size=100)
        C = np.random.binomial(1, 0.8, size=100)
        kmf = KaplanMeierFitter()
        kmf.fit(T, C).plot(show_censors=True)
        self.plt.title('test_show_censor_with_discrete_date')
        self.plt.show()
        return

    def test_show_censor_with_index_0(self):
        T = np.random.binomial(20, 0.9, size=100)  # lifelines should auto put a 0 in.
        C = np.random.binomial(1, 0.8, size=100)
        kmf = KaplanMeierFitter()
        kmf.fit(T, C).plot(show_censors=True)
        self.plt.title('test_show_censor_with_index_0')
        self.plt.show()
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
        self.plt.title("testing kmf flat styling + marker")
        self.plt.show()
        return

    def test_flat_style_no_censor(self):
        data1 = np.random.exponential(10, size=200)
        kmf = KaplanMeierFitter()
        kmf.fit(data1, label='test label 1')
        ax = kmf.plot(flat=True, censor_styles={'marker': '+', 'mew': 2, 'ms': 7})
        self.plt.title('test_flat_style_no_censor')
        self.plt.show()
        return
