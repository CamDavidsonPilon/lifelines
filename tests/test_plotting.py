from __future__ import print_function

import os
import pytest
import pandas as pd
import numpy as np
from lifelines.estimation import NelsonAalenFitter, KaplanMeierFitter, AalenAdditiveFitter,\
    CoxPHFitter, CoxTimeVaryingFitter
from lifelines.generate_datasets import generate_random_lifetimes, generate_hazard_rates
from lifelines.plotting import plot_lifetimes
from lifelines.datasets import load_waltons, load_regression_dataset, load_lcd,\
    load_panel_test, load_stanford_heart_transplants
from lifelines.generate_datasets import cumulative_integral


@pytest.mark.plottest
@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
class TestPlotting():

    @pytest.fixture
    def kmf(self):
        return KaplanMeierFitter()

    def setup_method(self, method):
        pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt
        self.plt = plt

    def test_negative_times_still_plots(self, block, kmf):
        n = 40
        T = np.linspace(-2, 3, n)
        C = np.random.randint(2, size=n)
        kmf.fit(T, C)
        ax = kmf.plot()
        self.plt.title('test_negative_times_still_plots')
        self.plt.show(block=block)
        return

    def test_kmf_plotting(self, block, kmf):
        data1 = np.random.exponential(10, size=(100))
        data2 = np.random.exponential(2, size=(200, 1))
        data3 = np.random.exponential(4, size=(500, 1))
        kmf.fit(data1, label='test label 1')
        ax = kmf.plot()
        kmf.fit(data2, label='test label 2')
        kmf.plot(ax=ax)
        kmf.fit(data3, label='test label 3')
        kmf.plot(ax=ax)
        self.plt.title("test_kmf_plotting")
        self.plt.show(block=block)
        return

    def test_kmf_with_risk_counts(self, block, kmf):
        data1 = np.random.exponential(10, size=(100))
        kmf.fit(data1)
        kmf.plot(at_risk_counts=True)
        self.plt.title("test_kmf_with_risk_counts")
        self.plt.show(block=block)


    def test_kmf_with_inverted_axis(self, block, kmf):

        T = np.random.exponential(size=100)
        kmf = KaplanMeierFitter()
        kmf.fit(T, label='t2')
        ax = kmf.plot(invert_y_axis=True, at_risk_counts=True)

        T = np.random.exponential(3, size=100)
        kmf = KaplanMeierFitter()
        kmf.fit(T, label='t1')
        kmf.plot(invert_y_axis=True, ax=ax, ci_force_lines=False)

        self.plt.title("test_kmf_with_inverted_axis")
        self.plt.show(block=block)

    def test_naf_plotting_with_custom_colours(self, block):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(500))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(color="r")
        naf.fit(data2)
        naf.plot(ax=ax, c="k")
        self.plt.title('test_naf_plotting_with_custom_coloirs')
        self.plt.show(block=block)
        return

    def test_aalen_additive_plot(self, block):
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
        self.plt.show(block=block)
        return

    def test_aalen_additive_smoothed_plot(self, block):
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
        self.plt.show(block=block)
        return

    def test_naf_plotting_slice(self, block):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(200, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(loc=slice(0, None))
        naf.fit(data2)
        naf.plot(ax=ax, ci_force_lines=True, iloc=slice(100, 180))
        self.plt.title('test_naf_plotting_slice')
        self.plt.show(block=block)
        return

    def test_plot_lifetimes_calendar(self, block):
        self.plt.figure()
        t = np.linspace(0, 20, 1000)
        hz, coef, covrt = generate_hazard_rates(1, 5, t)
        N = 20
        current = 10
        birthtimes = current * np.random.uniform(size=(N,))
        T, C = generate_random_lifetimes(hz, t, size=N, censor=current - birthtimes)
        plot_lifetimes(T, event_observed=C, birthtimes=birthtimes, block=block)

    def test_plot_lifetimes_relative(self, block):
        self.plt.figure()
        t = np.linspace(0, 20, 1000)
        hz, coef, covrt = generate_hazard_rates(1, 5, t)
        N = 20
        T, C = generate_random_lifetimes(hz, t, size=N, censor=True)
        plot_lifetimes(T, event_observed=C, block=block)

    def test_naf_plot_cumulative_hazard(self, block):
        data1 = np.random.exponential(5, size=(200, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot()
        naf.plot_cumulative_hazard(ax=ax, ci_force_lines=True)
        self.plt.title("I should have plotted the same thing, but different styles + color!")
        self.plt.show(block=block)
        return

    def test_naf_plot_cumulative_hazard_bandwidth_2(self, block):
        data1 = np.random.exponential(5, size=(2000, 1))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=1., loc=slice(0, 7.))
        self.plt.title('test_naf_plot_cumulative_hazard_bandwidth_2')
        self.plt.show(block=block)
        return

    def test_naf_plot_cumulative_hazard_bandwith_1(self, block):
        data1 = np.random.exponential(5, size=(2000, 1)) ** 2
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=5., iloc=slice(0, 1700))
        self.plt.title('test_naf_plot_cumulative_hazard_bandwith_1')
        self.plt.show(block=block)
        return

    def test_show_censor_with_discrete_date(self, block, kmf):
        T = np.random.binomial(20, 0.1, size=100)
        C = np.random.binomial(1, 0.8, size=100)
        kmf.fit(T, C).plot(show_censors=True)
        self.plt.title('test_show_censor_with_discrete_date')
        self.plt.show(block=block)
        return

    def test_show_censor_with_index_0(self, block, kmf):
        T = np.random.binomial(20, 0.9, size=100)  # lifelines should auto put a 0 in.
        C = np.random.binomial(1, 0.8, size=100)
        kmf.fit(T, C).plot(show_censors=True)
        self.plt.title('test_show_censor_with_index_0')
        self.plt.show(block=block)
        return

    def test_flat_style_with_customer_censor_styles(self, block, kmf):
        data1 = np.random.exponential(10, size=200)
        kmf.fit(data1, label='test label 1')
        kmf.plot(ci_force_lines=True, show_censors=True,
                 censor_styles={'marker': '+', 'mew': 2, 'ms': 7})
        self.plt.title('test_flat_style_no_censor')
        self.plt.show(block=block)
        return

    def test_loglogs_plot(self, block, kmf):
        data1 = np.random.exponential(10, size=200)
        data2 = np.random.exponential(5, size=200)
        kmf.fit(data1, label='test label 1')
        ax = kmf.plot_loglogs()

        kmf.fit(data2, label='test label 2')
        ax = kmf.plot_loglogs(ax=ax)

        self.plt.title('test_loglogs_plot')
        self.plt.show(block=block)
        return

    def test_seaborn_doesnt_cause_kmf_plot_error(self, block, kmf, capsys):
        import seaborn as sns

        df = load_waltons()

        T = df['T']
        E = df['E']

        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E)
        kmf.plot()

        self.plt.title('test_seaborn_doesnt_cause_kmf_plot_error')
        self.plt.show(block=block)
        _, err = capsys.readouterr()
        assert err == ""

    def test_coxph_plotting(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot()
        self.plt.title('test_coxph_plotting')
        self.plt.show(block=block)

    def test_coxph_plotting_with_subset_of_columns(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot(columns=['var1', 'var2'])
        self.plt.title('test_coxph_plotting_with_subset_of_columns')
        self.plt.show(block=block)

    def test_coxph_plotting_with_subset_of_columns_and_standardized(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot(True, columns=['var1', 'var2'])
        self.plt.title('test_coxph_plotting_with_subset_of_columns_and_standardized')
        self.plt.show(block=block)

    def test_coxph_plotting_normalized(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot(True)
        self.plt.title('test_coxph_plotting_normalized')
        self.plt.show(block=block)

    def test_coxtv_plotting_with_subset_of_columns_and_standardized(self, block):
        df = load_stanford_heart_transplants()
        ctv = CoxTimeVaryingFitter()
        ctv.fit(df, id_col='id', event_col='event')
        ctv.plot(True, columns=['age', 'year'])
        self.plt.title('test_coxtv_plotting_with_subset_of_columns_and_standardized')
        self.plt.show(block=block)

    def test_kmf_left_censorship_plots(self, block):
        kmf = KaplanMeierFitter()
        lcd_dataset = load_lcd()
        alluvial_fan = lcd_dataset.loc[lcd_dataset['group'] == 'alluvial_fan']
        basin_trough = lcd_dataset.loc[lcd_dataset['group'] == 'basin_trough']
        kmf.fit(alluvial_fan['T'], alluvial_fan['C'], left_censorship=True, label='alluvial_fan')
        ax = kmf.plot()

        kmf.fit(basin_trough['T'], basin_trough['C'], left_censorship=True, label='basin_trough')
        ax = kmf.plot(ax=ax)
        self.plt.title("test_kmf_left_censorship_plots")
        self.plt.show(block=block)
        return

    def test_aaf_panel_dataset(self, block):

        panel_dataset = load_panel_test()
        aaf = AalenAdditiveFitter()
        aaf.fit(panel_dataset, id_col='id', duration_col='t', event_col='E')
        aaf.plot()
        self.plt.title("test_aaf_panel_dataset")
        self.plt.show(block=block)
        return

    def test_aalen_additive_fit_no_censor(self, block):
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
            ax = self.plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].loc[:15].plot(legend=False, ax=ax)
            ax = aaf.plot(loc=slice(0, 15), ax=ax, columns=[col], legend=False)
        self.plt.title("test_aalen_additive_fit_no_censor")
        self.plt.show(block=block)
        return

    def test_aalen_additive_fit_with_censor(self, block):
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
            ax = self.plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].loc[:15].plot(legend=False, ax=ax)
            ax = aaf.plot(loc=slice(0, 15), ax=ax, columns=[col], legend=False)
        self.plt.title("test_aalen_additive_fit_with_censor")
        self.plt.show(block=block)
        return
