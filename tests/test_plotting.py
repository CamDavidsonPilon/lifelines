# -*- coding: utf-8 -*-
import os
import pytest
import pandas as pd
import numpy as np
import scipy

from lifelines import (
    NelsonAalenFitter,
    KaplanMeierFitter,
    CoxPHFitter,
    CoxTimeVaryingFitter,
    AalenAdditiveFitter,
    WeibullFitter,
    LogNormalFitter,
    LogLogisticFitter,
    WeibullAFTFitter,
    ExponentialFitter,
    AalenJohansenFitter,
)

from tests.test_estimation import known_parametric_univariate_fitters

from lifelines.generate_datasets import generate_random_lifetimes, generate_hazard_rates
from lifelines.plotting import plot_lifetimes, cdf_plot, qq_plot
from lifelines.datasets import (
    load_waltons,
    load_regression_dataset,
    load_lcd,
    load_panel_test,
    load_stanford_heart_transplants,
    load_rossi,
    load_multicenter_aids_cohort_study,
    load_nh4,
)
from lifelines.generate_datasets import cumulative_integral


@pytest.fixture()
def waltons():
    return load_waltons()[["T", "E"]].iloc[:50]


@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
class TestPlotting:
    @pytest.fixture
    def kmf(self):
        return KaplanMeierFitter()

    def setup_method(self, method):
        pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt

        self.plt = plt

    def test_parametric_univarite_fitters_has_plotting_methods(self, known_parametric_univariate_fitters):
        positive_sample_lifetimes = np.arange(1, 100)
        for fitter in known_parametric_univariate_fitters:
            f = fitter().fit(positive_sample_lifetimes)
            assert f.plot_cumulative_hazard() is not None
            assert f.plot_survival_function() is not None
            assert f.plot_hazard() is not None

    def test_negative_times_still_plots(self, block, kmf):
        n = 40
        T = np.linspace(-2, 3, n)
        C = np.random.randint(2, size=n)
        kmf.fit(T, C)
        ax = kmf.plot()
        self.plt.title("test_negative_times_still_plots")
        self.plt.show(block=block)
        return

    def test_kmf_plotting(self, block, kmf):
        data1 = np.random.exponential(10, size=(100))
        data2 = np.random.exponential(2, size=(200, 1))
        data3 = np.random.exponential(4, size=(500, 1))
        kmf.fit(data1, label="test label 1")
        ax = kmf.plot()
        kmf.fit(data2, label="test label 2")
        kmf.plot(ax=ax)
        kmf.fit(data3, label="test label 3")
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

    def test_naf_plotting_with_custom_colours(self, block):
        data1 = np.random.exponential(5, size=(200, 1))
        data2 = np.random.exponential(1, size=(500))
        naf = NelsonAalenFitter()
        naf.fit(data1)
        ax = naf.plot(color="r")
        naf.fit(data2)
        naf.plot(ax=ax, color="k")
        self.plt.title("test_naf_plotting_with_custom_coloirs")
        self.plt.show(block=block)
        return

    def test_ajf_plotting(self, block):
        E = [0, 1, 1, 2, 2, 0]
        T = [1, 2, 3, 4, 5, 6]
        ajf = AalenJohansenFitter().fit(T, E, event_of_interest=1)
        ajf.plot()
        self.plt.title("test_ajf_plotting")
        self.plt.show(block=block)
        return

    def test_ajf_plotting_no_confidence_intervals(self, block):
        E = [0, 1, 1, 2, 2, 0]
        T = [1, 2, 3, 4, 5, 6]
        ajf = AalenJohansenFitter(calculate_variance=False).fit(T, E, event_of_interest=1)
        ajf.plot(ci_show=False)
        self.plt.title("test_ajf_plotting_no_confidence_intervals")
        self.plt.show(block=block)
        return

    def test_ajf_plotting_with_add_count_at_risk(self, block):
        E = [0, 1, 1, 2, 2, 0]
        T = [1, 2, 3, 4, 5, 6]
        ajf = AalenJohansenFitter().fit(T, E, event_of_interest=1)
        ajf.plot(at_risk_counts=True)
        self.plt.title("test_ajf_plotting_with_add_count_at_risk")
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
        T[np.isinf(T)] = 10
        C = np.random.binomial(1, 1.0, size=n)
        X["T"] = T
        X["E"] = C

        # fit the aaf, no intercept as it is already built into X, X[2] is ones
        aaf = AalenAdditiveFitter(coef_penalizer=0.1, fit_intercept=False)

        aaf.fit(X, "T", "E")
        ax = aaf.plot(iloc=slice(0, aaf.cumulative_hazards_.shape[0] - 100))
        ax.set_xlabel("time")
        ax.set_title("test_aalen_additive_plot")
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
        X["T"] = T
        X["E"] = C

        # fit the aaf, no intercept as it is already built into X, X[2] is ones
        aaf = AalenAdditiveFitter(coef_penalizer=0.1, fit_intercept=False)
        aaf.fit(X, "T", "E")
        ax = aaf.smoothed_hazards_(1).iloc[0 : aaf.cumulative_hazards_.shape[0] - 500].plot()
        ax.set_xlabel("time")
        ax.set_title("test_aalen_additive_smoothed_plot")
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
        self.plt.title("test_naf_plotting_slice")
        self.plt.show(block=block)
        return

    def test_plot_lifetimes_calendar(self, block, waltons):
        T, E = waltons["T"], waltons["E"]
        current = 10
        birthtimes = current * np.random.uniform(size=(T.shape[0],))
        ax = plot_lifetimes(T, event_observed=E, entry=birthtimes)
        assert ax is not None
        self.plt.title("test_plot_lifetimes_calendar")
        self.plt.show(block=block)

    def test_plot_lifetimes_left_truncation(self, block, waltons):
        T, E = waltons["T"], waltons["E"]
        N = 20
        current = 10

        birthtimes = current * np.random.uniform(size=(T.shape[0],))
        ax = plot_lifetimes(T, event_observed=E, entry=birthtimes, left_truncated=True)
        assert ax is not None
        self.plt.title("test_plot_lifetimes_left_truncation")
        self.plt.show(block=block)

    def test_MACS_data_with_plot_lifetimes(self, block):
        df = load_multicenter_aids_cohort_study()

        plot_lifetimes(
            df["T"] - df["W"],
            event_observed=df["D"],
            entry=df["W"],
            event_observed_color="#383838",
            event_censored_color="#383838",
            left_truncated=True,
        )
        self.plt.ylabel("Patient Number")
        self.plt.xlabel("Years from AIDS diagnosis")
        self.plt.title("test_MACS_data_with_plot_lifetimes")
        self.plt.show(block=block)

    def test_plot_lifetimes_relative(self, block, waltons):
        T, E = waltons["T"], waltons["E"]
        ax = plot_lifetimes(T, event_observed=E)
        assert ax is not None
        self.plt.title("test_plot_lifetimes_relative")
        self.plt.show(block=block)

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
        naf.plot_hazard(bandwidth=1.0, loc=slice(0, 7.0))
        self.plt.title("test_naf_plot_cumulative_hazard_bandwidth_2")
        self.plt.show(block=block)
        return

    def test_naf_plot_cumulative_hazard_bandwith_1(self, block):
        data1 = np.random.exponential(5, size=(2000, 1)) ** 2
        naf = NelsonAalenFitter()
        naf.fit(data1)
        naf.plot_hazard(bandwidth=5.0, iloc=slice(0, 1700))
        self.plt.title("test_naf_plot_cumulative_hazard_bandwith_1")
        self.plt.show(block=block)
        return

    def test_weibull_plotting(self, block):
        T = 50 * np.random.exponential(1, size=(200, 1)) ** 2
        wf = WeibullFitter().fit(T, timeline=np.linspace(0, 5, 100))
        wf.plot_hazard()
        self.plt.title("test_weibull_plotting:hazard")
        self.plt.show(block=block)

        wf.plot_cumulative_hazard()
        self.plt.title("test_weibull_plotting:cumulative_hazard")
        self.plt.show(block=block)
        return

    def test_label_can_be_changed_on_univariate_fitters(self, block):
        T = np.random.exponential(5, size=(2000, 1)) ** 2
        wf = WeibullFitter().fit(T, timeline=np.linspace(0, 5))
        ax = wf.plot_hazard(label="abc")

        wf.plot_cumulative_hazard(ax=ax, label="123")
        self.plt.title("test_label_can_be_changed_on_univariate_fitters")
        self.plt.show(block=block)
        return

    def test_show_censor_with_discrete_date(self, block, kmf):
        T = np.random.binomial(20, 0.1, size=100)
        C = np.random.binomial(1, 0.8, size=100)
        kmf.fit(T, C).plot(show_censors=True)
        self.plt.title("test_show_censor_with_discrete_date")
        self.plt.show(block=block)
        return

    def test_show_censor_with_index_0(self, block, kmf):
        T = np.random.binomial(20, 0.9, size=100)  # lifelines should auto put a 0 in.
        C = np.random.binomial(1, 0.8, size=100)
        kmf.fit(T, C).plot(show_censors=True)
        self.plt.title("test_show_censor_with_index_0")
        self.plt.show(block=block)
        return

    def test_flat_style_with_customer_censor_styles(self, block, kmf):
        data1 = np.random.exponential(10, size=200)
        kmf.fit(data1, label="test label 1")
        kmf.plot(ci_force_lines=True, show_censors=True, censor_styles={"marker": "+", "mew": 2, "ms": 7})
        self.plt.title("test_flat_style_no_censor")
        self.plt.show(block=block)
        return

    def test_loglogs_plot(self, block, kmf):
        data1 = np.random.exponential(10, size=200)
        data2 = np.random.exponential(5, size=200)
        kmf.fit(data1, label="test label 1")
        ax = kmf.plot_loglogs()

        kmf.fit(data2, label="test label 2")
        ax = kmf.plot_loglogs(ax=ax)

        self.plt.title("test_loglogs_plot")
        self.plt.show(block=block)
        return

    def test_seaborn_doesnt_cause_kmf_plot_error(self, block, kmf, capsys):
        import seaborn as sns

        df = load_waltons()

        T = df["T"]
        E = df["E"]

        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E)
        kmf.plot()

        self.plt.title("test_seaborn_doesnt_cause_kmf_plot_error")
        self.plt.show(block=block)
        _, err = capsys.readouterr()
        assert err == ""

    def test_coxph_plotting(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot()
        self.plt.title("test_coxph_plotting")
        self.plt.show(block=block)

    def test_coxph_plotting_with_subset_of_columns(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot(columns=["var1", "var2"])
        self.plt.title("test_coxph_plotting_with_subset_of_columns")
        self.plt.show(block=block)

    def test_coxph_plot_covariate_groups(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest")
        cp.plot_covariate_groups("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_covariate_groups")
        self.plt.show(block=block)

    def test_coxph_plot_covariate_groups_with_strata(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata=["paro", "fin"])
        cp.plot_covariate_groups("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_covariate_groups_with_strata")
        self.plt.show(block=block)

    def test_coxph_plot_covariate_groups_with_single_strata(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata="paro")
        cp.plot_covariate_groups("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_covariate_groups_with_strata")
        self.plt.show(block=block)

    def test_coxph_plot_covariate_groups_with_nonnumeric_strata(self, block):
        df = load_rossi()
        df["strata"] = np.random.choice(["A", "B"], size=df.shape[0])
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata="strata")
        cp.plot_covariate_groups("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_covariate_groups_with_single_strata")
        self.plt.show(block=block)

    def test_coxph_plot_covariate_groups_with_multiple_variables(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest")
        cp.plot_covariate_groups(["age", "prio"], [[10, 0], [50, 10], [80, 90]])
        self.plt.title("test_coxph_plot_covariate_groups_with_multiple_variables")
        self.plt.show(block=block)

    def test_coxph_plot_covariate_groups_with_multiple_variables_and_strata(self, block):
        df = load_rossi()
        df["strata"] = np.random.choice(["A", "B"], size=df.shape[0])
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata="strata")
        cp.plot_covariate_groups(["age", "prio"], [[10, 0], [50, 10], [80, 90]])
        self.plt.title("test_coxph_plot_covariate_groups_with_multiple_variables_and_strata")
        self.plt.show(block=block)

    def test_coxtv_plotting_with_subset_of_columns(self, block):
        df = load_stanford_heart_transplants()
        ctv = CoxTimeVaryingFitter()
        ctv.fit(df, id_col="id", event_col="event")
        ctv.plot(columns=["age", "year"])
        self.plt.title("test_coxtv_plotting_with_subset_of_columns")
        self.plt.show(block=block)

    def test_coxtv_plotting(self, block):
        df = load_stanford_heart_transplants()
        ctv = CoxTimeVaryingFitter()
        ctv.fit(df, id_col="id", event_col="event")
        ctv.plot(fmt="o")
        self.plt.title("test_coxtv_plotting")
        self.plt.show(block=block)

    def test_kmf_left_censorship_plots(self, block):
        kmf = KaplanMeierFitter()
        lcd_dataset = load_lcd()
        alluvial_fan = lcd_dataset.loc[lcd_dataset["group"] == "alluvial_fan"]
        basin_trough = lcd_dataset.loc[lcd_dataset["group"] == "basin_trough"]
        kmf.fit_left_censoring(alluvial_fan["T"], alluvial_fan["E"], label="alluvial_fan")
        ax = kmf.plot()

        kmf.fit_left_censoring(basin_trough["T"], basin_trough["E"], label="basin_trough")
        ax = kmf.plot(ax=ax)
        self.plt.title("test_kmf_left_censorship_plots")
        self.plt.show(block=block)
        return

    def test_aalen_additive_fit_no_censor(self, block):
        n = 2500
        d = 6
        timeline = np.linspace(0, 70, 10000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        X.columns = coef.columns
        cumulative_hazards = pd.DataFrame(
            cumulative_integral(coef.values, timeline), index=timeline, columns=coef.columns
        )
        T = generate_random_lifetimes(hz, timeline)
        X["T"] = T
        X["E"] = np.random.binomial(1, 1, n)
        X[np.isinf(X)] = 10
        aaf = AalenAdditiveFitter()
        aaf.fit(X, "T", "E")

        for i in range(d + 1):
            ax = self.plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].loc[:15].plot(ax=ax)
            ax = aaf.plot(loc=slice(0, 15), ax=ax, columns=[col])
        self.plt.title("test_aalen_additive_fit_no_censor")
        self.plt.show(block=block)
        return

    def test_aalen_additive_fit_with_censor(self, block):
        n = 2500
        d = 6
        timeline = np.linspace(0, 70, 10000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        X.columns = coef.columns
        cumulative_hazards = pd.DataFrame(
            cumulative_integral(coef.values, timeline), index=timeline, columns=coef.columns
        )
        T = generate_random_lifetimes(hz, timeline)
        T[np.isinf(T)] = 10
        X["T"] = T
        X["E"] = np.random.binomial(1, 0.99, n)

        aaf = AalenAdditiveFitter()
        aaf.fit(X, "T", "E")

        for i in range(d + 1):
            ax = self.plt.subplot(d + 1, 1, i + 1)
            col = cumulative_hazards.columns[i]
            ax = cumulative_hazards[col].loc[:15].plot(ax=ax)
            ax = aaf.plot(loc=slice(0, 15), ax=ax, columns=[col])
        self.plt.title("test_aalen_additive_fit_with_censor")
        self.plt.show(block=block)
        return

    def test_weibull_aft_plotting(self, block):
        df = load_regression_dataset()
        aft = WeibullAFTFitter()
        aft.fit(df, "T", "E")
        aft.plot()
        self.plt.tight_layout()
        self.plt.title("test_weibull_aft_plotting")
        self.plt.show(block=block)

    def test_weibull_aft_plotting_with_subset_of_columns(self, block):
        df = load_regression_dataset()
        aft = WeibullAFTFitter()
        aft.fit(df, "T", "E")
        aft.plot(columns=["var1", "var2"])
        self.plt.tight_layout()
        self.plt.title("test_weibull_aft_plotting_with_subset_of_columns")
        self.plt.show(block=block)

    def test_weibull_aft_plot_covariate_groups(self, block):
        df = load_rossi()
        aft = WeibullAFTFitter()
        aft.fit(df, "week", "arrest")
        aft.plot_covariate_groups("age", [10, 50, 80])
        self.plt.tight_layout()
        self.plt.title("test_weibull_aft_plot_covariate_groups")
        self.plt.show(block=block)

    def test_weibull_aft_plot_covariate_groups_with_multiple_columns(self, block):
        df = load_rossi()
        aft = WeibullAFTFitter()
        aft.fit(df, "week", "arrest")
        aft.plot_covariate_groups(["age", "prio"], [[10, 0], [50, 10], [80, 50]])
        self.plt.tight_layout()
        self.plt.title("test_weibull_aft_plot_covariate_groups_with_multiple_columns")
        self.plt.show(block=block)

    def test_left_censorship_cdf_plots(self, block):
        df = load_nh4()
        fig, axes = self.plt.subplots(2, 2, figsize=(9, 5))
        axes = axes.reshape(4)
        for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
            model.fit_left_censoring(df["NH4.mg.per.L"], ~df["Censored"])
            ax = cdf_plot(model, ax=axes[i])
            assert ax is not None
        self.plt.suptitle("test_left_censorship_cdf_plots")
        self.plt.show(block=block)

    def test_right_censorship_cdf_plots(self, block):
        df = load_rossi()
        fig, axes = self.plt.subplots(2, 2, figsize=(9, 5))
        axes = axes.reshape(4)
        for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
            model.fit(df["week"], df["arrest"])
            ax = cdf_plot(model, ax=axes[i])
            assert ax is not None
        self.plt.suptitle("test_right_censorship_cdf_plots")
        self.plt.show(block=block)

    def test_qq_plot_left_censoring(self, block):
        df = load_nh4()
        fig, axes = self.plt.subplots(2, 2, figsize=(9, 5))
        axes = axes.reshape(4)
        for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
            model.fit_left_censoring(df["NH4.mg.per.L"], ~df["Censored"])
            ax = qq_plot(model, ax=axes[i])
            assert ax is not None
        self.plt.suptitle("test_qq_plot_left_censoring")
        self.plt.show(block=block)

    def test_qq_plot_left_censoring2(self, block):
        df = load_lcd()
        fig, axes = self.plt.subplots(2, 2, figsize=(9, 5))
        axes = axes.reshape(4)
        for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
            model.fit_left_censoring(df["T"], df["E"])
            ax = qq_plot(model, ax=axes[i])
            assert ax is not None
        self.plt.suptitle("test_qq_plot_left_censoring2")
        self.plt.show(block=block)

    def test_qq_plot_left_censoring_with_known_distribution(self, block):
        N = 300
        T_actual = scipy.stats.fisk(8, 0, 1).rvs(N)

        MIN_0 = np.percentile(T_actual, 5)
        MIN_1 = np.percentile(T_actual, 10)

        T = T_actual.copy()
        ix = np.random.randint(3, size=N)

        T = np.where(ix == 0, np.maximum(T, MIN_0), T)
        T = np.where(ix == 1, np.maximum(T, MIN_1), T)
        E = T_actual == T

        fig, axes = self.plt.subplots(2, 2, figsize=(9, 5))
        axes = axes.reshape(4)
        for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
            model.fit_left_censoring(T, E)
            ax = qq_plot(model, ax=axes[i])
            assert ax is not None
        self.plt.suptitle("test_qq_plot_left_censoring_with_known_distribution")
        self.plt.show(block=block)

    def test_qq_plot_right_censoring_with_known_distribution(self, block):
        N = 3000
        T_actual = scipy.stats.fisk(8, 0, 1).rvs(N)
        C = scipy.stats.fisk(8, 0, 1).rvs(N)
        E = T_actual < C
        T = np.minimum(T_actual, C)

        fig, axes = self.plt.subplots(2, 2, figsize=(9, 5))
        axes = axes.reshape(4)
        for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
            model.fit(T, E)
            ax = qq_plot(model, ax=axes[i])
            assert ax is not None
        self.plt.suptitle("test_qq_plot_right_censoring_with_known_distribution")
        self.plt.show(block=block)
