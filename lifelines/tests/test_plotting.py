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
    BreslowFlemingHarringtonFitter,
)

from lifelines.tests.test_estimation import known_parametric_univariate_fitters

from lifelines.generate_datasets import generate_random_lifetimes, generate_hazard_rates
from lifelines.plotting import plot_lifetimes, cdf_plot, qq_plot, rmst_plot, add_at_risk_counts
from lifelines.datasets import (
    load_waltons,
    load_regression_dataset,
    load_lcd,
    load_panel_test,
    load_stanford_heart_transplants,
    load_rossi,
    load_multicenter_aids_cohort_study,
    load_nh4,
    load_diabetes,
)
from lifelines.generate_datasets import cumulative_integral
from lifelines.calibration import survival_probability_calibration


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

    def test_parametric_univariate_fitters_has_hazard_plotting_methods(self, block, known_parametric_univariate_fitters):
        positive_sample_lifetimes = np.arange(1, 100)
        for fitter in known_parametric_univariate_fitters:
            f = fitter().fit(positive_sample_lifetimes)
            assert f.plot_hazard() is not None
        self.plt.title("test_parametric_univariate_fitters_has_hazard_plotting_methods")
        self.plt.show(block=block)

    def test_parametric_univaraite_fitters_has_cumhazard_plotting_methods(self, block, known_parametric_univariate_fitters):
        positive_sample_lifetimes = np.arange(1, 100)
        for fitter in known_parametric_univariate_fitters:
            f = fitter().fit(positive_sample_lifetimes)
            assert f.plot_cumulative_hazard() is not None

        self.plt.title("test_parametric_univaraite_fitters_has_cumhazard_plotting_methods")
        self.plt.show(block=block)

    def test_parametric_univariate_fitters_has_survival_plotting_methods(self, block, known_parametric_univariate_fitters):
        positive_sample_lifetimes = np.arange(1, 100)
        for fitter in known_parametric_univariate_fitters:
            f = fitter().fit(positive_sample_lifetimes)
            assert f.plot_survival_function() is not None

        self.plt.title("test_parametric_univariate_fitters_has_survival_plotting_methods")
        self.plt.show(block=block)

    def test_negative_times_still_plots(self, block, kmf):
        n = 40
        T = np.linspace(-2, 3, n)
        C = np.random.randint(2, size=n)
        kmf.fit(T, C)
        ax = kmf.plot_survival_function()
        self.plt.title("test_negative_times_still_plots")
        self.plt.show(block=block)
        return

    def test_kmf_plotting(self, block, kmf):
        data1 = np.random.exponential(10, size=(100))
        data2 = np.random.exponential(2, size=(200, 1))
        data3 = np.random.exponential(4, size=(500, 1))
        kmf.fit(data1, label="test label 1")
        ax = kmf.plot_survival_function()
        kmf.fit(data2, label="test label 2")
        kmf.plot_survival_function(ax=ax)
        kmf.fit(data3, label="test label 3")
        kmf.plot_survival_function(ax=ax)
        self.plt.title("test_kmf_plotting")
        self.plt.show(block=block)
        return

    def test_kmf_with_risk_counts(self, block, kmf):
        data1 = np.random.exponential(10, size=(100))
        kmf.fit(data1)
        kmf.plot_survival_function(at_risk_counts=True)
        self.plt.title("test_kmf_with_risk_counts")
        self.plt.show(block=block)

    def test_kmf_add_at_risk_counts_with_subplot(self, block, kmf):
        T = np.random.exponential(10, size=(100))
        E = np.random.binomial(1, 0.8, size=(100))
        kmf.fit(T, E)

        fig = self.plt.figure()
        axes = fig.subplots(1, 2)
        kmf.plot_survival_function(ax=axes[0])
        add_at_risk_counts(kmf, ax=axes[0])
        kmf.plot_survival_function(ax=axes[1])

        self.plt.title("test_kmf_add_at_risk_counts_with_subplot")
        self.plt.show(block=block)

    def test_kmf_add_at_risk_counts_with_specific_rows(self, block, kmf):
        T = np.random.exponential(10, size=(100))
        E = np.random.binomial(1, 0.8, size=(100))
        kmf.fit(T, E)

        fig = self.plt.figure()
        ax = fig.subplots(1, 1)
        kmf.plot_survival_function(ax=ax)
        add_at_risk_counts(kmf, ax=ax, rows_to_show=["Censored", "At risk"])
        self.plt.tight_layout()
        self.plt.title("test_kmf_add_at_risk_counts_with_specific_rows")
        self.plt.show(block=block)

    def test_kmf_add_at_risk_counts_with_single_row_multi_groups(self, block, kmf):
        T = np.random.exponential(10, size=(100))
        E = np.random.binomial(1, 0.8, size=(100))
        kmf_test = KaplanMeierFitter().fit(T, E, label="test")

        T = np.random.exponential(15, size=(1000))
        E = np.random.binomial(1, 0.6, size=(1000))
        kmf_con = KaplanMeierFitter().fit(T, E, label="con")

        fig = self.plt.figure()
        ax = fig.subplots(1, 1)

        kmf_test.plot(ax=ax)
        kmf_con.plot(ax=ax)

        ax.set_ylim([0.0, 1.1])
        ax.set_xlim([0.0, 100])
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival probability")

        add_at_risk_counts(kmf_test, kmf_con, ax=ax, rows_to_show=["At risk"], ypos=-0.4)
        self.plt.title("test_kmf_add_at_risk_counts_with_single_row_multi_groups")
        self.plt.tight_layout()
        self.plt.show(block=block)

    def test_kmf_add_at_risk_counts_with_custom_subplot(self, block, kmf):
        # https://github.com/CamDavidsonPilon/lifelines/issues/991#issuecomment-614427882
        import lifelines
        import matplotlib as mpl
        from lifelines.datasets import load_waltons

        plt = self.plt
        waltons = load_waltons()
        ix = waltons["group"] == "control"

        img_no = 3

        height = 4 * img_no
        half_inch = 0.5 / height  # in percent height
        _fig = plt.figure(figsize=(6, height), dpi=100)
        gs = mpl.gridspec.GridSpec(img_no, 1)
        # plt.subplots_adjust(left=0.08, right=0.98, bottom=half_inch, top=1 - half_inch)

        for i in range(img_no):
            ax = plt.subplot(gs[i, 0])
            kmf_control = lifelines.KaplanMeierFitter()
            ax = kmf_control.fit(waltons.loc[ix]["T"], waltons.loc[ix]["E"], label="control").plot(ax=ax)
            kmf_exp = lifelines.KaplanMeierFitter()
            ax = kmf_exp.fit(waltons.loc[~ix]["T"], waltons.loc[~ix]["E"], label="exp").plot(ax=ax)
            ax = lifelines.plotting.add_at_risk_counts(kmf_exp, kmf_control, ax=ax)

        plt.subplots_adjust(hspace=0.6)
        plt.title("test_kmf_add_at_risk_counts_with_custom_subplot")
        plt.show(block=block)

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

    def test_kmf_with_interval_censoring_plotting(self, block):
        kmf = KaplanMeierFitter()
        left, right = load_diabetes()["left"], load_diabetes()["right"]
        kmf.fit_interval_censoring(left, right)
        kmf.plot_survival_function(color="r")
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
            df["T"],
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

    def test_breslow_fleming_harrington_plotting(self, block):
        T = 50 * np.random.exponential(1, size=(200, 1)) ** 2
        bf = BreslowFlemingHarringtonFitter().fit(T)
        bf.plot()
        self.plt.title("test_breslow_fleming_harrington_plotting")
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

    def test_parametric_plotting_with_show_censors(self, block):
        n = 200
        T = (np.sqrt(50) * np.random.exponential(1, size=n)) ** 2
        E = T < 100
        T = np.minimum(T, 100)

        wf = WeibullFitter().fit(T, E)
        wf.plot_density(show_censors=True)
        wf.plot_cumulative_density(show_censors=True)

        self.plt.title("test_parametric_plotting_with_show_censors:cumulative_density")
        self.plt.show(block=block)

        wf.plot_survival_function(show_censors=True)
        self.plt.title("test_parametric_plotting_with_show_censors:survival_function")
        self.plt.show(block=block)

        wf.plot_cumulative_hazard(show_censors=True)
        self.plt.title("test_parametric_plotting_with_show_censors:cumulative_hazard")
        self.plt.show(block=block)

        wf.plot_density(show_censors=True)
        self.plt.title("test_parametric_plotting_with_show_censors:density")
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

    def test_flat_style_with_custom_censor_styles(self, block, kmf):
        data1 = np.random.exponential(10, size=200)
        E = np.random.rand(200) < 0.8
        kmf.fit(data1, E, label="test label 1")
        kmf.plot_survival_function(ci_force_lines=True, show_censors=True, censor_styles={"marker": "|", "mew": 1, "ms": 10})
        self.plt.title("test_flat_style_no_censor")
        self.plt.show(block=block)
        return

    def test_loglogs_plot(self, block, kmf):
        data1 = np.random.exponential(10, size=200)
        data2 = np.random.exponential(5, size=200)
        kmf.fit(data1, label="test label 1")
        ax = kmf.plot_survival_function_loglogs()

        kmf.fit(data2, label="test label 2")
        ax = kmf.plot_survival_function_loglogs(ax=ax)

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
        kmf.plot_survival_function()

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

    def test_coxph_plotting_with_hazards_ratios(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot(hazard_ratios=True)
        self.plt.title("test_coxph_plotting")
        self.plt.show(block=block)

    def test_coxph_plotting_with_subset_of_columns(self, block):
        df = load_regression_dataset()
        cp = CoxPHFitter()
        cp.fit(df, "T", "E")
        cp.plot(columns=["var1", "var2"])
        self.plt.title("test_coxph_plotting_with_subset_of_columns")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest")
        cp.plot_partial_effects_on_outcome("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_partial_effects_on_outcome")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_cumulative_hazard(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest")
        cp.plot_partial_effects_on_outcome("age", [10, 50, 80], y="cumulative_hazard")
        self.plt.title("test_coxph_plot_partial_effects_on_outcome")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_strata(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata=["wexp"])
        cp.plot_partial_effects_on_outcome("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_partial_effects_on_outcome_with_strata")
        self.plt.show(block=block)

    def test_aft_plot_partial_effects_on_outcome_with_categorical(self, block):
        df = load_rossi()
        df["cat"] = np.random.choice(["a", "b", "c"], size=df.shape[0])
        aft = WeibullAFTFitter()
        aft.fit(df, "week", "arrest", formula="cat + age + fin")
        aft.plot_partial_effects_on_outcome("cat", values=["a", "b", "c"])
        self.plt.title("test_aft_plot_partial_effects_on_outcome_with_categorical")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_strata_and_complicated_dtypes(self, block):
        # from https://github.com/CamDavidsonPilon/lifelines/blob/master/examples/Customer%20Churn.ipynb
        churn_data = pd.read_csv(
            "https://raw.githubusercontent.com/"
            "treselle-systems/customer_churn_analysis/"
            "master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
        churn_data = churn_data.set_index("customerID")
        churn_data = churn_data.drop(["TotalCharges"], axis=1)

        churn_data = churn_data.applymap(lambda x: "No" if str(x).startswith("No ") else x)
        churn_data["Churn"] = churn_data["Churn"] == "Yes"
        strata_cols = ["InternetService"]

        cph = CoxPHFitter().fit(
            churn_data,
            "tenure",
            "Churn",
            formula="gender + SeniorCitizen + Partner + Dependents  + MultipleLines + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + Contract + PaperlessBilling + PaymentMethod + MonthlyCharges",
            strata=strata_cols,
        )
        cph.plot_partial_effects_on_outcome("Contract", values=["Month-to-month", "One year", "Two year"], plot_baseline=False)
        self.plt.title("test_coxph_plot_partial_effects_on_outcome_with_strata_and_complicated_dtypes")
        self.plt.show(block=block)

    def test_spline_coxph_plot_partial_effects_on_outcome_with_strata(self, block):
        df = load_rossi()
        cp = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=2)
        cp.fit(df, "week", "arrest", strata=["wexp"])
        cp.plot_partial_effects_on_outcome("age", [10, 50, 80])
        self.plt.title("test_spline_coxph_plot_partial_effects_on_outcome_with_strata")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_single_strata(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata="paro")
        cp.plot_partial_effects_on_outcome("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_partial_effects_on_outcome_with_strata")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_nonnumeric_strata(self, block):
        df = load_rossi()
        df["strata"] = np.random.choice(["A", "B"], size=df.shape[0])
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata="strata")
        cp.plot_partial_effects_on_outcome("age", [10, 50, 80])
        self.plt.title("test_coxph_plot_partial_effects_on_outcome_with_single_strata")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_multiple_variables(self, block):
        df = load_rossi()
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest")
        cp.plot_partial_effects_on_outcome(["age", "prio"], [[10, 0], [50, 10], [80, 90]])
        self.plt.title("test_coxph_plot_partial_effects_on_outcome_with_multiple_variables")
        self.plt.show(block=block)

    def test_coxph_plot_partial_effects_on_outcome_with_multiple_variables_and_strata(self, block):
        df = load_rossi()
        df["strata"] = np.random.choice(["A", "B"], size=df.shape[0])
        cp = CoxPHFitter()
        cp.fit(df, "week", "arrest", strata="strata")
        cp.plot_partial_effects_on_outcome(["age", "prio"], [[10, 0], [50, 10], [80, 90]])
        self.plt.title("test_coxph_plot_partial_effects_on_outcome_with_multiple_variables_and_strata")
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
        ax = kmf.plot_survival_function()

        kmf.fit_left_censoring(basin_trough["T"], basin_trough["E"], label="basin_trough")
        ax = kmf.plot_survival_function(ax=ax)
        self.plt.title("test_kmf_left_censorship_plots")
        self.plt.show(block=block)
        return

    def test_aalen_additive_fit_no_censor(self, block):
        n = 2500
        d = 6
        timeline = np.linspace(0, 70, 10000)
        hz, coef, X = generate_hazard_rates(n, d, timeline)
        X.columns = coef.columns
        cumulative_hazards = pd.DataFrame(cumulative_integral(coef.values, timeline), index=timeline, columns=coef.columns)
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
        cumulative_hazards = pd.DataFrame(cumulative_integral(coef.values, timeline), index=timeline, columns=coef.columns)
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

    def test_weibull_aft_plot_partial_effects_on_outcome(self, block):
        df = load_rossi()
        aft = WeibullAFTFitter()
        aft.fit(df, "week", "arrest")
        aft.plot_partial_effects_on_outcome("age", [10, 50, 80])
        self.plt.tight_layout()
        self.plt.title("test_weibull_aft_plot_partial_effects_on_outcome")
        self.plt.show(block=block)

    def test_weibull_aft_plot_partial_effects_on_outcome_with_multiple_columns(self, block):
        df = load_rossi()
        aft = WeibullAFTFitter()
        aft.fit(df, "week", "arrest")
        aft.plot_partial_effects_on_outcome(["age", "prio"], [[10, 0], [50, 10], [80, 50]])
        self.plt.tight_layout()
        self.plt.title("test_weibull_aft_plot_partial_effects_on_outcome_with_multiple_columns")
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

    def test_qq_plot_with_weights_and_entry(self, block):
        from lifelines.utils import survival_events_from_table

        df = pd.DataFrame(index=[60, 171, 263, 427, 505, 639])
        df["death"] = [1, 1, 1, 0, 1, 0]
        df["censored"] = [0, 0, 0, 3, 0, 330]
        T, E, W = survival_events_from_table(df, observed_deaths_col="death", censored_col="censored")
        wf = WeibullFitter().fit(T, E, weights=W, entry=0.0001 * np.ones_like(T))
        ax = qq_plot(wf)
        self.plt.suptitle("test_qq_plot_with_weights_and_entry")
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

    def test_rmst_plot_with_single_model(self, block):
        waltons = load_waltons()
        kmf = KaplanMeierFitter().fit(waltons["T"], waltons["E"])

        rmst_plot(kmf, t=40.0)
        self.plt.title("test_rmst_plot_with_single_model")
        self.plt.show(block=block)

    def test_rmst_plot_with_two_model(self, block):
        waltons = load_waltons()
        ix = waltons["group"] == "control"
        kmf_con = KaplanMeierFitter().fit(waltons.loc[ix]["T"], waltons.loc[ix]["E"], label="control")
        kmf_exp = KaplanMeierFitter().fit(waltons.loc[~ix]["T"], waltons.loc[~ix]["E"], label="exp")

        rmst_plot(kmf_con, model2=kmf_exp, t=40.0)
        self.plt.title("test_rmst_plot_with_two_model")
        self.plt.show(block=block)

    def test_hide_ci_from_legend(self, block):
        waltons = load_waltons()
        kmf = KaplanMeierFitter().fit(waltons["T"], waltons["E"])
        ax = kmf.plot_survival_function(ci_show=True, ci_only_lines=True, ci_legend=False)
        ax.legend(title="Legend title")
        self.plt.title("test_hide_ci_from_legend")
        self.plt.show(block=block)

    def test_logx_plotting(self, block):
        waltons = load_waltons()
        kmf = KaplanMeierFitter().fit(np.exp(waltons["T"]), waltons["E"], timeline=np.logspace(0, 40))
        ax = kmf.plot_survival_function(logx=True)

        wf = WeibullFitter().fit(np.exp(waltons["T"]), waltons["E"], timeline=np.logspace(0, 40))
        wf.plot_survival_function(logx=True, ax=ax)

        self.plt.title("test_logx_plotting")
        self.plt.show(block=block)

    def test_survival_probability_calibration(self, block):
        rossi = load_rossi()
        cph = CoxPHFitter().fit(rossi, "week", "arrest")
        survival_probability_calibration(cph, rossi, 25)
        self.plt.title("test_survival_probability_calibration")
        self.plt.show(block=block)

    def test_survival_probability_calibration_on_out_of_sample_data(self, block):
        rossi = load_rossi()
        rossi = rossi.sample(frac=1.0)
        cph = CoxPHFitter().fit(rossi.loc[:300], "week", "arrest")
        survival_probability_calibration(cph, rossi.loc[300:], 25)
        self.plt.title("test_survival_probability_calibration_on_out_of_sample_data")
        self.plt.show(block=block)

    def test_at_risk_looks_right_when_scales_are_magnitudes_of_order_larger(self, block):

        T1 = list(map(lambda v: v.right, pd.cut(np.arange(32000), 100, retbins=False)))
        T2 = list(map(lambda v: v.right, pd.cut(np.arange(9000), 100, retbins=False)))
        T3 = list(map(lambda v: v.right, pd.cut(np.arange(900), 100, retbins=False)))
        T4 = list(map(lambda v: v.right, pd.cut(np.arange(90), 100, retbins=False)))
        T5 = list(map(lambda v: v.right, pd.cut(np.arange(9), 100, retbins=False)))

        kmf1 = KaplanMeierFitter().fit(T1, label="Category A")
        kmf2 = KaplanMeierFitter().fit(T2, label="Category")
        kmf3 = KaplanMeierFitter().fit(T3, label="CatB")
        kmf4 = KaplanMeierFitter().fit(T4, label="Categ")
        kmf5 = KaplanMeierFitter().fit(T5, label="Categowdary B")

        ax = kmf1.plot()
        ax = kmf2.plot(ax=ax)
        ax = kmf3.plot(ax=ax)
        ax = kmf4.plot(ax=ax)
        ax = kmf5.plot(ax=ax)

        add_at_risk_counts(kmf1, kmf2, kmf3, kmf5, ax=ax)

        self.plt.title("test_at_risk_looks_right_when_scales_are_magnitudes_of_order_larger")
        self.plt.tight_layout()
        self.plt.show(block=block)

    def test_at_risk_looks_right_when_scales_are_magnitudes_of_order_larger_single_attribute(self, block):

        T1 = list(map(lambda v: v.right, pd.cut(np.arange(32000), 100, retbins=False)))
        T2 = list(map(lambda v: v.right, pd.cut(np.arange(9000), 100, retbins=False)))
        T3 = list(map(lambda v: v.right, pd.cut(np.arange(900), 100, retbins=False)))
        T4 = list(map(lambda v: v.right, pd.cut(np.arange(90), 100, retbins=False)))
        T5 = list(map(lambda v: v.right, pd.cut(np.arange(9), 100, retbins=False)))

        kmf1 = KaplanMeierFitter().fit(T1, label="Category A")
        kmf2 = KaplanMeierFitter().fit(T2, label="Category")
        kmf3 = KaplanMeierFitter().fit(T3, label="CatB")
        kmf4 = KaplanMeierFitter().fit(T4, label="Categ")
        kmf5 = KaplanMeierFitter().fit(T5, label="Categowdary B")

        ax = kmf1.plot()
        ax = kmf2.plot(ax=ax)
        ax = kmf3.plot(ax=ax)
        ax = kmf4.plot(ax=ax)
        ax = kmf5.plot(ax=ax)

        add_at_risk_counts(kmf1, kmf2, kmf3, kmf4, kmf5, ax=ax, rows_to_show=["At risk"])

        self.plt.title("test_at_risk_looks_right_when_scales_are_magnitudes_of_order_larger")
        self.plt.tight_layout()
        self.plt.show(block=block)

    def test_at_risk_looks_with_start_of_period_counts(self, block):

        T = load_waltons()["T"]
        E = load_waltons()["E"]
        kmf = KaplanMeierFitter().fit(T, E)

        ax = kmf.plot_survival_function()
        add_at_risk_counts(kmf, ax=ax, at_risk_count_from_start_of_period=True)
        self.plt.title("test_at_risk_looks_with_start_of_period_counts")
        self.plt.tight_layout()
        self.plt.show(block=block)

        ax = kmf.plot_survival_function()
        add_at_risk_counts(kmf, ax=ax, at_risk_count_from_start_of_period=False)
        self.plt.title("test_at_risk_looks_with_start_of_period_counts")
        self.plt.tight_layout()
        self.plt.show(block=block)

    def test_at_risk_with_late_entry(self, block):

        T = load_multicenter_aids_cohort_study()["T"]
        E = load_multicenter_aids_cohort_study()["D"]
        L = load_multicenter_aids_cohort_study()["W"]
        kmf = KaplanMeierFitter().fit(T, E, entry=L)

        ax = kmf.plot_survival_function()
        add_at_risk_counts(kmf, ax=ax, at_risk_count_from_start_of_period=True)
        print(kmf.event_table.head(50))
        self.plt.title("test_at_risk_with_late_entry")
        self.plt.tight_layout()
        self.plt.show(block=block)
