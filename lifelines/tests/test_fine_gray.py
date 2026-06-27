# -*- coding: utf-8 -*-
"""
Tests for FineGrayFitter (Fine & Gray 1999, proportional subdistribution hazard).
"""
import warnings
import numpy as np
import pandas as pd
import pytest

from lifelines import FineGrayFitter, AalenJohansenFitter


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _simulate_competing(n=500, beta=0.8, seed=42):
    """
    Simulate simple competing-risk data.

    Cause-specific hazards:
      - cause 1: h_1(t|x) = lambda_1 * exp(beta * x)  (event of interest)
      - cause 2: h_2(t) = lambda_2                     (competing event)
      - censoring: C ~ Exp(lambda_c)

    Returns a DataFrame with columns T, E (0=censored,1=event,2=competing), x.
    """
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, size=n).astype(float)
    lambda1 = 0.3
    lambda2 = 0.15
    lambda_c = 0.10

    T1 = rng.exponential(1.0 / (lambda1 * np.exp(beta * x)))
    T2 = rng.exponential(1.0 / lambda2, size=n)
    C = rng.exponential(1.0 / lambda_c, size=n)

    T_obs = np.minimum(np.minimum(T1, T2), C)
    E = np.where(T1 < T2, 1, 2)
    E = np.where(np.minimum(T1, T2) < C, E, 0)

    return pd.DataFrame({"T": T_obs, "E": E, "x": x})


def _minimal_df():
    """Tiny hand-crafted DataFrame with clear structure."""
    return pd.DataFrame(
        {
            "T":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "E":  [1, 2, 1, 0, 1, 2, 0, 1, 2,  0,  1,  2],
            "x1": [0, 1, 0, 1, 0, 1, 0, 1, 0,  1,  0,  1],
        }
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestFineGrayInstantiation:

    def test_default_alpha(self):
        fgf = FineGrayFitter()
        assert fgf.alpha == 0.05

    def test_custom_alpha(self):
        fgf = FineGrayFitter(alpha=0.10)
        assert fgf.alpha == 0.10

    def test_repr_before_fit(self):
        assert "FineGrayFitter" in repr(FineGrayFitter())

    def test_repr_after_fit(self):
        df = _minimal_df()
        fgf = FineGrayFitter().fit(df, "T", "E", event_of_interest=1)
        r = repr(fgf)
        assert "fitted with" in r
        assert "12" in r  # n observations


# ---------------------------------------------------------------------------
# Fit validation errors
# ---------------------------------------------------------------------------

class TestFineGrayFitValidation:

    def test_missing_duration_col_raises(self):
        df = _minimal_df()
        with pytest.raises(ValueError, match="not found"):
            FineGrayFitter().fit(df, "BAD_COL", "E", event_of_interest=1)

    def test_missing_event_col_raises(self):
        df = _minimal_df()
        with pytest.raises(ValueError, match="not found"):
            FineGrayFitter().fit(df, "T", "BAD_COL", event_of_interest=1)

    def test_event_of_interest_not_in_data_raises(self):
        df = _minimal_df()
        with pytest.raises(ValueError, match="event_of_interest"):
            FineGrayFitter().fit(df, "T", "E", event_of_interest=99)

    def test_negative_durations_raises(self):
        df = _minimal_df().copy()
        df.loc[0, "T"] = -1
        with pytest.raises(ValueError, match="0"):
            FineGrayFitter().fit(df, "T", "E", event_of_interest=1)

    def test_non_numeric_covariate_raises(self):
        df = _minimal_df().copy()
        df["category"] = ["a", "b"] * 6
        with pytest.raises(TypeError, match="numeric"):
            FineGrayFitter().fit(df, "T", "E", event_of_interest=1)

    def test_predict_before_fit_raises(self):
        df = _minimal_df()
        fgf = FineGrayFitter()
        with pytest.raises(RuntimeError, match="fit"):
            fgf.predict_cumulative_incidence(df)


# ---------------------------------------------------------------------------
# Basic attribute shapes
# ---------------------------------------------------------------------------

class TestFineGrayAttributes:

    @pytest.fixture(autouse=True)
    def fit_model(self):
        self.df = _minimal_df()
        self.fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fgf.fit(self.df, "T", "E", event_of_interest=1)

    def test_params_shape(self):
        assert self.fgf.params_.shape == (1,)  # one covariate x1

    def test_params_index(self):
        assert list(self.fgf.params_.index) == ["x1"]

    def test_variance_matrix_shape(self):
        vm = self.fgf.variance_matrix_
        assert vm.shape == (1, 1)

    def test_standard_errors_shape(self):
        assert self.fgf.standard_errors_.shape == (1,)

    def test_confidence_intervals_shape(self):
        ci = self.fgf.confidence_intervals_
        assert ci.shape == (1, 2)

    def test_log_likelihood_is_finite(self):
        assert np.isfinite(self.fgf.log_likelihood_)

    def test_aic_greater_than_log_lik(self):
        assert self.fgf.AIC_partial_ > self.fgf.log_likelihood_

    def test_bic_is_finite(self):
        assert np.isfinite(self.fgf.BIC_partial_)

    def test_baseline_cumh_is_non_decreasing(self):
        cumh = self.fgf.baseline_cumulative_subdistribution_hazard_.values
        assert np.all(np.diff(cumh) >= -1e-12)

    def test_baseline_cif_bounds(self):
        cif = self.fgf.baseline_cumulative_incidence_.values
        assert np.all(cif >= 0.0)
        assert np.all(cif <= 1.0)

    def test_event_of_interest_stored(self):
        assert self.fgf.event_of_interest == 1


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestFineGrayPrediction:

    @pytest.fixture(autouse=True)
    def fit_model(self):
        self.df = _minimal_df()
        self.fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fgf.fit(self.df, "T", "E", event_of_interest=1)

    def test_predict_cif_returns_dataframe(self):
        result = self.fgf.predict_cumulative_incidence(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_predict_cif_shape_with_times(self):
        times = [2.0, 5.0, 8.0]
        result = self.fgf.predict_cumulative_incidence(self.df, times=times)
        assert result.shape[0] == len(times)
        assert result.shape[1] == len(self.df)

    def test_predict_cif_default_times(self):
        result = self.fgf.predict_cumulative_incidence(self.df)
        n_event_times = int((self.df["E"] == 1).sum())  # unique event times of type 1
        assert result.shape[0] == n_event_times

    def test_predict_cif_values_in_unit_interval(self):
        result = self.fgf.predict_cumulative_incidence(self.df, times=[1, 5, 10])
        assert np.all(result.values >= 0.0)
        assert np.all(result.values <= 1.0)

    def test_predict_cif_non_decreasing_over_time(self):
        times = np.linspace(1, 12, 30)
        result = self.fgf.predict_cumulative_incidence(self.df.iloc[:1], times=times)
        vals = result.values[:, 0]
        assert np.all(np.diff(vals) >= -1e-10)

    def test_predict_log_partial_hazard_shape(self):
        lph = self.fgf.predict_log_partial_hazard(self.df)
        assert lph.shape == (len(self.df),)

    def test_predict_partial_hazard_positive(self):
        ph = self.fgf.predict_partial_hazard(self.df)
        assert np.all(ph.values > 0)

    def test_predict_partial_hazard_equals_exp_log_ph(self):
        lph = self.fgf.predict_log_partial_hazard(self.df).values
        ph = self.fgf.predict_partial_hazard(self.df).values
        np.testing.assert_allclose(ph, np.exp(lph))


# ---------------------------------------------------------------------------
# Statistical correctness on simulated data
# ---------------------------------------------------------------------------

class TestFineGrayStatistical:

    def test_positive_covariate_raises_cif(self):
        """Positive beta should increase the CIF for high-x subjects."""
        df = _simulate_competing(n=600, beta=1.2, seed=1)
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        assert fgf.params_["x"] > 0

    def test_negative_covariate_lowers_cif(self):
        """Negative beta should decrease the CIF for high-x subjects."""
        df = _simulate_competing(n=600, beta=-1.2, seed=2)
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        assert fgf.params_["x"] < 0

    def test_coefficient_sign_is_correct(self):
        """Fitted beta should have same sign as true beta across seeds."""
        n_correct = 0
        for seed in range(10):
            df = _simulate_competing(n=400, beta=0.9, seed=seed)
            fgf = FineGrayFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fgf.fit(df, "T", "E", event_of_interest=1)
            n_correct += int(fgf.params_["x"] > 0)
        assert n_correct >= 8, "Expected beta > 0 in most trials"

    def test_zero_covariate_gives_flat_predictor(self):
        """A covariate always 0 → linear predictor = 0 for all subjects."""
        df = _simulate_competing(n=200, beta=0.9, seed=3)
        df["x_zero"] = 0.0
        df = df.drop(columns=["x"])
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        ph = fgf.predict_partial_hazard(df).values
        np.testing.assert_allclose(ph, np.ones_like(ph), atol=1e-6)

    def test_high_cif_for_high_risk_group(self):
        """CIF at t=median should be higher for x=1 than x=0 when beta > 0."""
        df = _simulate_competing(n=800, beta=1.5, seed=5)
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)

        t_mid = float(np.median(df.loc[df["E"] == 1, "T"]))
        X_hi = pd.DataFrame({"x": [1.0]})
        X_lo = pd.DataFrame({"x": [0.0]})
        cif_hi = float(fgf.predict_cumulative_incidence(X_hi, times=[t_mid]).values[0, 0])
        cif_lo = float(fgf.predict_cumulative_incidence(X_lo, times=[t_mid]).values[0, 0])
        assert cif_hi > cif_lo

    def test_confidence_intervals_cover_true_beta(self):
        """95% CI should contain the true beta approximately 95% of the time."""
        true_beta = 0.8
        n_covered = 0
        n_trials = 20
        for seed in range(n_trials):
            df = _simulate_competing(n=300, beta=true_beta, seed=seed + 100)
            fgf = FineGrayFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fgf.fit(df, "T", "E", event_of_interest=1)
            lo = float(fgf.confidence_intervals_.iloc[0, 0])
            hi = float(fgf.confidence_intervals_.iloc[0, 1])
            n_covered += int(lo <= true_beta <= hi)
        # Expect ~95% coverage; allow wide tolerance for small n
        assert n_covered >= 14, "Coverage too low: %d/%d" % (n_covered, n_trials)

    def test_two_covariates(self):
        """Model should handle two covariates without error."""
        rng = np.random.default_rng(77)
        n = 300
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        T1 = rng.exponential(1.0 / (0.3 * np.exp(0.5 * x1 - 0.3 * x2)))
        T2 = rng.exponential(1.0 / 0.2, size=n)
        C = rng.exponential(1.0 / 0.1, size=n)
        T_obs = np.minimum(np.minimum(T1, T2), C)
        E = np.where(T1 < T2, 1, 2)
        E = np.where(np.minimum(T1, T2) < C, E, 0)
        df = pd.DataFrame({"T": T_obs, "E": E, "x1": x1, "x2": x2})
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        assert fgf.params_.shape == (2,)
        assert fgf.params_["x1"] > 0  # should detect positive effect

    def test_no_covariates_baseline_matches_aalen_johansen(self):
        """
        Without covariates, the Breslow-estimated baseline CIF should be close
        to the non-parametric Aalen-Johansen estimate.
        """
        rng = np.random.default_rng(9)
        n = 600
        T = rng.exponential(1.0, size=n)
        E = rng.choice([0, 1, 2], size=n, p=[0.15, 0.55, 0.30])
        df = pd.DataFrame({"T": T, "E": E, "x_const": np.zeros(n)})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf = FineGrayFitter().fit(df, "T", "E", event_of_interest=1)

        ajf = AalenJohansenFitter()
        ajf.fit(T, E, event_of_interest=1)

        # Evaluate Aalen-Johansen at Fine-Gray event times
        fg_times = fgf.baseline_cumulative_incidence_.index.values
        aj_cif = ajf.cumulative_density_.reindex(fg_times, method="ffill").fillna(0).values.ravel()
        fg_cif = fgf.baseline_cumulative_incidence_.values

        # Should be within 10 percentage points everywhere
        max_diff = np.max(np.abs(fg_cif - aj_cif))
        assert max_diff < 0.10, "Max deviation from AJ: %.4f" % max_diff


# ---------------------------------------------------------------------------
# Weights column
# ---------------------------------------------------------------------------

class TestFineGrayWeights:

    def test_weights_col_accepted(self):
        df = _minimal_df().copy()
        df["w"] = 1.0
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1, weights_col="w")
        assert hasattr(fgf, "params_")

    def test_equal_weights_same_as_no_weights(self):
        df = _minimal_df().copy()
        df["w"] = 1.0
        fgf1 = FineGrayFitter()
        fgf2 = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf1.fit(df, "T", "E", event_of_interest=1)
            fgf2.fit(df, "T", "E", event_of_interest=1, weights_col="w")
        np.testing.assert_allclose(fgf1.params_.values, fgf2.params_.values, atol=1e-8)


# ---------------------------------------------------------------------------
# Multi-cause competing events
# ---------------------------------------------------------------------------

class TestFineGrayMultiCause:

    def test_three_cause_event(self):
        """Model should work with three distinct event codes."""
        rng = np.random.default_rng(42)
        n = 300
        T = rng.exponential(1.0, size=n)
        E = rng.choice([0, 1, 2, 3], size=n, p=[0.10, 0.40, 0.30, 0.20])
        x = rng.normal(size=n)
        df = pd.DataFrame({"T": T, "E": E, "x": x})
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        assert fgf.params_.shape == (1,)
        # Competing events are 2 and 3 — both should be in the IPCW risk set
        cif = fgf.predict_cumulative_incidence(df, times=[0.5, 1.0, 2.0])
        assert np.all(cif.values >= 0)


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

class TestFineGraySummary:

    def test_print_summary_runs(self, capsys):
        df = _minimal_df()
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        fgf.print_summary()
        captured = capsys.readouterr()
        assert "FineGrayFitter" in captured.out

    def test_print_summary_contains_coef(self, capsys):
        df = _minimal_df()
        fgf = FineGrayFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgf.fit(df, "T", "E", event_of_interest=1)
        fgf.print_summary()
        captured = capsys.readouterr()
        assert "coef" in captured.out


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

class TestFineGrayImport:

    def test_importable_from_lifelines(self):
        from lifelines import FineGrayFitter as FGF
        assert FGF is FineGrayFitter

    def test_in_all(self):
        import lifelines
        assert "FineGrayFitter" in lifelines.__all__
