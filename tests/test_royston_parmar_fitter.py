# -*- coding: utf-8 -*-
"""
Tests for lifelines.fitters.royston_parmar_fitter.FlexibleParametricPHFitter.
"""

import numpy as np
import pandas as pd
import pytest

from lifelines import FlexibleParametricPHFitter
from lifelines.datasets import load_rossi
from lifelines.utils import concordance_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rossi():
    return load_rossi()


@pytest.fixture(scope="module")
def fitter_baseline(rossi):
    """Baseline-only (no covariates) fitted model."""
    m = FlexibleParametricPHFitter(n_baseline_knots=4)
    m.fit(rossi, "week", "arrest")
    return m


@pytest.fixture(scope="module")
def fitter_with_covariates(rossi):
    """Model with covariates."""
    m = FlexibleParametricPHFitter(n_baseline_knots=4)
    m.fit(rossi, "week", "arrest", covariates=["fin", "age", "prio"])
    return m


@pytest.fixture(scope="module")
def newdata(rossi):
    """A small prediction dataset."""
    return rossi[["fin", "age", "prio"]].head(10).reset_index(drop=True)


@pytest.fixture(scope="module")
def newdata_no_cov(rossi):
    """Dummy prediction dataset for baseline model."""
    return rossi.head(10).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Basic fit tests
# ---------------------------------------------------------------------------

def test_fit_returns_self(rossi):
    m = FlexibleParametricPHFitter(n_baseline_knots=4)
    result = m.fit(rossi, "week", "arrest")
    assert result is m


def test_fit_no_covariates(rossi):
    m = FlexibleParametricPHFitter(n_baseline_knots=4)
    m.fit(rossi, "week", "arrest")
    assert len(m.covariates) == 0
    # baseline model: params are gamma0_ ... gamma_{K-2}_  (n_knots - 1 params)
    assert len(m.params_) == m.n_baseline_knots - 1


def test_fit_with_covariates(rossi):
    m = FlexibleParametricPHFitter(n_baseline_knots=4)
    m.fit(rossi, "week", "arrest", covariates=["fin", "age", "prio"])
    assert "fin" in m.params_.index
    assert "age" in m.params_.index
    assert "prio" in m.params_.index
    # 3 spline params + 3 covariate betas
    assert len(m.params_) == (m.n_baseline_knots - 1) + 3


def test_log_likelihood_finite_after_fit(fitter_baseline, fitter_with_covariates):
    assert np.isfinite(fitter_baseline.log_likelihood_)
    assert np.isfinite(fitter_with_covariates.log_likelihood_)


def test_aic_computed(fitter_baseline, fitter_with_covariates):
    assert np.isfinite(fitter_baseline.AIC_)
    assert np.isfinite(fitter_with_covariates.AIC_)
    # AIC = 2k - 2 log L, must be positive for reasonable LL
    n_params_baseline = fitter_baseline.n_baseline_knots - 1
    expected_aic = 2 * n_params_baseline - 2 * fitter_baseline.log_likelihood_
    assert abs(fitter_baseline.AIC_ - expected_aic) < 1e-6


def test_knot_count_respected(rossi):
    for k in [2, 3, 5, 6]:
        m = FlexibleParametricPHFitter(n_baseline_knots=k)
        m.fit(rossi, "week", "arrest")
        assert len(m.knots_) == k, f"Expected {k} knots, got {len(m.knots_)}"
        assert len(m.params_) == k - 1


def test_custom_knot_locations(rossi):
    # Provide knots in original time scale
    custom_knots = [1.0, 10.0, 30.0, 52.0]  # weeks, within [1, 52]
    m = FlexibleParametricPHFitter(knot_locations=custom_knots)
    m.fit(rossi, "week", "arrest")
    # knots_ are in log scale
    np.testing.assert_allclose(m.knots_, np.log(custom_knots), rtol=1e-10)


# ---------------------------------------------------------------------------
# Prediction shape & range tests
# ---------------------------------------------------------------------------

def test_predict_survival_function_shape(fitter_with_covariates, newdata):
    sf = fitter_with_covariates.predict_survival_function(newdata)
    assert isinstance(sf, pd.DataFrame)
    # should have n_subjects columns, default 200 time points rows
    assert sf.shape[1] == len(newdata)
    assert sf.shape[0] == 200


def test_survival_function_between_0_and_1(fitter_with_covariates, newdata):
    sf = fitter_with_covariates.predict_survival_function(newdata)
    assert (sf.values >= 0).all(), "Survival function has negative values"
    assert (sf.values <= 1).all(), "Survival function exceeds 1"


def test_survival_decreasing_in_time(fitter_with_covariates, newdata):
    sf = fitter_with_covariates.predict_survival_function(newdata)
    # S(t) must be non-increasing along rows
    diffs = np.diff(sf.values, axis=0)
    # allow tiny floating point noise, use a loose tolerance
    assert (diffs <= 1e-8).all(), "Survival function is not monotonically non-increasing"


def test_predict_cumulative_hazard_monotone(fitter_with_covariates, newdata):
    ch = fitter_with_covariates.predict_cumulative_hazard(newdata)
    diffs = np.diff(ch.values, axis=0)
    assert (diffs >= -1e-8).all(), "Cumulative hazard is not monotonically non-decreasing"


def test_predict_median_positive(fitter_with_covariates, newdata):
    medians = fitter_with_covariates.predict_median(newdata)
    assert isinstance(medians, pd.Series)
    assert len(medians) == len(newdata)
    # finite medians should be > 0
    finite_mask = np.isfinite(medians.values)
    assert finite_mask.any(), "All medians are infinite — likely a convergence issue"
    assert (medians.values[finite_mask] > 0).all()


def test_predict_with_explicit_times(fitter_with_covariates, newdata):
    times = np.array([1.0, 5.0, 10.0, 20.0, 40.0])
    sf = fitter_with_covariates.predict_survival_function(newdata, times=times)
    assert sf.shape[0] == len(times)
    assert sf.shape[1] == len(newdata)


# ---------------------------------------------------------------------------
# Concordance index (discrimination)
# ---------------------------------------------------------------------------

def test_concordance_index_above_chance(rossi):
    """
    On a well-separated synthetic dataset the C-index should be clearly above 0.5.
    """
    rng = np.random.default_rng(42)
    n = 400
    # Strong single covariate X ~ N(0,1); true hazard rate = exp(2*X)
    X_col = rng.standard_normal(n)
    true_scale = np.exp(-2 * X_col)        # Exponential scale = 1/rate
    T_true = rng.exponential(scale=true_scale)
    C = rng.exponential(scale=10.0, size=n)
    T_obs = np.minimum(T_true, C)
    E = (T_true <= C).astype(int)

    df = pd.DataFrame({"T": T_obs + 1e-6, "E": E, "X": X_col})
    m = FlexibleParametricPHFitter(n_baseline_knots=4)
    m.fit(df, "T", "E", covariates=["X"])

    # Use partial hazard exp(beta * X) as risk score
    risk = np.exp(m.params_["X"] * df["X"].values)
    ci = concordance_index(df["T"], -risk, df["E"])
    assert ci > 0.65, f"C-index {ci:.3f} not above 0.65 on strongly separated data"


# ---------------------------------------------------------------------------
# Baseline-only prediction tests
# ---------------------------------------------------------------------------

def test_baseline_survival_is_decreasing(fitter_baseline):
    # Predict over the stored fine grid
    t = fitter_baseline.baseline_cumulative_hazard_.index.values
    H = fitter_baseline.baseline_cumulative_hazard_["baseline cumulative hazard"].values
    diffs = np.diff(H)
    assert (diffs >= -1e-8).all(), "Baseline cumulative hazard is not monotone"


def test_baseline_survival_starts_below_1(fitter_baseline):
    S = fitter_baseline.baseline_survival_["baseline survival"].values
    assert S[0] <= 1.0
    assert S[0] > 0.0


def test_predict_baseline_no_covariate_model(fitter_baseline, newdata_no_cov):
    # For baseline model, predict with empty covariate frame
    X_empty = newdata_no_cov[[]].copy()   # empty columns, same index
    sf = fitter_baseline.predict_survival_function(X_empty)
    # All subjects should have the same survival curve (no covariates)
    assert sf.shape[1] == len(newdata_no_cov)
    # all columns should be equal
    vals = sf.values
    np.testing.assert_allclose(vals[:, 0], vals[:, 1], rtol=1e-10)
