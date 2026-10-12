import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatisticalWarning


def assert_weighted_estimates(km, survival, greenwood):
    # The fixtures use hand-computed weighted risk sets, independent of event_table.
    assert_allclose(km.survival_function_.iloc[:, 0], survival, rtol=1e-13, atol=0)
    assert_allclose(km._cumulative_sq_, greenwood, rtol=1e-13, atol=0)
    expected_ci = []
    z = norm.ppf(0.975)
    for s, g in zip(survival, greenwood):
        if s in (0, 1):
            expected_ci.append([s, s])
        else:
            log_s = np.log(s)
            with np.errstate(over="ignore"):
                expected_ci.append([s ** np.exp(-z * np.sqrt(g) / log_s), s ** np.exp(z * np.sqrt(g) / log_s)])
    assert_allclose(km.confidence_interval_, expected_ci, rtol=1e-12, atol=0)
    assert np.isfinite(km._cumulative_sq_).all()
    assert (km._cumulative_sq_ >= 0).all()
    ci = km.confidence_interval_.to_numpy()
    assert ((0 <= ci) & (ci <= 1)).all()
    assert (ci[:, 0] <= survival).all()
    assert (survival <= ci[:, 1]).all()


@pytest.mark.parametrize(
    "weights, survival, greenwood",
    [
        ([0.1, 0.2, 0.3], [1, 5 / 6, 1 / 2], [0, 1 / 3, 5 / 3]),
        ([0.3, 0.2, 0.1], [1, 1 / 2, 1 / 6], [0, 5 / 3, 25 / 3]),
        ([0.1, 0.1, 0.1], [1, 2 / 3, 1 / 3], [0, 5 / 3, 20 / 3]),
    ],
)
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("terminal_observed", [True, False])
def test_fractional_weights_terminal_event_and_censoring(weights, survival, greenwood, scale, terminal_observed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatisticalWarning)
        km = KaplanMeierFitter().fit([1, 2, 3], [1, 1, terminal_observed], weights=np.array(weights) * scale)
    survival = np.array(survival + [0 if terminal_observed else survival[-1]])
    greenwood = np.array(greenwood + [greenwood[-1]]) / scale
    assert_weighted_estimates(km, survival, greenwood)
    if terminal_observed:
        assert km.survival_function_.iloc[-1, 0] == 0
        assert (km.confidence_interval_.iloc[-1] == 0).all()
    else:
        assert km.survival_function_.iloc[-1, 0] > 0


@pytest.mark.parametrize("weight", [0.4, 1.5, 1.0, 2.0])
def test_weighted_greenwood_original_issue(weight):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatisticalWarning)
        km = KaplanMeierFitter().fit([1, 2, 3, 4, 5, 6], [1, 1, 0, 1, 1, 1], weights=np.full(6, weight))
    survival = np.array([1, 5 / 6, 2 / 3, 2 / 3, 4 / 9, 2 / 9, 0])
    greenwood = np.array([0, 1 / 30, 1 / 12, 1 / 12, 1 / 4, 3 / 4, 3 / 4]) / weight
    assert_weighted_estimates(km, survival, greenwood)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_terminal_roundoff_from_both_sides(scale, direction):
    population = pd.Series([np.nextafter(scale, direction)])
    deaths = pd.Series([scale])
    km = KaplanMeierFitter()
    assert km._additive_f(population, deaths).iloc[0] == -np.inf
    assert km._additive_var(population, deaths).iloc[0] == 0


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("method", ["_additive_f", "_additive_var"])
def test_materially_excess_deaths_are_rejected(scale, method):
    # Even this small relative discrepancy is far larger than floating-point roundoff.
    with pytest.raises(ValueError, match="deaths.*at.risk"):
        getattr(KaplanMeierFitter(), method)(pd.Series([scale]), pd.Series([scale * (1 + 1e-8)]))


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_small_positive_risk_remainder_is_not_exhaustion(scale):
    # Default np.isclose tolerances would incorrectly collapse this real remainder.
    population = pd.Series([scale])
    deaths = pd.Series([scale * (1 - 1e-8)])
    remainder = population - deaths
    km = KaplanMeierFitter()
    increment = km._additive_f(population, deaths)
    assert np.isfinite(increment.iloc[0])
    assert np.exp(increment.iloc[0]) > 0
    assert_allclose(np.exp(increment), remainder / population, rtol=1e-13, atol=0)
    assert_allclose(km._additive_var(population, deaths), deaths / (population * remainder), rtol=1e-13)


def test_terminal_risk_set_avoids_accumulated_roundoff():
    weights = np.full(1000, 0.1)
    with pytest.warns(StatisticalWarning):
        km = KaplanMeierFitter().fit(np.arange(1, 1001), weights=weights)
    assert km.event_table["at_risk"].iloc[-1] == weights[-1]
    assert km.survival_function_.iloc[-1, 0] == 0
    assert (km.confidence_interval_.iloc[-1] == 0).all()
    assert np.isfinite(km._cumulative_sq_).all()
    assert (km._cumulative_sq_ >= 0).all()


def test_small_terminal_censoring_weight_remains_positive():
    with pytest.warns(StatisticalWarning):
        km = KaplanMeierFitter().fit([1, 2, 3], [1, 1, 0], weights=[1, 1, 1e-8])
    assert km.survival_function_.iloc[-1, 0] > 0
    assert_allclose(km.survival_function_.iloc[-1, 0], 1e-8 / (2 + 1e-8), rtol=1e-7, atol=0)
    assert_allclose(km.survival_function_.iloc[-1, 0], km.survival_function_.iloc[-2, 0], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_left_censoring_empty_risk_set_has_no_negative_roundoff(scale):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatisticalWarning)
        km = KaplanMeierFitter().fit_left_censoring([1, 2, 3], weights=np.array([0.1, 0.2, 0.3]) * scale)
    assert_allclose(km.survival_function_.iloc[:, 0], [1, 5 / 6, 1 / 2, 0], rtol=1e-13, atol=0)
    assert_allclose(km._cumulative_sq_, np.array([25 / 3, 25 / 3, 5 / 3, 0]) / scale, rtol=1e-13, atol=0)
