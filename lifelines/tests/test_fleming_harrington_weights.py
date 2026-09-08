import numpy as np
import pytest
from scipy.stats import chi2

from lifelines.datasets import load_waltons
from lifelines.statistics import multivariate_logrank_test


def _two_group_reference(table, p, q):
    # Direct risk-set formula, independent of lifelines' survival estimators.
    time = table["T"].to_numpy()
    event = table["E"].to_numpy()
    group = table["group"].to_numpy() == "control"
    frequency = table["weights"].to_numpy()
    survival = 1.0
    observed_minus_expected = 0.0
    variance = 0.0
    for t in np.unique(time):
        at_risk = time >= t
        died = (time == t) & event.astype(bool)
        n = frequency[at_risk].sum()
        n_a = frequency[at_risk & group].sum()
        deaths = frequency[died].sum()
        deaths_a = frequency[died & group].sum()
        weight = survival**p * (1 - survival) ** q
        observed_minus_expected += weight * (deaths_a - deaths * n_a / n)
        if n > 1:
            variance += weight**2 * n_a * (n - n_a) * deaths * (n - deaths) / (n**2 * (n - 1))
        survival *= 1 - deaths / n
    statistic = observed_minus_expected**2 / variance
    return statistic, chi2.sf(statistic, 1)


@pytest.mark.parametrize("p,q", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 3), (4, 4)])
def test_fh_frequency_weights_match_expanded_data_and_risk_set_reference(p, q):
    data = load_waltons()
    grouped = data.groupby(["T", "E", "group"]).size().rename("weights").reset_index()
    expected_stat, expected_p = _two_group_reference(grouped, p, q)
    expanded = multivariate_logrank_test(data["T"], data["group"], data["E"], weightings="fleming-harrington", p=p, q=q)
    compressed = multivariate_logrank_test(
        grouped["T"], grouped["group"], grouped["E"], weights=grouped["weights"], weightings="fleming-harrington", p=p, q=q
    )
    for result in [expanded, compressed]:
        np.testing.assert_allclose(result.test_statistic, expected_stat, rtol=1e-10)
        np.testing.assert_allclose(result.p_value, expected_p, rtol=1e-10, atol=0)


@pytest.mark.parametrize("p,q", [(1, 1), (0, 1)])
def test_fh_is_invariant_to_time_origin(p, q):
    time = np.array([2, 6, 1, 9, 0, 3, 5, 4, 11])
    group = np.array([0] * 5 + [1] * 4)
    at_zero = multivariate_logrank_test(time, group, weightings="fleming-harrington", p=p, q=q)
    shifted = multivariate_logrank_test(time + 1, group, weightings="fleming-harrington", p=p, q=q)
    np.testing.assert_allclose(at_zero.test_statistic, shifted.test_statistic)
    np.testing.assert_allclose(at_zero.p_value, shifted.p_value)
