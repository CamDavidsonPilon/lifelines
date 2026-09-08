import numpy as np
import pandas as pd
import pytest

from lifelines import CoxPHFitter
from lifelines.datasets import load_lung
from lifelines.statistics import TimeTransformers, proportional_hazard_test


@pytest.mark.parametrize("as_series", [False, True])
def test_rank_transform_uses_event_times_and_averages_ties(as_series):
    times = np.array([4.0, 1.0, 2.0, 2.0, 3.0])
    events = np.array([1, 1, 1, 1, 0], dtype=bool)
    if as_series:
        times = pd.Series(times, index=["e", "a", "c", "b", "d"])
    result = TimeTransformers().get("rank")(times, events, np.ones(5))
    np.testing.assert_allclose(np.asarray(result)[events], [4.0, 1.0, 2.5, 2.5])


def test_rank_transform_preserves_ordered_untied_event_ranks():
    times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    events = pd.Series([True, False, True, False, True])
    result = TimeTransformers().get("rank")(times, events, np.ones(5))
    np.testing.assert_array_equal(result[events], [1.0, 2.0, 3.0])


def test_ph_rank_is_invariant_to_order_within_ties():
    x = np.linspace(-2.0, 2.0, 400)
    frame = pd.DataFrame({"T": np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], len(x)), "E": 1, "x": np.tile(x, 5)})
    results = []
    for data in [frame, frame.sample(frac=1, random_state=19)]:
        model = CoxPHFitter().fit(data, "T", "E")
        assert model.params_["x"] == pytest.approx(0, abs=1e-12)
        result = proportional_hazard_test(model, data)
        results.append(result.summary.loc["x", "p"])
    np.testing.assert_allclose(results, [1.0, 1.0], atol=1e-12)


def test_ph_rank_is_invariant_to_stratum_labels():
    frame = load_lung()[["time", "status", "age", "sex", "ph.ecog"]].dropna()
    results = []
    coefficients = []
    for reverse in [False, True]:
        data = frame.copy()
        levels = data.pop("ph.ecog")
        data["stratum"] = (3 - levels if reverse else levels).astype(str)
        model = CoxPHFitter().fit(data, "time", "status", strata="stratum", formula="age + sex")
        coefficients.append(model.params_.values)
        results.append(proportional_hazard_test(model, data).summary["p"].values)
    np.testing.assert_allclose(coefficients[0], coefficients[1], atol=1e-12)
    np.testing.assert_allclose(results[0], results[1], atol=1e-12)
