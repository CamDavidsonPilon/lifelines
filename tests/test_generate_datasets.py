# test_generate_datasets.py
from __future__ import print_function
import os

import pytest
import matplotlib.pyplot as plt

from lifelines.estimation import KaplanMeierFitter, NelsonAalenFitter
from lifelines.generate_datasets import exponential_survival_data


def test_exponential_data_sets_correct_censor():
    print(os.environ)
    N = 20000
    censorship = 0.2
    T, C = exponential_survival_data(N, censorship, scale=10)
    assert abs(C.mean() - (1 - censorship)) < 0.02


@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
def test_exponential_data_sets_fit():
    N = 20000
    T, C = exponential_survival_data(N, 0.2, scale=10)
    naf = NelsonAalenFitter()
    naf.fit(T, C).plot()
    plt.title("Should be a linear with slope = 0.1")


@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
def test_kmf_minimum_observation_bias():
    N = 250
    kmf = KaplanMeierFitter()
    T, C = exponential_survival_data(N, 0.1, scale=10)
    B = 0.01 * T
    kmf.fit(T, C, entry=B)
    kmf.plot()
    plt.title("Should have larger variances in the tails")
