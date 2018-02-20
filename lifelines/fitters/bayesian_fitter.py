# -*- coding: utf-8 -*-
from __future__ import print_function
import warnings
import numpy as np
import pandas as pd
from numpy.random import beta
from numpy import dot, exp
from numpy.linalg import solve, norm, inv
from scipy.integrate import trapz
import scipy.stats as stats

from lifelines.fitters import BaseFitter
from lifelines.utils import survival_table_from_events, inv_normal_cdf, normalize,\
    significance_code, concordance_index, _get_index, qth_survival_times,\
    pass_for_numeric_dtypes_or_raise, check_low_var, _preprocess_inputs, coalesce

class BayesianFitter(BaseFitter):
    """
    If you have small data, and KM feels too uncertain, you can use the BayesianFitter to
    generate sample survival functions. The algorithm is:

    S_i(T) = \Prod_{t=0}^T (1 - p_t)

    where p_t ~ Beta( 0.01 + d_t, 0.01 + n_t - d_t), d_t is the number of deaths and n_t is the size of the
    population at risk at time t. The prior is a Beta(0.01, 0.01) for each time point (high values led to a
    high bias).

    Parameters:
        samples: the number of sample survival functions to return
    """

    def __init__(self, samples=300):
        self.beta = beta
        self.samples = samples

    def fit(self, durations, censorship=None, timeline=None, entry=None):
        """
        Parameters:
          durations: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          censorship: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if censorship==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)

        Returns:
          self, with new properties like 'sample_survival_functions_'.
        """
        v = _preprocess_inputs(durations, censorship, timeline, entry)
        self.durations, self.censorship, self.timeline, self.entry, self.event_table = v
        self.sampled_survival_functions_ = self.generate_sample_path(self.samples)
        return self

    def plot(self, **kwargs):
        kwargs['alpha'] = coalesce(kwargs.pop('alpha', None), 0.05)
        kwargs['legend'] = False
        kwargs['c'] = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), '#348ABD')
        ax = self.sampled_survival_functions_.plot(**kwargs)
        return ax

    def generate_sample_path(self, n=1):
        deaths = self.event_table['observed']
        population = self.event_table['entrance'].cumsum() - self.event_table['removed'].cumsum().shift(1).fillna(0)
        d = deaths.shape[0]
        samples = 1. - beta(0.01 + deaths, 0.01 + population - deaths, size=(n,d))
        sample_paths = pd.DataFrame(np.exp(np.log(samples).cumsum(1)).T, index=self.timeline)
        return sample_paths
