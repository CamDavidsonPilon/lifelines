# -*- coding: utf-8 -*-
from __future__ import print_function
from itertools import combinations

import numpy as np
from scipy import stats
import pandas as pd

from lifelines.utils import group_survival_table_from_events


def sample_size_necessary_under_cph(power, ratio_of_participants, p_exp, p_con,
                                    postulated_hazard_ratio, alpha=0.05):
    """
    This computes the sample size for needed power to compare two groups under a Cox
    Proportional Hazard model.

    References:
        https://cran.r-project.org/web/packages/powerSurvEpi/powerSurvEpi.pdf

    Parameters:
        power: power to detect the magnitude of the hazard ratio as small as that specified by postulated_hazard_ratio.
        ratio_of_participants: ratio of participants in experimental group over control group.
        p_exp: probability of failure in experimental group over period of study.
        p_con: probability of failure in control group over period of study
        postulated_hazard_ratio: the postulated hazard ratio
        alpha: type I error rate

    Returns:
        n_exp, n_con: the samples sizes need for the experiment and control group, respectively, to achieve desired power
    """
    z = lambda p: stats.norm.ppf(p)

    m = 1.0 / ratio_of_participants \
        * ((ratio_of_participants * postulated_hazard_ratio + 1.0) / (postulated_hazard_ratio - 1.0)) ** 2 \
        * (z(1. - alpha / 2.) + z(power)) ** 2

    n_exp = m * ratio_of_participants / (ratio_of_participants * p_exp + p_con)
    n_con = m / (ratio_of_participants * p_exp + p_con)

    return int(np.ceil(n_exp)), int(np.ceil(n_con))


def power_under_cph(n_exp, n_con, p_exp, p_con, postulated_hazard_ratio, alpha=0.05):
    """
    This computes the sample size for needed power to compare two groups under a Cox
    Proportional Hazard model.

    References:
        https://cran.r-project.org/web/packages/powerSurvEpi/powerSurvEpi.pdf

    Parameters:
        n_exp: size of the experiment group.
        n_con: size of the control group.
        p_exp: probability of failure in experimental group over period of study.
        p_con: probability of failure in control group over period of study
        postulated_hazard_ratio: the postulated hazard ratio
        alpha: type I error rate

    Returns:
        power: power to detect the magnitude of the hazard ratio as small as that specified by postulated_hazard_ratio.
    """
    z = lambda p: stats.norm.ppf(p)

    m = n_exp * p_exp + n_con * p_con
    k = float(n_exp) / float(n_con)
    return stats.norm.cdf(np.sqrt(k * m) * abs(postulated_hazard_ratio - 1) / (k * postulated_hazard_ratio + 1) - z(1 - alpha / 2.))


def logrank_test(event_times_A, event_times_B, event_observed_A=None, event_observed_B=None,
                 alpha=0.95, t_0=-1, **kwargs):
    """
    Measures and reports on whether two intensity processes are different. That is, given two
    event series, determines whether the data generating processes are statistically different.
    The test-statistic is chi-squared under the null hypothesis.

    H_0: both event series are from the same generating processes
    H_A: the event series are from different generating processes.

    See Survival and Event Analysis, page 108. This implicitly uses the log-rank weights.

    Parameters:
      event_times_foo: a (nx1) array of event durations (birth to death,...) for the population.
      censorship_bar: a (nx1) array of censorship flags, 1 if observed, 0 if not. Default assumes all observed.
      t_0: the period under observation, -1 for all time.
      alpha: the level of signifiance
      kwargs: add keywords and meta-data to the experiment summary

    Returns:
      results: a StatisticalResult object with properties 'p_value', 'summary', 'test_statistic', 'test_result'

    """

    event_times_A, event_times_B = np.array(event_times_A), np.array(event_times_B)
    if event_observed_A is None:
        event_observed_A = np.ones(event_times_A.shape[0])
    if event_observed_B is None:
        event_observed_B = np.ones(event_times_B.shape[0])

    event_times = np.r_[event_times_A, event_times_B]
    groups = np.r_[np.zeros(event_times_A.shape[0], dtype=int), np.ones(event_times_B.shape[0], dtype=int)]
    event_observed = np.r_[event_observed_A, event_observed_B]
    return multivariate_logrank_test(event_times, groups, event_observed,
                                     alpha=alpha, t_0=t_0, **kwargs)


def pairwise_logrank_test(event_durations, groups, event_observed=None,
                          alpha=0.95, t_0=-1, bonferroni=True, **kwargs):
    """
    Perform the logrank test pairwise for all n>2 unique groups (use the more appropriate logrank_test for n=2).
    We have to be careful here: if there are n groups, then there are n*(n-1)/2 pairs -- so many pairs increase
    the chance that here will exist a significantly different pair purely by chance. For this reason, we use the
    Bonferroni correction (rewight the alpha value higher to accomidate the multiple tests).


    Parameters:
      event_durations: a (n,) numpy array the (partial) lifetimes of all individuals
      groups: a (n,) numpy array of unique group labels for each individual.
      event_observed: a (n,) numpy array of event_observed events: 1 if observed death, 0 if censored. Defaults
          to all observed.
      alpha: the level of signifiance desired.
      t_0: the final time to compare the series' up to. Defaults to all.
      bonferroni: If true, uses the Bonferroni correction to compare the M=n(n-1)/2 pairs, i.e alpha = alpha/M
            See (here)[http://en.wikipedia.org/wiki/Bonferroni_correction].
      kwargs: add keywords and meta-data to the experiment summary.

    Returns:
        R: a (n,n) dataframe of StatisticalResults (None on the diagonal)

    """

    if event_observed is None:
        event_observed = np.ones((event_durations.shape[0], 1))

    n = np.max(event_durations.shape)
    assert n == np.max(event_durations.shape) == np.max(event_observed.shape), "inputs must be of the same length."
    groups, event_durations, event_observed = map(lambda x: pd.Series(np.reshape(x, (n,))), [groups, event_durations, event_observed])

    unique_groups = np.unique(groups)

    n = unique_groups.shape[0]

    if bonferroni:
        m = 0.5 * n * (n - 1)
        alpha = 1 - (1 - alpha) / m

    R = np.zeros((n, n), dtype=object)

    np.fill_diagonal(R, None)

    for i1, i2 in combinations(np.arange(n), 2):
        g1, g2 = unique_groups[[i1, i2]]
        ix1, ix2 = (groups == g1), (groups == g2)
        test_name = str(g1) + " vs. " + str(g2)
        result = logrank_test(event_durations.ix[ix1], event_durations.ix[ix2],
                              event_observed.ix[ix1], event_observed.ix[ix2],
                              alpha=alpha, t_0=t_0, use_bonferroni=bonferroni,
                              test_name=test_name, **kwargs)
        R[i1, i2], R[i2, i1] = result, result

    return pd.DataFrame(R, columns=unique_groups, index=unique_groups)


def multivariate_logrank_test(event_durations, groups, event_observed=None,
                              alpha=0.95, t_0=-1, **kwargs):
    """
    This test is a generalization of the logrank_test: it can deal with n>2 populations (and should
      be equal when n=2):

    H_0: all event series are from the same generating processes
    H_A: there exist atleast one group that differs from the other.

    Parameters:
      event_durations: a (n,) numpy array the (partial) lifetimes of all individuals
      groups: a (n,) numpy array of unique group labels for each individual.
      event_observed: a (n,) numpy array of event observations: 1 if observed death, 0 if censored. Defaults
          to all observed.
      alpha: the level of significance desired.
      t_0: the final time to compare the series' up to. Defaults to all.
      kwargs: add keywords and meta-data to the experiment summary.

    Returns
      results: a StatisticalResult object with properties 'p_value', 'summary', 'test_statistic', 'test_result'

    """
    if not (0 < alpha <= 1.):
        raise ValueError('alpha parameter must be between 0 and 1.')

    event_durations, groups = np.asarray(event_durations), np.asarray(groups)
    if event_observed is None:
        event_observed = np.ones((event_durations.shape[0], 1))
    else:
        event_observed = np.asarray(event_observed)

    n = np.max(event_durations.shape)
    assert n == np.max(event_durations.shape) == np.max(event_observed.shape), "inputs must be of the same length."
    groups, event_durations, event_observed = map(lambda x: pd.Series(np.reshape(x, (n,))), [groups, event_durations, event_observed])

    unique_groups, rm, obs, _ = group_survival_table_from_events(groups, event_durations, event_observed, limit=t_0)
    n_groups = unique_groups.shape[0]

    # compute the factors needed
    N_j = obs.sum(0).values
    n_ij = (rm.sum(0).values - rm.cumsum(0).shift(1).fillna(0))
    d_i = obs.sum(1)
    n_i = rm.values.sum() - rm.sum(1).cumsum().shift(1).fillna(0)
    ev = n_ij.mul(d_i / n_i, axis='index').sum(0)

    # vector of observed minus expected
    Z_j = N_j - ev

    assert abs(Z_j.sum()) < 10e-8, "Sum is not zero."  # this should move to a test eventually.

    # compute covariance matrix
    factor = (((n_i - d_i) / (n_i - 1)).replace(np.inf, 1)) * d_i
    n_ij['_'] = n_i.values
    V_ = n_ij.mul(np.sqrt(factor) / n_i, axis='index').fillna(1)
    V = -np.dot(V_.T, V_)
    ix = np.arange(n_groups)
    V[ix, ix] = -V[-1, ix] + V[ix, ix]
    V = V[:-1, :-1]

    # take the first n-1 groups
    U = Z_j.iloc[:-1].dot(np.linalg.pinv(V[:-1, :-1]).dot(Z_j.iloc[:-1]))  # Z.T*inv(V)*Z

    # compute the p-values and tests
    test_result, p_value = chisq_test(U, n_groups - 1, alpha)

    return StatisticalResult(test_result, p_value, U, t_0=t_0, test='logrank',
                             alpha=alpha, null_distribution='chi squared',
                             df=n_groups - 1, **kwargs)


class StatisticalResult(object):

    def __init__(self, test_result, p_value, test_statistic, **kwargs):
        self.p_value = p_value
        self.test_statistic = test_statistic
        self.test_result = "Reject Null" if test_result else "Cannot Reject Null"
        self.is_significant = test_result is True

        for kw, value in kwargs.items():
            setattr(self, kw, value)

        self._kwargs = kwargs

    def print_summary(self):
        print(self.__unicode__())

    @property
    def summary(self):
        cols = ['p-value', 'test_statistic', 'test_result', 'is_significant']
        return pd.DataFrame([[self.p_value, self.test_statistic, self.test_result, self.is_significant]], columns=cols)

    def __repr__(self):
        return "<lifelines.StatisticalResult: \n%s\n>" % self.__unicode__()

    def __unicode__(self):
        HEADER = "   __ p-value ___|__ test statistic __|____ test result ____|__ is significant __"
        meta_data = self._pretty_print_meta_data(self._kwargs)
        s = ""
        s += "Results\n"
        s += meta_data + "\n"
        s += HEADER + "\n"
        s += '{:>16.5f} | {:>18.3f} |  {: ^19}| {: ^18}'.format(self.p_value, self.test_statistic, self.test_result, 'True' if self.is_significant else 'False')
        return s

    def _pretty_print_meta_data(self, dictionary):
        s = ""
        for k, v in dictionary.items():
            s += "   " + str(k).replace('_', ' ') + ": " + str(v) + "\n"
        return s


def chisq_test(U, degrees_freedom, alpha):
    p_value = stats.chi2.sf(U, degrees_freedom)
    if p_value < 1 - alpha:
        return True, p_value
    else:
        return None, p_value


def two_sided_z_test(Z, alpha):
    p_value = 1 - np.max(stats.norm.cdf(Z), 1 - stats.norm.cdf(Z))
    if p_value < 1 - alpha / 2.:
        return True, p_value
    else:
        return None, p_value
