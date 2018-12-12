# -*- coding: utf-8 -*-
from __future__ import print_function
from itertools import combinations

import numpy as np
from scipy import stats
import pandas as pd

from lifelines.utils import group_survival_table_from_events, significance_code, significance_codes_as_text


def sample_size_necessary_under_cph(power, ratio_of_participants, p_exp, p_con, postulated_hazard_ratio, alpha=0.05):
    """
    This computes the sample size for needed power to compare two groups under a Cox
    Proportional Hazard model.

    Parameters
    ----------

    power : float
        power to detect the magnitude of the hazard ratio as small as that specified by postulated_hazard_ratio.
    ratio_of_participants: ratio of participants in experimental group over control group.
    
    p_exp : float
        probability of failure in experimental group over period of study.
    
    p_con : float 
        probability of failure in control group over period of study
    
    postulated_hazard_ratio : float 
        the postulated hazard ratio
    
    alpha : float, optional (default=0.05)
        type I error rate


    Returns
    -------

    n_exp : integer
        the samples sizes need for the experiment to achieve desired power

    n_con : integer
        the samples sizes need for the control group to achieve desired power


    Examples
    --------
    >>> from lifelines.statistics import sample_size_necessary_under_cph
    >>> 
    >>> desired_power = 0.8
    >>> ratio_of_participants = 1.
    >>> p_exp = 0.25
    >>> p_con = 0.35
    >>> postulated_hazard_ratio = 0.7
    >>> n_exp, n_con = sample_size_necessary_under_cph(desired_power, ratio_of_participants, p_exp, p_con, postulated_hazard_ratio)
    >>> # (421, 421)

    Notes
    -----
    `Reference <https://cran.r-project.org/web/packages/powerSurvEpi/powerSurvEpi.pdf>`_.

    See Also
    --------
    power_under_cph
    """

    def z(p):
        return stats.norm.ppf(p)

    m = (
        1.0
        / ratio_of_participants
        * ((ratio_of_participants * postulated_hazard_ratio + 1.0) / (postulated_hazard_ratio - 1.0)) ** 2
        * (z(1.0 - alpha / 2.0) + z(power)) ** 2
    )

    n_exp = m * ratio_of_participants / (ratio_of_participants * p_exp + p_con)
    n_con = m / (ratio_of_participants * p_exp + p_con)

    return int(np.ceil(n_exp)), int(np.ceil(n_con))


def power_under_cph(n_exp, n_con, p_exp, p_con, postulated_hazard_ratio, alpha=0.05):
    """
    This computes the power of the hypothesis test that the two groups, experiment and control,
    have different hazards (that is, the relative hazard ratio is different from 1.)

    Parameters
    ----------

    n_exp : integer
        size of the experiment group.
    
    n_con : integer
        size of the control group.
    
    p_exp : float
        probability of failure in experimental group over period of study.
    
    p_con : float
        probability of failure in control group over period of study
    
    postulated_hazard_ratio : float 
    the postulated hazard ratio
    
    alpha : float, optional (default=0.05)
        type I error rate

    Returns
    -------

    power : float
        power to detect the magnitude of the hazard ratio as small as that specified by postulated_hazard_ratio.


    Notes
    -----
    `Reference <https://cran.r-project.org/web/packages/powerSurvEpi/powerSurvEpi.pdf>`_.


    See Also
    --------
    sample_size_necessary_under_cph
    """

    def z(p):
        return stats.norm.ppf(p)

    m = n_exp * p_exp + n_con * p_con
    k = float(n_exp) / float(n_con)
    return stats.norm.cdf(
        np.sqrt(k * m) * abs(postulated_hazard_ratio - 1) / (k * postulated_hazard_ratio + 1) - z(1 - alpha / 2.0)
    )


def logrank_test(
    event_times_A, event_times_B, event_observed_A=None, event_observed_B=None, alpha=0.95, t_0=-1, **kwargs
):
    """
    Measures and reports on whether two intensity processes are different. That is, given two
    event series, determines whether the data generating processes are statistically different.
    The test-statistic is chi-squared under the null hypothesis.

     - H_0: both event series are from the same generating processes
     - H_A: the event series are from different generating processes.


    This implicitly uses the log-rank weights.

    Parameters
    ----------

    event_times_A: iterable
        a (n,) list-like of event durations (birth to death,...) for the first population.

    event_times_B: iterable
        a (n,) list-like of event durations (birth to death,...) for the second population.

    censorship_A: iterable, optional
        a (n,) list-like of censorship flags, (1 if observed, 0 if not), for the first population. 
        Default assumes all observed.

    censorship_B: iterable, optional
        a (n,) list-like of censorship flags, (1 if observed, 0 if not), for the second population. 
        Default assumes all observed.

    t_0: float, optional (default=-1)
        the period under observation, -1 for all time.

    alpha: float, optional (default=0.95)
        the confidence level

    kwargs: 
        add keywords and meta-data to the experiment summary


    Returns
    -------

    results : StatisticalResult
      a StatisticalResult object with properties 'p_value', 'summary', 'test_statistic', 'print_summary'

    Examples
    --------
    >>> from lifelines.statistics import logrank_test
    >>> results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    >>> results.print_summary()
    >>> print(results.p_value)        # 0.46759
    >>> print(results.test_statistic) # 0.528

    Notes
    -----
    This is a special case of the function ``multivariate_logrank_test``, which is used internally. 
    See Survival and Event Analysis, page 108.

    See Also
    --------
    multivariate_logrank_test
    pairwise_logrank_test
    """
    event_times_A, event_times_B = (np.array(event_times_A), np.array(event_times_B))
    if event_observed_A is None:
        event_observed_A = np.ones(event_times_A.shape[0])
    if event_observed_B is None:
        event_observed_B = np.ones(event_times_B.shape[0])

    event_times = np.r_[event_times_A, event_times_B]
    groups = np.r_[np.zeros(event_times_A.shape[0], dtype=int), np.ones(event_times_B.shape[0], dtype=int)]
    event_observed = np.r_[event_observed_A, event_observed_B]
    return multivariate_logrank_test(event_times, groups, event_observed, alpha=alpha, t_0=t_0, **kwargs)


def pairwise_logrank_test(
    event_durations, groups, event_observed=None, alpha=0.95, t_0=-1, bonferroni=True, **kwargs
):  # pylint: disable=too-many-locals

    """
    Perform the logrank test pairwise for all n>2 unique groups (use the more appropriate logrank_test for n=2).
    We have to be careful here: if there are n groups, then there are n*(n-1)/2 pairs -- so many pairs increase
    the chance that here will exist a significantly different pair purely by chance. For this reason, we use the
    Bonferroni correction (rewight the alpha value higher to accomidate the multiple tests).


    Parameters
    ----------

    event_durations: iterable
        a (n,) list-like representing the (possibly partial) durations of all individuals

    groups: iterable
        a (n,) list-like of unique group labels for each individual.

    event_observed: iterable, optional
        a (n,) list-like of event_observed events: 1 if observed death, 0 if censored. Defaults to all observed.

    t_0: float, optional (default=-1)
        the period under observation, -1 for all time.

    alpha: float, optional (default=0.95)
        the confidence level

    bonferroni: boolean, optional (default=True)
        If True, uses the Bonferroni correction to compare the M=n(n-1)/2 pairs, i.e alpha = alpha/M.

    kwargs: 
        add keywords and meta-data to the experiment summary.


    Returns
    -------

    results : DataFrame
        a (n,n) dataframe of StatisticalResults (None on the diagonal)


    See Also
    --------
    multivariate_logrank_test
    logrank_test
    """

    if event_observed is None:
        event_observed = np.ones((event_durations.shape[0], 1))

    n = np.max(np.asarray(event_durations).shape)

    groups, event_durations, event_observed = map(
        lambda x: np.asarray(x).reshape(n), [groups, event_durations, event_observed]
    )

    if not (n == event_durations.shape[0] == event_observed.shape[0]):
        raise ValueError("inputs must be of the same length.")

    groups, event_durations, event_observed = pd.Series(groups), pd.Series(event_durations), pd.Series(event_observed)

    unique_groups = np.unique(groups)

    n_unique_groups = unique_groups.shape[0]

    if bonferroni:
        m = 0.5 * n_unique_groups * (n_unique_groups - 1)
        alpha = 1 - (1 - alpha) / m

    R = np.zeros((n_unique_groups, n_unique_groups), dtype=object)

    np.fill_diagonal(R, None)

    for i1, i2 in combinations(np.arange(n_unique_groups), 2):
        g1, g2 = unique_groups[[i1, i2]]
        ix1, ix2 = (groups == g1), (groups == g2)
        test_name = str(g1) + " vs. " + str(g2)
        result = logrank_test(
            event_durations.loc[ix1],
            event_durations.loc[ix2],
            event_observed.loc[ix1],
            event_observed.loc[ix2],
            alpha=alpha,
            t_0=t_0,
            use_bonferroni=bonferroni,
            test_name=test_name,
            **kwargs
        )
        R[i1, i2], R[i2, i1] = result, result

    return pd.DataFrame(R, columns=unique_groups, index=unique_groups)


def multivariate_logrank_test(
    event_durations, groups, event_observed=None, alpha=0.95, t_0=-1, **kwargs
):  # pylint: disable=too-many-locals

    """
    This test is a generalization of the logrank_test: it can deal with n>2 populations (and should
    be equal when n=2):

     - H_0: all event series are from the same generating processes
     - H_A: there exist atleast one group that differs from the other.


    Parameters
    ----------

    event_durations: iterable
        a (n,) list-like representing the (possibly partial) durations of all individuals

    groups: iterable
        a (n,) list-like of unique group labels for each individual.

    event_observed: iterable, optional
        a (n,) list-like of event_observed events: 1 if observed death, 0 if censored. Defaults to all observed.

    t_0: float, optional (default=-1)
        the period under observation, -1 for all time.

    alpha: float, optional (default=0.95)
        the confidence level

    kwargs: 
        add keywords and meta-data to the experiment summary.


    Returns
    -------

    results : StatisticalResult
       a StatisticalResult object with properties 'p_value', 'summary', 'test_statistic', 'print_summary'

    Examples
    --------

    >>> df = pd.DataFrame({
    >>>    'durations': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
    >>>    'events': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    >>>    'groups': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    >>> })
    >>> result = multivariate_logrank_test(df['durations'], df['groups'], df['events'])
    >>> result.test_statistic
    >>> result.p_value
    >>> result.print_summary()


    >>> # numpy example
    >>> G = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    >>> T = [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7]
    >>> E = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    >>> result = multivariate_logrank_test(T, G, E)
    >>> result.test_statistic


    See Also
    --------
    pairwise_logrank_test
    logrank_test
    """
    if not (0 < alpha <= 1.0):
        raise ValueError("alpha parameter must be between 0 and 1.")

    event_durations, groups = np.asarray(event_durations), np.asarray(groups)
    if event_observed is None:
        event_observed = np.ones((event_durations.shape[0], 1))
    else:
        event_observed = np.asarray(event_observed)

    n = np.max(event_durations.shape)
    assert n == np.max(event_durations.shape) == np.max(event_observed.shape), "inputs must be of the same length."
    groups, event_durations, event_observed = map(
        lambda x: pd.Series(np.asarray(x).reshape(n)), [groups, event_durations, event_observed]
    )

    unique_groups, rm, obs, _ = group_survival_table_from_events(groups, event_durations, event_observed, limit=t_0)
    n_groups = unique_groups.shape[0]

    # compute the factors needed
    N_j = obs.sum(0).values
    n_ij = rm.sum(0).values - rm.cumsum(0).shift(1).fillna(0)
    d_i = obs.sum(1)
    n_i = rm.values.sum() - rm.sum(1).cumsum().shift(1).fillna(0)
    ev = n_ij.mul(d_i / n_i, axis="index").sum(0)

    # vector of observed minus expected
    Z_j = N_j - ev

    assert abs(Z_j.sum()) < 10e-8, "Sum is not zero."  # this should move to a test eventually.

    # compute covariance matrix
    factor = (((n_i - d_i) / (n_i - 1)).replace([np.inf, np.nan], 1)) * d_i / n_i ** 2
    n_ij["_"] = n_i.values
    V_ = n_ij.mul(np.sqrt(factor), axis="index").fillna(0)
    V = -np.dot(V_.T, V_)
    ix = np.arange(n_groups)
    V[ix, ix] = V[ix, ix] - V[-1, ix]
    V = V[:-1, :-1]

    # take the first n-1 groups
    U = Z_j.iloc[:-1].dot(np.linalg.pinv(V[:-1, :-1])).dot(Z_j.iloc[:-1])  # Z.T*inv(V)*Z

    # compute the p-values and tests
    _, p_value = chisq_test(U, n_groups - 1, alpha)

    return StatisticalResult(
        p_value, U, t_0=t_0, alpha=alpha, null_distribution="chi squared", df=n_groups - 1, **kwargs
    )


class StatisticalResult(object):
    def __init__(self, p_value, test_statistic, **kwargs):
        self.p_value = p_value
        self.test_statistic = test_statistic

        for kw, value in kwargs.items():
            setattr(self, kw, value)

        self._kwargs = kwargs

    def print_summary(self):
        """
        Prints a prettier version of the ``summary`` DataFrame with more metadata.

        """
        print(self.__unicode__())

    @property
    def summary(self):
        """
        
        Returns
        -------

        summary: DataFrame
            a DataFrame containing the test statistics and the p-value

        """
        cols = ["test_statistic", "p"]
        return pd.DataFrame([[self.test_statistic, self.p_value]], columns=cols)

    def __repr__(self):
        return "<lifelines.StatisticalResult: \n%s\n>" % self.__unicode__()

    def __unicode__(self):
        # pylint: disable=unnecessary-lambda
        meta_data = self._pretty_print_meta_data(self._kwargs)
        df = self.summary
        df[""] = significance_code(self.p_value)

        s = ""
        s += "\n" + meta_data + "\n\n"
        s += df.to_string(float_format=lambda f: "{:4.4f}".format(f), index=False)

        s += "\n---"
        s += "\n" + significance_codes_as_text()
        return s

    def _pretty_print_meta_data(self, dictionary):
        return ", ".join([str(k) + "=" + str(v) for k, v in dictionary.items()])


def chisq_test(U, degrees_freedom, alpha):
    p_value = stats.chi2.sf(U, degrees_freedom)
    if p_value < 1 - alpha:
        return True, p_value
    return None, p_value


def two_sided_z_test(Z, alpha):
    p_value = 1 - np.max(stats.norm.cdf(Z), 1 - stats.norm.cdf(Z))
    if p_value < 1 - alpha / 2.0:
        return True, p_value
    return None, p_value
