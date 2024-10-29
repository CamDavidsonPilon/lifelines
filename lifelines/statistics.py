# -*- coding: utf-8 -*-

from itertools import combinations
import warnings

import numpy as np
from scipy import stats
import pandas as pd

from lifelines import utils
from lifelines.utils import (
    group_survival_table_from_events,
    string_rjustify,
    format_p_value,
    format_floats,
    interpolate_at_times_and_return_pandas,
    _expected_value_of_survival_up_to_t,
    _expected_value_of_survival_squared_up_to_t,
)

from lifelines import KaplanMeierFitter

__all__ = [
    "StatisticalResult",
    "logrank_test",
    "multivariate_logrank_test",
    "pairwise_logrank_test",
    "survival_difference_at_fixed_point_in_time_test",
    "proportional_hazard_test",
    "power_under_cph",
    "sample_size_necessary_under_cph",
]


class StatisticalResult:
    """
    This class holds the result of statistical tests with a nice printer wrapper to display the results.

    Note
    -----
    This class' API changed in version 0.16.0.

    Parameters
    ----------
    p_value: iterable or float
        the p-values of a statistical test(s)
    test_statistic: iterable or float
        the test statistics of a statistical test(s). Must be the same size as p-values if iterable.
    test_name: string
        the test that was used. lifelines should set this.
    name: iterable or string
        if this class holds multiple results (ex: from a pairwise comparison), this can hold the names. Must be the same size as p-values if iterable.
    kwargs:
        additional information to attach to the object and display in ``print_summary()``.

    """

    def __init__(self, p_value, test_statistic, name=None, test_name=None, **kwargs):
        self.p_value = p_value
        self.test_statistic = test_statistic
        self.test_name = test_name

        self._p_value = utils._to_1d_array(p_value)
        self._test_statistic = utils._to_1d_array(test_statistic)

        assert len(self._p_value) == len(self._test_statistic)

        if name is not None:
            self.name = utils._to_list(name)
            assert len(self.name) == len(self._test_statistic)
        else:
            self.name = None

        for kw, value in kwargs.items():
            setattr(self, kw, value)

        kwargs["test_name"] = test_name
        self._kwargs = kwargs

    def _print_specific_style(self, style, decimals=2, **kwargs):
        """
        Parameters
        -----------

        style: str
          One of {'ascii', 'html', 'latex'}

        """
        if style == "html":
            return self._html_print(decimals=decimals, **kwargs)
        elif style == "ascii":
            return self._ascii_print(decimals=decimals, **kwargs)
        elif style == "latex":
            return self._latex_print(decimals=decimals, **kwargs)
        else:
            raise ValueError("style not available.")

    def print_summary(self, decimals=2, style=None, **kwargs):
        """
        Print summary statistics describing the results.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        style: string,
            {html, ascii, latex}, default ascii
        kwargs:
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """
        if style is not None:
            self._print_specific_style(style, decimals=decimals, **kwargs)
        else:
            try:
                from IPython.display import display

                display(self)
            except ImportError:
                self._ascii_print(decimals=decimals, **kwargs)

    def _html_print(self, decimals=2, **kwargs):
        print(self.to_html(decimals, **kwargs))

    def _latex_print(self, decimals=2, **kwargs):
        print(self.to_latex(decimals, **kwargs))

    def _ascii_print(self, decimals=2, **kwargs):
        print(self.to_ascii(decimals, **kwargs))

    def to_html(self, decimals=2, **kwargs):
        extra_kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))
        summary_df = self.summary

        headers = []
        for k, v in extra_kwargs.items():
            headers.append((k, v))

        header_df = pd.DataFrame.from_records(headers).set_index(0)
        header_html = header_df.to_html(header=False, notebook=True, index_names=False)

        summary_html = summary_df.to_html(float_format=format_floats(decimals), formatters={**{"p": format_p_value(decimals)}})

        return header_html + summary_html

    def to_latex(self, decimals=2, **kwargs):
        # This is using the new Style object in Pandas. Previously df.to_latex was giving a warning.
        s = self.summary.style
        s = s.format(precision=decimals)
        return s.to_latex()

    @property
    def summary(self):
        """

        Returns
        -------
        DataFrame
            a DataFrame containing the test statistics and the p-value

        """
        cols = ["test_statistic", "p"]

        # test to see if self.names is a tuple
        if self.name and isinstance(self.name[0], tuple):
            index = pd.MultiIndex.from_tuples(self.name)
        else:
            index = self.name

        df = pd.DataFrame(list(zip(self._test_statistic, self._p_value)), columns=cols, index=index).sort_index()
        df["-log2(p)"] = -utils.quiet_log2(df["p"])
        return df

    def to_ascii(self, decimals=2, **kwargs):
        extra_kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))
        meta_data = self._stringify_meta_data(extra_kwargs)

        df = self.summary

        s = "<lifelines.StatisticalResult: {0}>".format(self.test_name)
        s += "\n" + meta_data + "\n"
        s += "---\n"
        s += df.to_string(
            float_format=format_floats(decimals), index=self.name is not None, formatters={"p": format_p_value(decimals)}
        )

        return s

    def _stringify_meta_data(self, dictionary):
        longest_key = max([len(k) for k in dictionary])
        justify = string_rjustify(longest_key)
        s = ""
        for k, v in dictionary.items():
            s += "{} = {}\n".format(justify(k), v)

        return s

    def __add__(self, other):
        """useful for aggregating results easily"""
        p_values = np.r_[self._p_value, other._p_value]
        test_statistics = np.r_[self._test_statistic, other._test_statistic]
        names = self.name + other.name
        kwargs = dict(list(self._kwargs.items()) + list(other._kwargs.items()))
        return StatisticalResult(p_values, test_statistics, name=names, **kwargs)

    def _repr_latex_(self):
        return self.to_latex()

    def _repr_html_(self):
        return self.to_html()

    def __repr__(self):
        return self.to_ascii()


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
    .. code:: python

        from lifelines.statistics import sample_size_necessary_under_cph

        desired_power = 0.8
        ratio_of_participants = 1.
        p_exp = 0.25
        p_con = 0.35
        postulated_hazard_ratio = 0.7
        n_exp, n_con = sample_size_necessary_under_cph(desired_power, ratio_of_participants, p_exp, p_con, postulated_hazard_ratio)
        # (421, 421)

    References
    -----------
    https://cran.r-project.org/web/packages/powerSurvEpi/powerSurvEpi.pdf

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


def power_under_cph(n_exp, n_con, p_exp, p_con, postulated_hazard_ratio, alpha=0.05) -> float:
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

    float:
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


def survival_difference_at_fixed_point_in_time_test(point_in_time, fitterA, fitterB, **result_kwargs) -> StatisticalResult:
    """
    Often analysts want to compare the survival-ness of groups at specific times, rather than comparing the entire survival curves against each other.
    For example, analysts may be interested in 5-year survival. Statistically comparing the naive Kaplan-Meier points at a specific time
    actually has reduced power (see [1]). By transforming the survival function, we can recover more power. This function uses
    the log(-log(·)) transformation.


    Parameters
    ----------
    point_in_time: float,
        the point in time to analyze the survival curves at.

    fitterA:
        A lifelines univariate model fitted to the data. This can be a ``KaplanMeierFitter``, ``WeibullFitter``, etc.

    fitterB:
        the second lifelines model to compare against.

    result_kwargs:
        add keywords and meta-data to the experiment summary

    Returns
    -------

    StatisticalResult
      a StatisticalResult object with properties ``p_value``, ``summary``, ``test_statistic``, ``print_summary``

    Examples
    --------
    .. code:: python

        T1 = [1, 4, 10, 12, 12, 3, 5.4]
        E1 = [1, 0, 1,  0,  1,  1, 1]
        kmf1 = KaplanMeierFitter().fit(T1, E1)

        T2 = [4, 5, 7, 11, 14, 20, 8, 8]
        E2 = [1, 1, 1, 1,  1,  1,  1, 1]
        kmf2 = KaplanMeierFitter().fit(T2, E2)

        from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
        results = survival_difference_at_fixed_point_in_time_test(12.0, kmf1, kmf2)

        results.print_summary()
        print(results.p_value)        # 0.77
        print(results.test_statistic) # 0.09

    Notes
    -----
    1. Other transformations are possible, but Klein et al. [1] showed that the log(-log(·)) transform has the most desirable
    statistical properties.

    2. The API of this function changed in v0.25.3. This new API allows for right, left and interval censoring models to be tested.


    References
    -----------

    [1] Klein, J. P., Logan, B. , Harhoff, M. and Andersen, P. K. (2007), Analyzing survival curves at a fixed point in time. Statist. Med., 26: 4505-4519. doi:10.1002/sim.2864

    """
    if type(fitterB) != type(fitterA):
        warnings.warn(
            "This test compares survival functions, but your fitters are estimating the survival functions differently. This means that this test is also testing the different ways to estimate the survival function and will be unreliable.",
            UserWarning,
        )

    log = np.log
    clog = lambda s: log(-log(s))

    sA_t = fitterA.predict(point_in_time)
    sB_t = fitterB.predict(point_in_time)

    from lifelines.fitters import NonParametricUnivariateFitter, ParametricUnivariateFitter

    if isinstance(fitterA, NonParametricUnivariateFitter):
        sigma_sqA = interpolate_at_times_and_return_pandas(fitterA._cumulative_sq_, point_in_time)
    elif isinstance(fitterA, ParametricUnivariateFitter):
        sigma_sqA = fitterA._compute_variance_of_transform(fitterA._survival_function, [point_in_time]).squeeze()

    if isinstance(fitterB, NonParametricUnivariateFitter):
        sigma_sqB = interpolate_at_times_and_return_pandas(fitterB._cumulative_sq_, point_in_time)
    elif isinstance(fitterB, ParametricUnivariateFitter):
        sigma_sqB = fitterB._compute_variance_of_transform(fitterB._survival_function, [point_in_time]).squeeze()

    X = (clog(sA_t) - clog(sB_t)) ** 2 / (sigma_sqA / log(sA_t) ** 2 + sigma_sqB / log(sB_t) ** 2)
    p_value = _chisq_test_p_value(X, 1)

    return StatisticalResult(
        p_value,
        X,
        null_distribution="chi squared",
        degrees_of_freedom=1,
        point_in_time=point_in_time,
        test_name="survival_difference_at_fixed_point_in_time_test",
        fitterA=fitterA,
        fitterB=fitterB,
        **result_kwargs
    )


def logrank_test(
    durations_A,
    durations_B,
    event_observed_A=None,
    event_observed_B=None,
    t_0=-1,
    weights_A=None,
    weights_B=None,
    weightings=None,
    **kwargs
) -> StatisticalResult:
    r"""
    Measures and reports on whether two intensity processes are different. That is, given two
    event series, determines whether the data generating processes are statistically different.
    The test-statistic is chi-squared under the null hypothesis. Let :math:`h_i(t)` be the hazard ratio of
    group :math:`i` at time :math:`t`, then:

    .. math::
        \begin{align}
         & H_0: h_1(t) = h_2(t) \\
         & H_A: h_1(t) = c h_2(t), \;\; c \ne 1
        \end{align}

    This implicitly uses the log-rank weights.

    Note
    -----
    - *lifelines* logrank implementation only handles right-censored data.

    - The logrank test has maximum power when the assumption of proportional hazards is true. As a consequence, if the survival curves cross, the logrank test will give an inaccurate assessment of differences.

    - This implementation is a special case of the function ``multivariate_logrank_test``, which is used internally. See Survival and Event Analysis, page 108.

    - There are only disadvantages to using the log-rank test versus using the Cox regression. See more `here <https://discourse.datamethods.org/t/when-is-log-rank-preferred-over-univariable-cox-regression/2344>`_ for a discussion. To convert to using the Cox regression:

    .. code:: python

        from lifelines import CoxPHFitter

        dfA = pd.DataFrame({'E': event_observed_A, 'T': durations_A, 'groupA': 1})
        dfB = pd.DataFrame({'E': event_observed_B, 'T': durations_B, 'groupA': 0})
        df = pd.concat([dfA, dfB])

        cph = CoxPHFitter().fit(df, 'T', 'E')
        cph.print_summary()



    Parameters
    ----------

    durations_A: iterable
        a (n,) list-like of event durations (birth to death,...) for the first population.

    durations_B: iterable
        a (n,) list-like of event durations (birth to death,...) for the second population.

    event_observed_A: iterable, optional
        a (n,) list-like of censorship flags, (1 if observed, 0 if not), for the first population.
        Default assumes all observed.

    event_observed_B: iterable, optional
        a (n,) list-like of censorship flags, (1 if observed, 0 if not), for the second population.
        Default assumes all observed.

    weights_A: iterable, optional
        case weights

    weights_B: iterable, optional
        case weights

    t_0: float, optional (default=-1)
        The final time period under observation, and subjects who experience the event after this time are set to be censored.
        Specify -1 to use all time.

    weightings: str, optional
        apply a weighted logrank test: options are "wilcoxon" for Wilcoxon (also known as Breslow), "tarone-ware"
        for Tarone-Ware, "peto" for Peto test and "fleming-harrington" for Fleming-Harrington test.
        These are useful for testing for early or late differences in the survival curve. For the Fleming-Harrington
        test, keyword arguments p and q must also be provided with non-negative values.

        Weightings are applied at the ith ordered failure time, :math:`t_{i}`, according to:
            Wilcoxon: :math:`n_i`
            Tarone-Ware: :math:`\sqrt{n_i}`
            Peto: :math:`\bar{S}(t_i)`
            Fleming-Harrington: :math:`\hat{S}(t_i)^p \times (1 - \hat{S}(t_i))^q`

            where :math:`n_i` is the number at risk just prior to time :math:`t_{i}`, :math:`\bar{S}(t_i)` is
            Peto-Peto's modified survival estimate and :math:`\hat{S}(t_i)` is the left-continuous
            Kaplan-Meier survival estimate at time :math:`t_{i}`.

    Returns
    -------

    StatisticalResult
      a StatisticalResult object with properties ``p_value``, ``summary``, ``test_statistic``, ``print_summary``

    Examples
    --------

    .. code:: python

        T1 = [1, 4, 10, 12, 12, 3, 5.4]
        E1 = [1, 0, 1,  0,  1,  1, 1]

        T2 = [4, 5, 7, 11, 14, 20, 8, 8]
        E2 = [1, 1, 1, 1,  1,  1,  1, 1]

        from lifelines.statistics import logrank_test
        results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)

        results.print_summary()
        print(results.p_value)        # 0.7676
        print(results.test_statistic) # 0.0872


    See Also
    --------
    multivariate_logrank_test
    pairwise_logrank_test
    survival_difference_at_fixed_point_in_time_test
    """
    event_times_A, event_times_B = (np.array(durations_A), np.array(durations_B))
    if event_observed_A is None:
        event_observed_A = np.ones(event_times_A.shape[0])
    if event_observed_B is None:
        event_observed_B = np.ones(event_times_B.shape[0])
    if weights_A is None:
        weights_A = np.ones(event_times_A.shape[0])
    if weights_B is None:
        weights_B = np.ones(event_times_B.shape[0])

    event_times = np.r_[event_times_A, event_times_B]
    groups = np.r_[np.zeros(event_times_A.shape[0], dtype=int), np.ones(event_times_B.shape[0], dtype=int)]
    event_observed = np.r_[event_observed_A, event_observed_B]
    weights = np.r_[weights_A, weights_B]
    return multivariate_logrank_test(
        event_times, groups, event_observed, t_0=t_0, weights=weights, test_name="logrank_test", weightings=weightings, **kwargs
    )


def pairwise_logrank_test(
    event_durations, groups, event_observed=None, t_0=-1, weightings=None, **kwargs
) -> StatisticalResult:  # pylint: disable=too-many-locals

    r"""
    Perform the logrank test pairwise for all :math:`n \ge 2` unique groups.

    Parameters
    ----------

    event_durations: iterable
        a (n,) list-like representing the (possibly partial) durations of all individuals

    groups: iterable
        a (n,) list-like of unique group labels for each individual.

    event_observed: iterable, optional
        a (n,) list-like of event_observed events: 1 if observed death, 0 if censored. Defaults to all observed.

    t_0: float, optional (default=-1)
        The final time period under observation, and subjects who experience the event after this time are set to be censored.
        Specify -1 to use all time.

    weightings: str, optional
        apply a weighted logrank test: options are "wilcoxon" for Wilcoxon (also known as Breslow), "tarone-ware"
        for Tarone-Ware, "peto" for Peto test and "fleming-harrington" for Fleming-Harrington test.
        These are useful for testing for early or late differences in the survival curve. For the Fleming-Harrington
        test, keyword arguments p and q must also be provided with non-negative values.

        Weightings are applied at the ith ordered failure time, :math:`t_{i}`, according to:
            Wilcoxon: :math:`n_i`
            Tarone-Ware: :math:`\sqrt{n_i}`
            Peto: :math:`\bar{S}(t_i)`
            Fleming-Harrington: :math:`\hat{S}(t_i)^p \times (1 - \hat{S}(t_i))^q`

            where :math:`n_i` is the number at risk just prior to time :math:`t_{i}`, :math:`\bar{S}(t_i)` is
            Peto-Peto's modified survival estimate and :math:`\hat{S}(t_i)` is the left-continuous
            Kaplan-Meier survival estimate at time :math:`t_{i}`.
    kwargs:
        add keywords and meta-data to the experiment summary.


    Returns
    -------

    StatisticalResult
        a StatisticalResult object that contains all the pairwise comparisons (try ``StatisticalResult.summary`` or ``StatisticalResult.print_summary``)


    See Also
    --------
    multivariate_logrank_test
    logrank_test
    """

    if event_observed is None:
        event_observed = np.ones((event_durations.shape[0], 1))

    n = np.max(np.asarray(event_durations).shape)

    groups, event_durations, event_observed = map(lambda x: np.asarray(x).reshape(n), [groups, event_durations, event_observed])

    if not (n == event_durations.shape[0] == event_observed.shape[0]):
        raise ValueError("inputs must be of the same length.")

    groups, event_durations, event_observed = pd.Series(groups), pd.Series(event_durations), pd.Series(event_observed)

    unique_groups = np.unique(groups)

    n_unique_groups = unique_groups.shape[0]

    result = StatisticalResult([], [], [], test_name="pairwise_logrank_test")

    for i1, i2 in combinations(np.arange(n_unique_groups), 2):
        g1, g2 = unique_groups[[i1, i2]]
        ix1, ix2 = (groups == g1), (groups == g2)
        result += logrank_test(
            event_durations.loc[ix1],
            event_durations.loc[ix2],
            event_observed.loc[ix1],
            event_observed.loc[ix2],
            t_0=t_0,
            name=[(g1, g2)],
            weightings=weightings,
            **kwargs
        )

    return result


def difference_of_restricted_mean_survival_time_test(model1, model2, t):
    pass


def multivariate_logrank_test(
    event_durations, groups, event_observed=None, weights=None, t_0=-1, weightings=None, **kwargs
) -> StatisticalResult:  # pylint: disable=too-many-locals
    r"""
    This test is a generalization of the logrank_test: it can deal with n>2 populations (and should
    be equal when n=2):

    .. math::
        \begin{align}
         & H_0: h_1(t) = h_2(t) = h_3(t) = ... = h_n(t) \\
         & H_A: \text{there exist at least one group that differs from the other.}
        \end{align}


    Parameters
    ----------

    event_durations: iterable
        a (n,) list-like representing the (possibly partial) durations of all individuals

    groups: iterable
        a (n,) list-like of unique group labels for each individual.

    event_observed: iterable, optional
        a (n,) list-like of event_observed events: 1 if observed death, 0 if censored. Defaults to all observed.

    weights: iterable, optional
        case-weights

    t_0: float, optional (default=-1)
        The final time period under observation, and subjects who experience the event after this time are set to be censored.
        Specify -1 to use all time.

    weightings: str, optional
        apply a weighted logrank test: options are "wilcoxon" for Wilcoxon (also known as Breslow), "tarone-ware"
        for Tarone-Ware, "peto" for Peto test and "fleming-harrington" for Fleming-Harrington test.
        These are useful for testing for early or late differences in the survival curve. For the Fleming-Harrington
        test, keyword arguments p and q must also be provided with non-negative values.

        Weightings are applied at the ith ordered failure time, :math:`t_{i}`, according to:
            Wilcoxon: :math:`n_i`
            Tarone-Ware: :math:`\sqrt{n_i}`
            Peto: :math:`\bar{S}(t_i)`
            Fleming-Harrington: :math:`\hat{S}(t_i)^p \times (1 - \hat{S}(t_i))^q`

            where :math:`n_i` is the number at risk just prior to time :math:`t_{i}`, :math:`\bar{S}(t_i)` is
            Peto-Peto's modified survival estimate and :math:`\hat{S}(t_i)` is the left-continuous
            Kaplan-Meier survival estimate at time :math:`t_{i}`.

    kwargs:
        add keywords and meta-data to the experiment summary.


    Returns
    -------

    StatisticalResult
       a StatisticalResult object with properties ``p_value``, ``summary``, ``test_statistic``, ``print_summary``

    Examples
    --------

    .. code:: python

        df = pd.DataFrame({
           'durations': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
           'events': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
           'groups': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        })
        result = multivariate_logrank_test(df['durations'], df['groups'], df['events'])
        result.test_statistic
        result.p_value
        result.print_summary()


        # numpy example
        G = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        T = [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7]
        E = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
        result = multivariate_logrank_test(T, G, E)
        result.test_statistic


    See Also
    --------
    pairwise_logrank_test
    logrank_test
    """
    kwargs.setdefault("test_name", "multivariate_logrank_test")

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

    # censor all subjects that are beyond the specified t_0, see #1300
    if int(t_0) != -1:
        event_observed[event_durations > t_0] = 0

    unique_groups, rm, obs, _ = group_survival_table_from_events(groups, event_durations, event_observed, weights=weights)
    n_groups = unique_groups.shape[0]

    # compute the factors needed
    n_ij = rm.sum(0).values - rm.cumsum(0).shift(1).fillna(0)
    d_i = obs.sum(1)
    n_i = rm.values.sum() - rm.sum(1).cumsum().shift(1).fillna(0)
    ev_i = n_ij.mul(d_i / n_i, axis="index")

    # compute weightings for log-rank alternatives
    if weightings is None:
        w_i = np.ones(d_i.shape[0])
    elif weightings == "wilcoxon":
        kwargs["test_name"] = kwargs["test_name"].replace("logrank", "Wilcoxon")
        w_i = n_i
    elif weightings == "tarone-ware":
        kwargs["test_name"] = kwargs["test_name"].replace("logrank", "Tarone-Ware")
        w_i = np.sqrt(n_i)
    elif weightings == "peto":
        kwargs["test_name"] = kwargs["test_name"].replace("logrank", "Peto")
        w_i = np.cumprod(1.0 - (ev_i.sum(1)) / (n_i + 1))  # Peto-Peto's modified survival estimates.
    elif weightings == "fleming-harrington":
        if "p" in kwargs:
            p = kwargs["p"]
            if p < 0:
                raise ValueError("p must be non-negative.")
        else:
            raise ValueError("Must provide keyword argument p for Flemington-Harrington test statistic")
        if "q" in kwargs:
            q = kwargs["q"]
            if q < 0:
                raise ValueError("q must be non-negative.")
        else:
            raise ValueError("Must provide keyword argument q for Flemington-Harrington test statistic")
        kwargs["test_name"] = kwargs["test_name"].replace("logrank", "Flemington-Harrington")
        kmf = KaplanMeierFitter().fit(event_durations, event_observed=event_observed)
        s = kmf.survival_function_.to_numpy().flatten()[:-1]  # Left-continuous Kaplan-Meier survival estimate.
        w_i = np.power(s, p) * np.power(1.0 - s, q)
    else:
        raise ValueError("Invalid value for weightings.")

    # apply weights to observed and expected
    N_j = obs.mul(w_i, axis=0).sum(0).values
    ev = ev_i.mul(w_i, axis=0).sum(0)

    # vector of observed minus expected
    Z_j = N_j - ev

    assert abs(Z_j.sum()) < 10e-8, "Sum is not zero."  # this should move to a test eventually.

    # compute covariance matrix
    factor = (((n_i - d_i) / (n_i - 1)).replace([np.inf, np.nan], 1)) * d_i / n_i**2
    n_ij["_"] = n_i.values
    V_ = (n_ij.mul(w_i, axis=0)).mul(np.sqrt(factor), axis="index").fillna(0)  # weighted V_
    V = -np.dot(V_.T, V_)
    ix = np.arange(n_groups)
    V[ix, ix] = V[ix, ix] - V[-1, ix]
    V = V[:-1, :-1]

    # take the first n-1 groups
    U = Z_j.iloc[:-1] @ np.linalg.pinv(V[:-1, :-1]) @ Z_j.iloc[:-1]  # Z.T*inv(V)*Z

    # compute the p-values and tests
    p_value = _chisq_test_p_value(U, n_groups - 1)
    return StatisticalResult(p_value, U, t_0=t_0, null_distribution="chi squared", degrees_of_freedom=n_groups - 1, **kwargs)


def _chisq_test_p_value(U, degrees_freedom) -> float:
    p_value = stats.chi2.sf(U, degrees_freedom)
    return p_value


class TimeTransformers:

    TIME_TRANSFOMERS = {
        # using cumsum is kinda hacky (but smart), should be np.argsort but I run into problems later.
        "rank": lambda t, c, w: np.cumsum(c),
        "identity": lambda t, c, w: t,
        "log": lambda t, c, w: np.log(t),
        "km": lambda t, c, w: (1 - KaplanMeierFitter().fit(t, c, weights=w).survival_function_.loc[t, "KM_estimate"]),
    }

    def get(self, key_or_callable):
        return self.TIME_TRANSFOMERS.get(key_or_callable, key_or_callable)

    def iter(self, keys):
        for key, item in self.TIME_TRANSFOMERS.items():
            if key in keys:
                yield key, item


def proportional_hazard_test(
    fitted_cox_model, training_df, time_transform="rank", precomputed_residuals=None, **kwargs
) -> StatisticalResult:
    """
    Test whether any variable in a Cox model breaks the proportional hazard assumption. This method uses an approximation
    that R's ``survival`` use to use, but changed it in late 2019, hence there will be differences here between lifelines and R.

    Parameters
    ----------
    fitted_cox_model: CoxPHFitter
        the fitted Cox model, fitted with `training_df`, you wish to test. Currently only the CoxPHFitter is supported,
        but later CoxTimeVaryingFitter, too.
    training_df: DataFrame
        the DataFrame used in the call to the Cox model's ``fit``. Optional if providing ``precomputed_residuals``
    time_transform: vectorized function, list, or string, optional (default='rank')
        {'all', 'km', 'rank', 'identity', 'log'}
        One of the strings above, a list of strings, or a function to transform the time (must accept (time, durations, weights) however). 'all' will present all the transforms.
    precomputed_residuals: DataFrame, optional
        specify the scaled Schoenfeld residuals, if already computed.
    kwargs:
        additional parameters to add to the StatisticalResult

    Notes
    ------
    R uses the default `km`, we use `rank`, as this performs well versus other transforms. See
    http://eprints.lse.ac.uk/84988/1/06_ParkHendry2015-ReassessingSchoenfeldTests_Final.pdf

    References
    -----------
     - http://eprints.lse.ac.uk/84988/1/06_ParkHendry2015-ReassessingSchoenfeldTests_Final.pdf
     - "Extending the Cox Model"
     - https://github.com/therneau/survival/commit/5da455de4f16fbed7f867b1fc5b15f2157a132cd#diff-c784cc3eeb38f0a6227988a30f9c0730R36

    """
    if "transform" in kwargs:
        warnings.warn("Found 'transform' keyword being set. Did you mean to set 'time_transform' instead?", UserWarning)

    events, durations, weights = fitted_cox_model.event_observed, fitted_cox_model.durations, fitted_cox_model.weights
    n_deaths = events.sum()

    if precomputed_residuals is None:
        scaled_resids = fitted_cox_model.compute_residuals(training_df, kind="scaled_schoenfeld")
    else:
        scaled_resids = precomputed_residuals

    def compute_statistic(times, resids, n_deaths):
        demeaned_times = times - times.mean()
        T = (demeaned_times.values[:, None] * resids.values).sum(0) ** 2 / (
            n_deaths * (fitted_cox_model.standard_errors_**2) * (demeaned_times**2).sum()
        )
        return T

    if time_transform == "all":
        time_transform = list(TimeTransformers.TIME_TRANSFOMERS.keys())

    if isinstance(time_transform, list):

        result = StatisticalResult([], [], [])

        # yuck
        for transform_name, transform in ((_, TimeTransformers().get(_)) for _ in time_transform):
            times = transform(durations, events, weights)[events.values]
            T = compute_statistic(times, scaled_resids, n_deaths)
            p_values = utils._to_1d_array([_chisq_test_p_value(t, 1) for t in T])
            result += StatisticalResult(
                p_values,
                T,
                name=[(c, transform_name) for c in fitted_cox_model.params_.index],
                test_name="proportional_hazard_test",
                null_distribution="chi squared",
                degrees_of_freedom=1,
                model=str(fitted_cox_model),
                **kwargs
            )

    else:
        time_transformer = TimeTransformers().get(time_transform)
        assert callable(
            time_transformer
        ), "time_transform must be a callable function, or a string: {'rank', 'km', 'identity', 'log'}."

        times = time_transformer(durations, events, weights)[events.values]

        T = compute_statistic(times, scaled_resids, n_deaths)

        p_values = utils._to_1d_array([_chisq_test_p_value(t, 1) for t in T])
        result = StatisticalResult(
            p_values,
            T,
            name=fitted_cox_model.params_.index.tolist(),
            test_name="proportional_hazard_test",
            time_transform=time_transform,
            null_distribution="chi squared",
            degrees_of_freedom=1,
            model=str(fitted_cox_model),
            **kwargs
        )
    return result
