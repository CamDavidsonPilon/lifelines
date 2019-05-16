# -*- coding: utf-8 -*-

import warnings
import collections
from datetime import datetime
from functools import wraps


import numpy as np
from scipy.linalg import solve
from scipy import stats
import pandas as pd

from lifelines.utils.concordance import concordance_index


__all__ = [
    "qth_survival_times",
    "qth_survival_time",
    "median_survival_times",
    "survival_table_from_events",
    "group_survival_table_from_events",
    "survival_events_from_table",
    "datetimes_to_durations",
    "concordance_index",
    "k_fold_cross_validation",
    "to_long_format",
    "to_episodic_format",
    "add_covariate_to_timeline",
    "covariates_from_event_matrix",
]


class CensoringType:

    LEFT = 1
    INTERVAL = 2
    RIGHT = 3

    @classmethod
    def right_censoring(cls, function):
        @wraps(function)
        def f(self, *args, **kwargs):
            self._censoring_type = cls.RIGHT
            return function(self, *args, **kwargs)

        return f

    @classmethod
    def left_censoring(cls, function):
        @wraps(function)
        def f(self, *args, **kwargs):
            self._censoring_type = cls.LEFT
            return function(self, *args, **kwargs)

        return f

    @classmethod
    def interval_censoring(cls, function):
        @wraps(function)
        def f(self, *args, **kwargs):
            self._censoring_type = cls.INTERVAL
            return function(self, *args, **kwargs)

        return f

    @classmethod
    def is_right_censoring(cls, model):
        return model._censoring_type == cls.RIGHT

    @classmethod
    def is_left_censoring(cls, model):
        return model._censoring_type == cls.LEFT

    @classmethod
    def is_interval_censoring(cls, model):
        return model._censoring_type == cls.INTERVAL


class StatError(Exception):
    pass


class ConvergenceError(ValueError):
    # inherits from ValueError for backwards compatibility reasons
    def __init__(self, msg, original_exception=""):
        super(ConvergenceError, self).__init__(msg + "%s" % original_exception)
        self.original_exception = original_exception


class ConvergenceWarning(RuntimeWarning):
    pass


class StatisticalWarning(RuntimeWarning):
    pass


def qth_survival_times(q, survival_functions, cdf=False):
    """
    Find the times when one or more survival functions reach the qth percentile.

    Parameters
    ----------
    q: float or array
      a float between 0 and 1 that represents the time when the survival function hits the qth percentile.
    survival_functions: a (n,d) DataFrame or numpy array.
      If DataFrame, will return index values (actual times)
      If numpy array, will return indices.
    cdf: boolean, optional
      When doing left-censored data, cdf=True is used.

    Returns
    -------
    float, or DataFrame
         if d==1, returns a float, np.inf if infinity.
         if d > 1, an DataFrame containing the first times the value was crossed.

    See Also
    --------
    qth_survival_time, median_survival_times
    """
    # pylint: disable=cell-var-from-loop,misplaced-comparison-constant,no-else-return

    q = pd.Series(q)

    if not ((q <= 1).all() and (0 <= q).all()):
        raise ValueError("q must be between 0 and 1")

    survival_functions = pd.DataFrame(survival_functions)

    if survival_functions.shape[1] == 1 and q.shape == (1,):
        q = q[0]
        # If you add print statements to `qth_survival_time`, you'll see it's called
        # once too many times. This is expected Pandas behavior
        # https://stackoverflow.com/questions/21635915/why-does-pandas-apply-calculate-twice
        return survival_functions.apply(lambda s: qth_survival_time(q, s, cdf=cdf)).iloc[0]
    else:
        d = {_q: survival_functions.apply(lambda s: qth_survival_time(_q, s, cdf=cdf)) for _q in q}
        survival_times = pd.DataFrame(d).T

        #  Typically, one would expect that the output should equal the "height" of q.
        #  An issue can arise if the Series q contains duplicate values. We solve
        #  this by duplicating the entire row.
        if q.duplicated().any():
            survival_times = survival_times.loc[q]

        return survival_times


def qth_survival_time(q, survival_function, cdf=False):
    """
    Returns the time when a single survival function reaches the qth percentile.

    Parameters
    ----------
    q: float
      a float between 0 and 1 that represents the time when the survival function hit's the qth percentile.
    survival_function: Series or single-column DataFrame.
    cdf: boolean, optional
      When doing left-censored data, cdf=True is used.

    Returns
    -------
    float

    See Also
    --------
    qth_survival_times, median_survival_times
    """
    if type(survival_function) is pd.DataFrame:  # pylint: disable=unidiomatic-typecheck
        if survival_function.shape[1] > 1:
            raise ValueError(
                "Expecting a dataframe (or series) with a single column. Provide that or use utils.qth_survival_times."
            )

        survival_function = survival_function.T.squeeze()
    if cdf:
        if survival_function.iloc[0] > q:
            return -np.inf
        v = survival_function.index[survival_function.searchsorted([q])[0]]
    else:
        if survival_function.iloc[-1] > q:
            return np.inf
        v = survival_function.index[(-survival_function).searchsorted([-q])[0]]
    return v


def median_survival_times(density_or_survival_function, left_censorship=False):
    return qth_survival_times(0.5, density_or_survival_function, cdf=left_censorship)


def group_survival_table_from_events(
    groups, durations, event_observed, birth_times=None, limit=-1
):  # pylint: disable=too-many-locals
    """
    Joins multiple event series together into DataFrames. A generalization of
    `survival_table_from_events` to data with groups. Previously called `group_event_series` pre 0.2.3.

    Parameters
    ----------
    groups: a (n,) array
      individuals' group ids.
    durations: a (n,)  array
      durations of each individual
    event_observed: a (n,) array
      event observations, 1 if observed, 0 else.
    birth_times: a (n,) array
      when the subject was first observed. A subject's death event is then at [birth times + duration observed].
      Normally set to all zeros, but can be positive or negative.
    limit:

    Returns
    -------
    unique_groups: np.array
      array of all the unique groups present
    removed: DataFrame
      DataFrame of removal count data at event_times for each group, column names are 'removed:<group name>'
    observed: DataFrame
      DataFrame of observed count data at event_times for each group, column names are 'observed:<group name>'
    censored: DataFrame
      DataFrame of censored count data at event_times for each group, column names are 'censored:<group name>'

    Example
    -------
    >>> #input
    >>> group_survival_table_from_events(waltonG, waltonT, np.ones_like(waltonT)) #data available in test_suite.py
    >>> #output
    >>> [
    >>>     array(['control', 'miR-137'], dtype=object),
    >>>               removed:control  removed:miR-137
    >>>     event_at
    >>>     6                       0                1
    >>>     7                       2                0
    >>>     9                       0                3
    >>>     13                      0                3
    >>>     15                      0                2
    >>>     ,
    >>>               observed:control  observed:miR-137
    >>>     event_at
    >>>     6                        0                 1
    >>>     7                        2                 0
    >>>     9                        0                 3
    >>>     13                       0                 3
    >>>     15                       0                 2
    >>>     ,
    >>>               censored:control  censored:miR-137
    >>>     event_at
    >>>     6                        0                 0
    >>>     7                        0                 0
    >>>     9                        0                 0
    >>>     ,
    >>> ]

    See Also
    --------
    survival_table_from_events

    """

    n = np.max(groups.shape)
    assert n == np.max(durations.shape) == np.max(event_observed.shape), "inputs must be of the same length."

    if birth_times is None:
        # Create some birth times
        birth_times = np.zeros(np.max(durations.shape))
        birth_times[:] = np.min(durations)

    assert n == np.max(birth_times.shape), "inputs must be of the same length."

    groups, durations, event_observed, birth_times = [
        pd.Series(np.asarray(vector).reshape(n)) for vector in [groups, durations, event_observed, birth_times]
    ]
    unique_groups = groups.unique()

    for i, group in enumerate(unique_groups):
        ix = groups == group
        T = durations[ix]
        C = event_observed[ix]
        B = birth_times[ix]
        group_name = str(group)
        columns = [
            event_name + ":" + group_name for event_name in ["removed", "observed", "censored", "entrance", "at_risk"]
        ]
        if i == 0:
            survival_table = survival_table_from_events(T, C, B, columns=columns)
        else:
            survival_table = survival_table.join(survival_table_from_events(T, C, B, columns=columns), how="outer")

    survival_table = survival_table.fillna(0)
    # hmmm pandas its too bad I can't do data.loc[:limit] and leave out the if.
    if int(limit) != -1:
        survival_table = survival_table.loc[:limit]

    return (
        unique_groups,
        survival_table.filter(like="removed:"),
        survival_table.filter(like="observed:"),
        survival_table.filter(like="censored:"),
    )


def survival_table_from_events(
    death_times,
    event_observed,
    birth_times=None,
    columns=["removed", "observed", "censored", "entrance", "at_risk"],
    weights=None,
    collapse=False,
    intervals=None,
):  # pylint: disable=dangerous-default-value,too-many-locals
    """
    Parameters
    ----------
    death_times: (n,) array
      represent the event times
    event_observed: (n,) array
      1 if observed event, 0 is censored event.
    birth_times: a (n,) array, optional
      representing when the subject was first observed. A subject's death event is then at [birth times + duration observed].
      If None (default), birth_times are set to be the first observation or 0, which ever is smaller.
    columns: iterable, optional
      a 3-length array to call the, in order, removed individuals, observed deaths
      and censorships.
    weights: (n,1) array, optional
      Optional argument to use weights for individuals. Assumes weights of 1 if not provided.
    collapse: boolean, optional (default=False)
      If True, collapses survival table into lifetable to show events in interval bins
    intervals: iterable, optional
      Default None, otherwise a list/(n,1) array of interval edge measures. If left as None
      while collapse=True, then Freedman-Diaconis rule for histogram bins will be used to determine intervals.

    Returns
    -------
    DataFrame
      Pandas DataFrame with index as the unique times or intervals in event_times. The columns named
      'removed' refers to the number of individuals who were removed from the population
      by the end of the period. The column 'observed' refers to the number of removed
      individuals who were observed to have died (i.e. not censored.) The column
      'censored' is defined as 'removed' - 'observed' (the number of individuals who
      left the population due to event_observed)

    Example
    -------

    >>> #Uncollapsed output
    >>>           removed  observed  censored  entrance   at_risk
    >>> event_at
    >>> 0               0         0         0        11        11
    >>> 6               1         1         0         0        11
    >>> 7               2         2         0         0        10
    >>> 9               3         3         0         0         8
    >>> 13              3         3         0         0         5
    >>> 15              2         2         0         0         2
    >>> #Collapsed output
    >>>          removed observed censored at_risk
    >>>              sum      sum      sum     max
    >>> event_at
    >>> (0, 2]        34       33        1     312
    >>> (2, 4]        84       42       42     278
    >>> (4, 6]        64       17       47     194
    >>> (6, 8]        63       16       47     130
    >>> (8, 10]       35       12       23      67
    >>> (10, 12]      24        5       19      32

    See Also
    --------
    group_survival_table_from_events
    """
    removed, observed, censored, entrance, at_risk = columns
    death_times = np.asarray(death_times)
    if birth_times is None:
        birth_times = min(0, death_times.min()) * np.ones(death_times.shape[0])
    else:
        birth_times = np.asarray(birth_times)
        if np.any(birth_times > death_times):
            raise ValueError("birth time must be less than time of death.")

    if weights is None:
        weights = 1

    # deal with deaths and censorships
    df = pd.DataFrame(death_times, columns=["event_at"])
    df[removed] = np.asarray(weights)
    df[observed] = np.asarray(weights) * (np.asarray(event_observed).astype(bool))
    death_table = df.groupby("event_at").sum()
    death_table[censored] = (death_table[removed] - death_table[observed]).astype(int)

    # deal with late births
    births = pd.DataFrame(birth_times, columns=["event_at"])
    births[entrance] = np.asarray(weights)
    births_table = births.groupby("event_at").sum()
    event_table = death_table.join(births_table, how="outer", sort=True).fillna(0)  # http://wesmckinney.com/blog/?p=414
    event_table[at_risk] = event_table[entrance].cumsum() - event_table[removed].cumsum().shift(1).fillna(0)

    # group by intervals
    if (collapse) or (intervals is not None):
        event_table = _group_event_table_by_intervals(event_table, intervals)

    if (np.asarray(weights).astype(int) != weights).any():
        return event_table.astype(float)
    return event_table.astype(int)


def _group_event_table_by_intervals(event_table, intervals):
    event_table = event_table.reset_index()

    # use Freedman-Diaconis rule to determine bin size if user doesn't define intervals
    if intervals is None:
        event_max = event_table["event_at"].max()

        # need interquartile range for bin width
        q75, q25 = np.percentile(event_table["event_at"], [75, 25])
        event_iqr = q75 - q25

        bin_width = 2 * event_iqr * (len(event_table["event_at"]) ** (-1 / 3))

        intervals = np.arange(0, event_max + bin_width, bin_width)

    event_table = event_table.groupby(pd.cut(event_table["event_at"], intervals)).agg(
        {"removed": ["sum"], "observed": ["sum"], "censored": ["sum"], "at_risk": ["max"]}
    )
    # convert columns from multiindex
    event_table.columns = event_table.columns.droplevel(1)
    return event_table


def survival_events_from_table(survival_table, observed_deaths_col="observed", censored_col="censored"):
    """
    This is the inverse of the function ``survival_table_from_events``.

    Parameters
    ----------
    survival_table: DataFrame
        a pandas DataFrame with index as the durations and columns "observed" and "censored", referring to
           the number of individuals that died and were censored at time t.
    observed_deaths_col: str, optional (default: "observed")
        the column in the survival table that represents the number of subjects that were observed to die at a specific time
    censored_col: str,  optional (default: "censored")
        the column in the survival table that represents the number of subjects that were censored at a specific time

    Returns
    -------
    T: array
      durations of observation -- one element for observed time
    E: array
      event observations -- 1 if observed, 0 else.
    W: array
      weights - integer weights to "condense" the data

    Example
    -------
    >>> # Ex: The survival table, as a pandas DataFrame:
    >>>
    >>>                  observed  censored
    >>>    index
    >>>    1                1         0
    >>>    2                0         1
    >>>    3                1         0
    >>>    4                1         1
    >>>    5                0         1
    >>>
    >>> # would return
    >>> T = np.array([ 1.,  2.,  3.,  4.,  4.,  5.]),
    >>> E = np.array([ 1.,  0.,  1.,  1.,  0.,  0.])
    >>> W = np.array([ 1,  1,  1,  1,  1,  1])

    """
    T_ = []
    E_ = []
    W_ = []

    for t, row in survival_table.iterrows():
        if row[observed_deaths_col] > 0:
            T_.append(t)
            E_.append(1)
            W_.append(row[observed_deaths_col])
        if row[censored_col] > 0:
            T_.append(t)
            E_.append(0)
            W_.append(row[censored_col])

    return np.asarray(T_), np.asarray(E_), np.asarray(W_)


def datetimes_to_durations(
    start_times, end_times, fill_date=datetime.today(), freq="D", dayfirst=False, na_values=None
):
    """
    This is a very flexible function for transforming arrays of start_times and end_times
    to the proper format for lifelines: duration and event observation arrays.

    Parameters
    ----------
    start_times: an array, Series or DataFrame
        iterable representing start times. These can be strings, or datetime objects.
    end_times: an array, Series or DataFrame
        iterable representing end times. These can be strings, or datetimes. These values can be None, or an empty string, which corresponds to censorship.
    fill_date: datetime, optional (default=datetime.Today())
        the date to use if end_times is a None or empty string. This corresponds to last date
        of observation. Anything after this date is also censored.
    freq: string, optional (default='D')
        the units of time to use.  See Pandas 'freq'. Default 'D' for days.
    dayfirst: boolean, optional (default=False)
         convert assuming European-style dates, i.e. day/month/year.
    na_values : list, optional
        list of values to recognize as NA/NaN. Ex: ['', 'NaT']

    Returns
    -------
    T: numpy array
        array of floats representing the durations with time units given by freq.
    C: numpy array
        boolean array of event observations: 1 if death observed, 0 else.

    Examples
    --------
    >>> from lifelines.utils import datetimes_to_durations
    >>>
    >>> start_dates = ['2015-01-01', '2015-04-01', '2014-04-05']
    >>> end_dates = ['2016-02-02', None, '2014-05-06']
    >>>
    >>> T, E = datetimes_to_durations(start_dates, end_dates, freq="D")
    >>> T # array([ 397., 1414.,   31.])
    >>> E # array([ True, False,  True])

    """
    fill_date = pd.to_datetime(fill_date)
    freq_string = "timedelta64[%s]" % freq
    start_times = pd.Series(start_times).copy()
    end_times = pd.Series(end_times).copy()

    C = ~(pd.isnull(end_times).values | end_times.isin(na_values or [""]))
    end_times[~C] = fill_date
    start_times_ = pd.to_datetime(start_times, dayfirst=dayfirst)
    end_times_ = pd.to_datetime(end_times, dayfirst=dayfirst, errors="coerce")

    deaths_after_cutoff = end_times_ > fill_date
    C[deaths_after_cutoff] = False

    T = (end_times_ - start_times_).values.astype(freq_string).astype(float)
    if (T < 0).sum():
        warnings.warn("Warning: some values of start_times are after end_times")
    return T, C.values


def l1_log_loss(event_times, predicted_event_times, event_observed=None):
    r"""
    Calculates the l1 log-loss of predicted event times to true event times for *non-censored*
    individuals only.

    .. math::  1/N \sum_{i} |log(t_i) - log(q_i)|

    Parameters
    ----------
      event_times: a (n,) array of observed survival times.
      predicted_event_times: a (n,) array of predicted survival times.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.

    Returns
    -------
      l1-log-loss: a scalar
    """
    if event_observed is None:
        event_observed = np.ones_like(event_times)

    ix = event_observed.astype(bool)
    return np.abs(np.log(event_times[ix]) - np.log(predicted_event_times[ix])).mean()


def l2_log_loss(event_times, predicted_event_times, event_observed=None):
    r"""
    Calculates the l2 log-loss of predicted event times to true event times for *non-censored*
    individuals only.

    .. math::  1/N \sum_{i} (log(t_i) - log(q_i))**2

    Parameters
    ----------
      event_times: a (n,) array of observed survival times.
      predicted_event_times: a (n,) array of predicted survival times.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.

    Returns
    -------
      l2-log-loss: a scalar
    """
    if event_observed is None:
        event_observed = np.ones_like(event_times)

    ix = event_observed.astype(bool)
    return np.power(np.log(event_times[ix]) - np.log(predicted_event_times[ix]), 2).mean()


def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def inv_normal_cdf(p):
    return stats.norm.ppf(p)


def k_fold_cross_validation(
    fitters,
    df,
    duration_col,
    event_col=None,
    k=5,
    evaluation_measure=concordance_index,
    predictor="predict_expectation",
    predictor_kwargs={},
    fitter_kwargs={},
):  # pylint: disable=dangerous-default-value,too-many-arguments,too-many-locals
    """
    Perform cross validation on a dataset. If multiple models are provided,
    all models will train on each of the k subsets.

    Parameters
    ----------
    fitters: model
      one or several objects which possess a method: ``fit(self, data, duration_col, event_col)``
      Note that the last two arguments will be given as keyword arguments,
      and that event_col is optional. The objects must also have
      the "predictor" method defined below.
    df: DataFrame
      a Pandas DataFrame with necessary columns `duration_col` and (optional) `event_col`, plus
      other covariates. `duration_col` refers to the lifetimes of the subjects. `event_col`
      refers to whether the 'death' events was observed: 1 if observed, 0 else (censored).
    duration_col: (n,) array
      the column in DataFrame that contains the subjects lifetimes.
    event_col: (n,) array
      the column in DataFrame that contains the subject's death observation. If left
      as None, assumes all individuals are non-censored.
    k: int
      the number of folds to perform. n/k data will be withheld for testing on.
    evaluation_measure: function
      a function that accepts either (event_times, predicted_event_times),
      or (event_times, predicted_event_times, event_observed)
      and returns something (could be anything).
      Default: statistics.concordance_index: (C-index)
      between two series of event times
    predictor: string
      a string that matches a prediction method on the fitter instances.
      For example, ``predict_expectation`` or ``predict_percentile``.
      Default is "predict_expectation"
      The interface for the method is: ``predict(self, data, **optional_kwargs)``
    fitter_kwargs:
      keyword args to pass into fitter.fit method
    predictor_kwargs:
      keyword args to pass into predictor-method.

    Returns
    -------
    results: list
      (k,1) list of scores for each fold. The scores can be anything.
    """
    # Make sure fitters is a list
    try:
        fitters = list(fitters)
    except TypeError:
        fitters = [fitters]
    # Each fitter has its own scores
    fitterscores = [[] for _ in fitters]

    n, _ = df.shape
    df = df.copy()

    if event_col is None:
        event_col = "E"
        df[event_col] = 1.0

    df = df.reindex(np.random.permutation(df.index)).sort_values(event_col)

    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]

    testing_columns = df.columns.drop([duration_col, event_col])

    for i in range(1, k + 1):

        ix = assignments == i
        training_data = df.loc[~ix]
        testing_data = df.loc[ix]

        T_actual = testing_data[duration_col].values
        E_actual = testing_data[event_col].values
        X_testing = testing_data[testing_columns]

        for fitter, scores in zip(fitters, fitterscores):
            # fit the fitter to the training data
            fitter.fit(training_data, duration_col=duration_col, event_col=event_col, **fitter_kwargs)
            T_pred = getattr(fitter, predictor)(X_testing, **predictor_kwargs).values

            try:
                scores.append(evaluation_measure(T_actual, T_pred, E_actual))
            except TypeError:
                scores.append(evaluation_measure(T_actual, T_pred))

    # If a single fitter was given as argument, return a single result
    if len(fitters) == 1:
        return fitterscores[0]
    return fitterscores


def normalize(X, mean=None, std=None):
    """
    Normalize X. If mean OR std is None, normalizes
    X to have mean 0 and std 1.
    """
    if mean is None or std is None:
        mean = X.mean(0)
        std = X.std(0)
    return (X - mean) / std


def unnormalize(X, mean, std):
    """
    Reverse a normalization. Requires the original mean and
    standard deviation of the data set.
    """
    return X * std + mean


def epanechnikov_kernel(t, T, bandwidth=1.0):
    M = 0.75 * (1 - ((t - T) / bandwidth) ** 2)
    M[abs((t - T)) >= bandwidth] = 0
    return M


def ridge_regression(X, Y, c1=0.0, c2=0.0, offset=None, ix=None):
    """
    Also known as Tikhonov regularization. This solves the minimization problem:

    min_{beta} ||(beta X - Y)||^2 + c1||beta||^2 + c2||beta - offset||^2

    One can find more information here: http://en.wikipedia.org/wiki/Tikhonov_regularization

    Parameters
    ----------
    X: a (n,d) numpy array
    Y: a (n,) numpy array
    c1: float
    c2: float
    offset: a (d,) numpy array.
    ix: a boolean array of index to slice.

    Returns
    -------
    beta_hat: numpy array
      the solution to the minimization problem. V = (X*X^T + (c1+c2)I)^{-1} X^T
    """
    _, d = X.shape

    if c1 > 0 or c2 > 0:
        penalizer_matrix = (c1 + c2) * np.eye(d)
        A = np.dot(X.T, X) + penalizer_matrix
    else:
        A = np.dot(X.T, X)

    if offset is None or c2 == 0:
        b = np.dot(X.T, Y)
    else:
        b = np.dot(X.T, Y) + c2 * offset

    if ix is not None:
        M = np.c_[X.T[:, ix], b]
    else:
        M = np.c_[X.T, b]
    R = solve(A, M, assume_a="pos", check_finite=False)
    return R[:, -1], R[:, :-1]


def _additive_estimate(events, timeline, _additive_f, _additive_var, reverse):
    """
    Called to compute the Kaplan Meier and Nelson-Aalen estimates.

    """
    if reverse:
        events = events.sort_index(ascending=False)
        at_risk = events["entrance"].sum() - events["removed"].cumsum().shift(1).fillna(0)

        deaths = events["observed"]

        estimate_ = np.cumsum(_additive_f(at_risk, deaths)).sort_index().shift(-1).fillna(0)
        var_ = np.cumsum(_additive_var(at_risk, deaths)).sort_index().shift(-1).fillna(0)
    else:
        deaths = events["observed"]

        # Why subtract entrants like this? see https://github.com/CamDavidsonPilon/lifelines/issues/497
        # specifically, we kill people, compute the ratio, and then "add" the entrants. This means that
        # the population should not have the late entrants. The only exception to this rule
        # is the first period, where entrants happen _prior_ to deaths.
        entrances = events["entrance"].copy()
        entrances.iloc[0] = 0
        population = events["at_risk"] - entrances

        estimate_ = np.cumsum(_additive_f(population, deaths))
        var_ = np.cumsum(_additive_var(population, deaths))

    timeline = sorted(timeline)
    estimate_ = estimate_.reindex(timeline, method="pad").fillna(0)
    var_ = var_.reindex(timeline, method="pad")
    var_.index.name = "timeline"
    estimate_.index.name = "timeline"

    return estimate_, var_


def _preprocess_inputs(durations, event_observed, timeline, entry, weights):
    """
    Cleans and confirms input to what lifelines expects downstream
    """

    n = len(durations)
    durations = np.asarray(pass_for_numeric_dtypes_or_raise_array(durations)).reshape((n,))

    # set to all observed if event_observed is none
    if event_observed is None:
        event_observed = np.ones(n, dtype=int)
    else:
        event_observed = np.asarray(event_observed).reshape((n,)).copy().astype(int)

    if entry is not None:
        entry = np.asarray(entry).reshape((n,))

    event_table = survival_table_from_events(durations, event_observed, entry, weights=weights)
    if timeline is None:
        timeline = event_table.index.values
    else:
        timeline = np.asarray(timeline)

    return (durations, event_observed, timeline.astype(float), entry, event_table)


def _get_index(X):
    # we need a unique index because these are about to become column names.
    if isinstance(X, pd.DataFrame) and X.index.is_unique:
        index = list(X.index)
    else:
        # If it's not a dataframe, order is up to user
        index = list(range(X.shape[0]))
    return index


def pass_for_numeric_dtypes_or_raise_array(x):
    """
    Use the utility `to_numeric` to check that x is convertible to numeric values, and then convert. Any errors
    are reported back to the user.

    Parameters
    ----------
    x: list, array, Series, DataFrame

    Notes
    ------
    This actually allows objects like timedeltas (converted to microseconds), and strings as numbers.

    """
    try:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            v = pd.to_numeric(x.squeeze())
        else:
            v = pd.to_numeric(np.asarray(x).squeeze())

        if v.size == 0:
            raise ValueError("Empty array/Series passed in.")
        return v

    except:
        raise ValueError("Values must be numeric: no strings, datetimes, objects, etc.")


def check_for_numeric_dtypes_or_raise(df):
    nonnumeric_cols = [
        col for (col, dtype) in df.dtypes.iteritems() if dtype.name == "category" or dtype.kind not in "biuf"
    ]
    if len(nonnumeric_cols) > 0:  # pylint: disable=len-as-condition
        raise TypeError(
            "DataFrame contains nonnumeric columns: %s. Try 1) using pandas.get_dummies to convert the non-numeric column(s) to numerical data, 2) using it in stratification `strata=`, or 3) dropping the column(s)."
            % nonnumeric_cols
        )


def check_for_immediate_deaths(events, start, stop):
    # Only used in CTV. This checks for deaths immediately, that is (0,0) lives.
    if ((start == stop) & (stop == 0) & events).any():
        raise ValueError(
            """The dataset provided has subjects that die on the day of entry. (0, 0)
is not allowed in CoxTimeVaryingFitter. If suffices to add a small non-zero value to their end - example Pandas code:

> df.loc[ (df[start_col] == df[stop_col]) & (df[start_col] == 0) & df[event_col], stop_col] = 0.5

Alternatively, add 1 to every subjects' final end period.
"""
        )


def check_for_instantaneous_events(start, stop):
    if ((start == stop) & (stop == 0)).any():
        warning_text = """There exist rows in your dataframe with start and stop both at time 0:

        > df.loc[(df[start_col] == df[stop_col]) & (df[start_col] == 0)]

        These can be safely dropped, which will improve performance.

        > df = df.loc[~((df[start_col] == df[stop_col]) & (df[start_col] == 0))]
"""
        warnings.warn(warning_text, RuntimeWarning)


def check_for_overlapping_intervals(df):
    # only useful for time varying coefs, after we've done
    # some index creation
    # so slow.
    if not df.groupby(level=1).apply(lambda g: g.index.get_level_values(0).is_non_overlapping_monotonic).all():
        raise ValueError(
            "The dataset provided contains overlapping intervals. Check the start and stop col by id carefully. Try using this code snippet\
to help find:\
df.groupby(level=1).apply(lambda g: g.index.get_level_values(0).is_non_overlapping_monotonic)"
        )


def check_positivity(array):
    if np.any(array <= 0):
        raise ValueError(
            "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
        )


def _low_var(df):
    return df.var(0) < 1e-4


def check_low_var(df, prescript="", postscript=""):
    low_var = _low_var(df)
    if low_var.any():
        cols = str(list(df.columns[low_var]))
        warning_text = (
            "%sColumn(s) %s have very low variance. \
This may harm convergence. Try dropping this redundant column before fitting \
if convergence fails.%s"
            % (prescript, cols, postscript)
        )
        warnings.warn(warning_text, ConvergenceWarning)


def check_complete_separation_low_variance(df, events, event_col):

    events = events.astype(bool)
    deaths_only = df.columns[_low_var(df.loc[events])]
    censors_only = df.columns[_low_var(df.loc[~events])]
    total = df.columns[_low_var(df)]
    problem_columns = censors_only.union(deaths_only).difference(total).tolist()
    if problem_columns:
        warning_text = """Column {cols} have very low variance when conditioned on death event present or not. This may harm convergence. This could be a form of 'complete separation'. For example, try the following code:

>>> events = df['{event_col}'].astype(bool)
>>> print(df.loc[events, '{cols}'].var())
>>> print(df.loc[~events, '{cols}'].var())

A very low variance means that the column {cols} completely determines whether a subject dies or not. See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression """.format(
            cols=problem_columns[0], event_col=event_col
        )
        warnings.warn(warning_text, ConvergenceWarning)


def correlation(x, y):
    return np.corrcoef(x, y)[1, 0]


def check_complete_separation_close_to_perfect_correlation(df, durations):
    # slow for many columns
    THRESHOLD = 0.99
    n, _ = df.shape

    if n > 500:
        # let's sample to speed this up.
        df = df.sample(n=500, random_state=0).copy()
        durations = pd.Series(durations).sample(n=500, random_state=0).copy()

    rank_durations = durations.argsort()
    for col, series in df.iteritems():
        with np.errstate(invalid="ignore", divide="ignore"):
            rank_series = series.values.argsort()
            if abs(correlation(rank_durations, rank_series)) >= THRESHOLD:
                warning_text = (
                    "Column %s has high sample correlation with the duration column. This may harm convergence. This could be a form of 'complete separation'. \
    See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression"
                    % (col)
                )
                warnings.warn(warning_text, ConvergenceWarning)


def check_complete_separation(df, events, durations, event_col):
    check_complete_separation_low_variance(df, events, event_col)
    check_complete_separation_close_to_perfect_correlation(df, durations)


def check_nans_or_infs(df_or_array):
    if isinstance(df_or_array, (pd.Series, pd.DataFrame)):
        return check_nans_or_infs(df_or_array.values)

    if pd.isnull(df_or_array).any():
        raise TypeError("NaNs were detected in the dataset. Try using pd.isnull to find the problematic values.")

    try:
        infs = np.isinf(df_or_array)
    except TypeError:
        warning_text = (
            """Attempting to convert an unexpected datatype '%s' to float. Suggestion: 1) use `lifelines.utils.datetime_to_durations` to do conversions or 2) manually convert to floats/booleans."""
            % df_or_array.dtype
        )
        warnings.warn(warning_text)
        try:
            infs = np.isinf(df_or_array.astype(float))
        except:
            raise TypeError("Wrong dtype '%s'." % df_or_array.dtype)

    if infs.any():
        raise TypeError("Infs were detected in the dataset. Try using np.isinf to find the problematic values.")


def to_episodic_format(df, duration_col, event_col, id_col=None, time_gaps=1):
    """
    This function takes a "flat" dataset (that is, non-time-varying), and converts it into a time-varying dataset
    with static variables.

    Useful if your dataset has variables that do not satisfy the proportional hazard assumption, and you need to create a
    time-varying dataset to include interaction terms with time.


    Parameters
    ----------
    df: DataFrame
        a DataFrame of the static dataset.
    duration_col: string
        string representing the column in df that represents the durations of each subject.
    event_col: string
        string representing the column in df that represents whether the subject experienced the event or not.
    id_col: string, optional
        Specify the column that represents an id, else lifelines creates an auto-incrementing one.
    time_gaps: float or int
        Specify a desired time_gap. For example, if time_gap is 2 and a subject lives for 10.5 units of time,
        then the final long form will have 5 + 1 rows for that subject: (0, 2], (2, 4], (4, 6], (6, 8], (8, 10], (10, 10.5]
        Smaller time_gaps will produce larger DataFrames, and larger time_gaps will produce smaller DataFrames. In the limit,
        the long DataFrame will be identical to the original DataFrame.

    Returns
    --------
    DataFrame

    Example
    --------
    >>> from lifelines.datasets import load_rossi
    >>> from lifelines.utils import to_episodic_format
    >>> rossi = load_rossi()
    >>> long_rossi = to_episodic_format(rossi, 'week', 'arrest', time_gaps=2.)
    >>>
    >>> from lifelines import CoxTimeVaryingFitter
    >>> ctv = CoxTimeVaryingFitter()
    >>> # age variable violates proportional hazard
    >>> long_rossi['time * age'] = long_rossi['stop'] * long_rossi['age']
    >>> ctv.fit(long_rossi, id_col='id', event_col='arrest', show_progress=True)
    >>> ctv.print_summary()

    See Also
    --------
    add_covariate_to_timeline
    to_long_format

    """
    df = df.copy()
    df[duration_col] /= time_gaps
    df = to_long_format(df, duration_col)

    stop_col = "stop"
    start_col = "start"

    _, d = df.shape

    if id_col is None:
        id_col = "id"
        df.index.rename(id_col, inplace=True)
        df = df.reset_index()
        d_dftv = d + 1
    else:
        d_dftv = d

    # what dtype can I make it?
    dtype_dftv = object if (df.dtypes == object).any() else float

    # how many rows/cols do I need?
    n_dftv = int(np.ceil(df[stop_col]).sum())

    # alocate temporary numpy array to insert into
    tv_array = np.empty((n_dftv, d_dftv), dtype=dtype_dftv)

    special_columns = [stop_col, start_col, event_col]
    non_special_columns = df.columns.difference(special_columns).tolist()

    order_I_want = special_columns + non_special_columns

    df = df[order_I_want]

    position_counter = 0

    for _, row in df.iterrows():
        T, E = row[stop_col], row[event_col]
        T_int = int(np.ceil(T))
        values = np.tile(row.values, (T_int, 1))

        # modify first column, which is the old duration col.
        values[:, 0] = np.arange(1, T + 1, dtype=float)
        values[-1, 0] = T

        # modify second column.
        values[:, 1] = np.arange(0, T, dtype=float)

        # modify third column, which is the old event col
        values[:, 2] = 0.0
        values[-1, 2] = float(E)

        tv_array[position_counter : position_counter + T_int, :] = values

        position_counter += T_int

    dftv = pd.DataFrame(tv_array, columns=df.columns)
    dftv = dftv.astype(dtype=df.dtypes[non_special_columns + [event_col]].to_dict())
    dftv[start_col] *= time_gaps
    dftv[stop_col] *= time_gaps
    return dftv


def to_long_format(df, duration_col):
    """
    This function converts a survival analysis DataFrame to a lifelines "long" format. The lifelines "long"
    format is used in a common next function, ``add_covariate_to_timeline``.

    Parameters
    ----------
    df: DataFrame
        a DataFrame in the standard survival analysis form (one for per observation, with covariates, duration and event flag)
    duration_col: string
        string representing the column in df that represents the durations of each subject.

    Returns
    -------
    long_form_df: DataFrame
        A DataFrame with new columns. This can be fed into `add_covariate_to_timeline`

    See Also
    --------
    to_episodic_format
    add_covariate_to_timeline
    """
    return df.assign(start=0, stop=lambda s: s[duration_col]).drop(duration_col, axis=1)


def add_covariate_to_timeline(
    long_form_df,
    cv,
    id_col,
    duration_col,
    event_col,
    start_col="start",
    stop_col="stop",
    add_enum=False,
    overwrite=True,
    cumulative_sum=False,
    cumulative_sum_prefix="cumsum_",
    delay=0,
):  # pylint: disable=too-many-arguments
    """
    This is a util function to help create a long form table tracking subjects' covariate changes over time. It is meant
    to be used iteratively as one adds more and more covariates to track over time. Before using this function, it is recommended
    to view the documentation at https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#dataset-creation-for-time-varying-regression.

    Parameters
    ----------
    long_form_df: DataFrame
        a DataFrame that has the initial or intermediate "long" form of time-varying observations. Must contain
        columns id_col, 'start', 'stop', and event_col. See function `to_long_format` to transform data into long form.
    cv: DataFrame
        a DataFrame that contains (possibly more than) one covariate to track over time. Must contain columns
        id_col and duration_col. duration_col represents time since the start of the subject's life.
    id_col: string
        the column in long_form_df and cv representing a unique identifier for subjects.
    duration_col: string
        the column in cv that represents the time-since-birth the observation occurred at.
    event_col: string
        the column in df that represents if the event-of-interest occurred
    add_enum: boolean, optional
         a Boolean flag to denote whether to add a column enumerating rows per subject. Useful to specify a specific
        observation, ex: df[df['enum'] == 1] will grab the first observations per subject.
    overwrite: boolean, optional
        if True, covariate values in long_form_df will be overwritten by covariate values in cv if the column exists in both
        cv and long_form_df and the timestamps are identical. If False, the default behavior will be to sum
        the values together.
    cumulative_sum: boolean, optional
        sum over time the new covariates. Makes sense if the covariates are new additions, and not state changes (ex:
        administering more drugs vs taking a temperature.)
    cumulative_sum_prefix: string, optional
        a prefix to add to calculated cumulative sum columns
    delay: int, optional
        add a delay to covariates (useful for checking for reverse causality in analysis)

    Returns
    -------
    long_form_df: DataFrame
        A DataFrame with updated rows to reflect the novel times slices (if any) being added from cv, and novel (or updated) columns
        of new covariates from cv

    See Also
    --------
    to_episodic_format
    to_long_format
    covariates_from_event_matrix
    """

    def remove_redundant_rows(cv):
        """
        Removes rows where no change occurs. Ex:

        cv = pd.DataFrame.from_records([
            {'id': 1, 't': 0, 'var3': 0, 'var4': 1},
            {'id': 1, 't': 1, 'var3': 0, 'var4': 1},  # redundant, as nothing changed during the interval
            {'id': 1, 't': 6, 'var3': 1, 'var4': 1},
        ])

        If cumulative_sum, then redundant rows are not redundant.
        """
        if cumulative_sum:
            return cv
        cols = cv.columns.difference([duration_col])
        cv = cv.loc[(cv[cols].shift() != cv[cols]).any(axis=1)]
        return cv

    def transform_cv_to_long_format(cv):
        return cv.rename(columns={duration_col: start_col})

    def expand(df, cvs):
        id_ = df.name
        try:
            cv = cvs.get_group(id_)
        except KeyError:
            return df

        final_state = bool(df[event_col].iloc[-1])
        final_stop_time = df[stop_col].iloc[-1]
        df = df.drop([id_col, event_col, stop_col], axis=1).set_index(start_col)
        cv = cv.drop([id_col], axis=1).set_index(start_col).loc[:final_stop_time]

        if cumulative_sum:
            cv = cv.cumsum()
            cv = cv.add_prefix(cumulative_sum_prefix)

        # How do I want to merge existing columns at the same time - could be
        # new observations (update) or new treatment applied (sum).
        # There may be more options in the future.
        if not overwrite:
            expanded_df = cv.combine(df, lambda s1, s2: s1 + s2, fill_value=0, overwrite=False)
        elif overwrite:
            expanded_df = cv.combine_first(df)

        n = expanded_df.shape[0]
        expanded_df = expanded_df.reset_index()
        expanded_df[stop_col] = expanded_df[start_col].shift(-1)
        expanded_df[id_col] = id_
        expanded_df[event_col] = False
        expanded_df.at[n - 1, event_col] = final_state
        expanded_df.at[n - 1, stop_col] = final_stop_time

        if add_enum:
            expanded_df["enum"] = np.arange(1, n + 1)

        if cumulative_sum:
            expanded_df[cv.columns] = expanded_df[cv.columns].ffill().fillna(0)

        return expanded_df.ffill()

    if delay < 0:
        raise ValueError("delay parameter must be equal to or greater than 0")

    if any(col not in long_form_df for col in (id_col, event_col, start_col, stop_col)):
        raise IndexError("Missing column in long_form_df")

    cv[duration_col] += delay
    cv = cv.dropna()
    cv = cv.sort_values([id_col, duration_col])
    cvs = cv.pipe(remove_redundant_rows).pipe(transform_cv_to_long_format).groupby(id_col)

    long_form_df = long_form_df.groupby(id_col, group_keys=False).apply(expand, cvs=cvs)
    return long_form_df.reset_index(drop=True)


def covariates_from_event_matrix(df, id_col):
    """
    This is a helper function to handle binary event datastreams in a specific format and convert
    it to a format that add_covariate_to_timeline will accept. For example, suppose you have a
    dataset that looks like:

    .. code:: python

           id  promotion  movement  raise
        0   1        1.0       NaN    2.0
        1   2        NaN       5.0    NaN
        2   3        3.0       5.0    7.0


    where the values (aside from the id column) represent when an event occurred for a specific user, relative
    to the subject's birth/entry. This is a common way format to pull data from a SQL table. We call this a duration matrix, and we
    want to convert this DataFrame to a format that can be included in a long form DataFrame
    (see add_covariate_to_timeline for more details on this).

    The duration matrix should have 1 row per subject (but not necessarily all subjects).

    Parameters
    ----------
    df: DataFrame
      the DataFrame we want to transform
    id_col: string
      the column in long_form_df and cv representing a unique identifier for subjects.

    Example
    -------

    >>> cv = covariates_from_event_matrix(duration_df, 'id')
    >>> long_form_df = add_covariate_to_timeline(long_form_df, cv, 'id', 'duration', 'e', cumulative_sum=True)

    """
    df = df.set_index(id_col)
    df = df.stack().reset_index()
    df.columns = [id_col, "event", "duration"]
    df["_counter"] = 1
    return df.pivot_table(index=[id_col, "duration"], columns="event", fill_value=0)["_counter"].reset_index()


class StepSizer:
    """
    This class abstracts complicated step size logic out of the fitters. The API is as follows:

    > step_sizer = StepSizer(initial_step_size)
    > step_size = step_sizer.next()
    > step_sizer.update(some_convergence_norm)
    > step_size = step_sizer.next()


    ATM it contains lots of "magic constants"
    """

    def __init__(self, initial_step_size):
        initial_step_size = coalesce(initial_step_size, 0.95)
        self.initial_step_size = initial_step_size
        self.step_size = initial_step_size
        self.temper_back_up = False
        self.norm_of_deltas = []

    def update(self, norm_of_delta):
        SCALE = 1.2
        LOOKBACK = 3

        self.norm_of_deltas.append(norm_of_delta)

        # speed up convergence by increasing step size again
        if self.temper_back_up:
            self.step_size = min(self.step_size * SCALE, self.initial_step_size)

        # Only allow small steps
        if norm_of_delta >= 15.0:
            self.step_size *= 0.25
            self.temper_back_up = True
        elif 15.0 > norm_of_delta > 5.0:
            self.step_size *= 0.75
            self.temper_back_up = True

        # recent non-monotonically decreasing is a concern
        if len(self.norm_of_deltas) >= LOOKBACK and not self._is_monotonically_decreasing(
            self.norm_of_deltas[-LOOKBACK:]
        ):
            self.step_size *= 0.98

        # recent monotonically decreasing is good though
        if len(self.norm_of_deltas) >= LOOKBACK and self._is_monotonically_decreasing(self.norm_of_deltas[-LOOKBACK:]):
            self.step_size = min(self.step_size * SCALE, 1.0)

        return self

    @staticmethod
    def _is_monotonically_decreasing(array):
        return np.all(np.diff(array) < 0)

    def next(self):
        return self.step_size


def _to_array(x):
    if not isinstance(x, collections.Iterable):
        return np.array([x])
    return np.asarray(x)


def _to_list(x):
    if not isinstance(x, list):
        return [x]
    return x


def _to_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def format_p_value(decimals):
    threshold = 0.5 * 10 ** (-decimals)
    return lambda p: "<%s" % threshold if p < threshold else "{:4.{prec}f}".format(p, prec=decimals)


def format_exp_floats(decimals):
    """
    sometimes the exp. column can be too large
    """
    threshold = 10 ** 5
    return (
        lambda n: "{:.{prec}e}".format(n, prec=decimals) if n > threshold else "{:4.{prec}f}".format(n, prec=decimals)
    )


def format_floats(decimals):
    return lambda f: "{:4.{prec}f}".format(f, prec=decimals)


def dataframe_interpolate_at_times(df, times):
    return df.reindex(df.index.union(_to_array(times))).interpolate(method="index").loc[times].squeeze()


string_justify = lambda width: lambda s: s.rjust(width, " ")
