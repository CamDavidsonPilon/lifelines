# -*- coding: utf-8 -*-
from __future__ import print_function, division
import warnings
from datetime import datetime

import numpy as np
from numpy.linalg import inv
import pandas as pd
from pandas import to_datetime


class StatError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


def qth_survival_times(q, survival_functions):
    """
    This can be done much better.

    Parameters:
      q: a float between 0 and 1.
      survival_functions: a (n,d) dataframe or numpy array.
        If dataframe, will return index values (actual times)
        If numpy array, will return indices.

    Returns:
      v: if d==1, returns a float, np.inf if infinity.
         if d > 1, an DataFrame containing the first times the value was crossed.

    """
    q = pd.Series(q)
    assert (q <= 1).all() and (0 <= q).all(), 'q must be between 0 and 1'
    survival_functions = pd.DataFrame(survival_functions)
    if survival_functions.shape[1] == 1 and q.shape == (1,):
        return survival_functions.apply(lambda s: qth_survival_time(q[0], s)).ix[0]
    else:
        return pd.DataFrame({_q: survival_functions.apply(lambda s: qth_survival_time(_q, s)) for _q in q})


def qth_survival_time(q, survival_function):
    """
    Expects a Pandas series, returns the time when the qth probability is reached.
    """
    if survival_function.iloc[-1] > q:
        return np.inf
    v = (survival_function <= q).idxmax(0)
    return v


def median_survival_times(survival_functions):
    return qth_survival_times(0.5, survival_functions)


def group_survival_table_from_events(groups, durations, event_observed, birth_times=None, limit=-1):
    """
    Joins multiple event series together into dataframes. A generalization of
    `survival_table_from_events` to data with groups. Previously called `group_event_series` pre 0.2.3.

    Parameters:
        groups: a (n,) array of individuals' group ids.
        durations: a (n,)  array of durations of each individual
        event_observed: a (n,) array of event observations, 1 if observed, 0 else.
        birth_times: a (n,) array of numbers representing
          when the subject was first observed. A subject's death event is then at [birth times + duration observed].
          Normally set to all zeros, but can be positive or negative.

    Output:
        - np.array of unique groups
        - dataframe of removal count data at event_times for each group, column names are 'removed:<group name>'
        - dataframe of observed count data at event_times for each group, column names are 'observed:<group name>'
        - dataframe of censored count data at event_times for each group, column names are 'censored:<group name>'

    Example:
        #input
        group_survival_table_from_events(waltonG, waltonT, np.ones_like(waltonT)) #data available in test_suite.py

        #output
        [
            array(['control', 'miR-137'], dtype=object),

                      removed:control  removed:miR-137
            event_at
            6                       0                1
            7                       2                0
            9                       0                3
            13                      0                3
            15                      0                2
            ,
                      observed:control  observed:miR-137
            event_at
            6                        0                 1
            7                        2                 0
            9                        0                 3
            13                       0                 3
            15                       0                 2
            ,
                      censored:control  censored:miR-137
            event_at
            6                        0                 0
            7                        0                 0
            9                        0                 0
            ,
        ]

    """

    n = np.max(groups.shape)
    assert n == np.max(durations.shape) == np.max(event_observed.shape), "inputs must be of the same length."

    if birth_times is None:
        # Create some birth times
        birth_times = np.zeros(np.max(durations.shape))
        birth_times[:] = np.min(durations)

    assert n == np.max(birth_times.shape), "inputs must be of the same length."

    groups, durations, event_observed, birth_times = [pd.Series(np.reshape(data, (n,))) for data in [groups, durations, event_observed, birth_times]]
    unique_groups = groups.unique()

    for i, group in enumerate(unique_groups):
        ix = groups == group
        T = durations[ix]
        C = event_observed[ix]
        B = birth_times[ix]
        group_name = str(group)
        columns = [event_name + ":" + group_name for event_name in ['removed', 'observed', 'censored', 'entrance', 'at_risk']]
        if i == 0:
            data = survival_table_from_events(T, C, B, columns=columns)
        else:
            data = data.join(survival_table_from_events(T, C, B, columns=columns), how='outer')

    data = data.fillna(0)
    # hmmm pandas its too bad I can't do data.ix[:limit] and leave out the if.
    if int(limit) != -1:
        data = data.ix[:limit]

    return unique_groups, data.filter(like='removed:'), data.filter(like='observed:'), data.filter(like='censored:')


def survival_table_from_events(death_times, event_observed, birth_times=None,
                               columns=["removed", "observed", "censored", "entrance", "at_risk"],
                               weights=None):
    """
    Parameters:
        death_times: (n,) array of event times
        event_observed: (n,) boolean array, 1 if observed event, 0 is censored event.
        birth_times: a (n,) array of numbers representing
          when the subject was first observed. A subject's death event is then at [birth times + duration observed].
          If None (default), birth_times are set to be the first observation or 0, which ever is smaller.
        columns: a 3-length array to call the, in order, removed individuals, observed deaths
          and censorships.
        weights: Default None, otherwise (n,1) array. Optional argument to use weights for individuals.
    Returns:
        Pandas DataFrame with index as the unique times in event_times. The columns named
        'removed' refers to the number of individuals who were removed from the population
        by the end of the period. The column 'observed' refers to the number of removed
        individuals who were observed to have died (i.e. not censored.) The column
        'censored' is defined as 'removed' - 'observed' (the number of individuals who
         left the population due to event_observed)

    Example:

                  removed  observed  censored  entrance   at_risk
        event_at
        0               0         0         0        11        11
        6               1         1         0         0        11
        7               2         2         0         0        10
        9               3         3         0         0         8
        13              3         3         0         0         5
        15              2         2         0         0         2

    """
    removed, observed, censored, entrance, at_risk = columns
    death_times = np.asarray(death_times)
    if birth_times is None:
        birth_times = min(0, death_times.min()) * np.ones(death_times.shape[0])
    else:
        birth_times = np.asarray(birth_times)
        if np.any(birth_times > death_times):
            raise ValueError('birth time must be less than time of death.')

    # deal with deaths and censorships
    df = pd.DataFrame(death_times, columns=["event_at"])
    df[removed] = 1 if weights is None else weights
    df[observed] = np.asarray(event_observed)
    death_table = df.groupby("event_at").sum()
    death_table[censored] = (death_table[removed] - death_table[observed]).astype(int)

    # deal with late births
    births = pd.DataFrame(birth_times, columns=['event_at'])
    births[entrance] = 1
    births_table = births.groupby('event_at').sum()

    event_table = death_table.join(births_table, how='outer', sort=True).fillna(0)  # http://wesmckinney.com/blog/?p=414
    event_table[at_risk] = event_table[entrance].cumsum() - event_table[removed].cumsum().shift(1).fillna(0)
    return event_table.astype(float)


def survival_events_from_table(event_table, observed_deaths_col="observed", censored_col="censored"):
    """
    This is the inverse of the function ``survival_table_from_events``.

    Parameters
        event_table: a pandas DataFrame with index as the durations (!!) and columns "observed" and "censored", referring to
           the number of individuals that died and were censored at time t.

    Returns
        T: a np.array of durations of observation -- one element for each individual in the population.
        C: a np.array of event observations -- one element for each individual in the population. 1 if observed, 0 else.

    Ex: The survival table, as a pandas DataFrame:

                      observed  censored
        index
        1                1         0
        2                0         1
        3                1         0
        4                1         1
        5                0         1

    would return
        T = np.array([ 1.,  2.,  3.,  4.,  4.,  5.]),
        C = np.array([ 1.,  0.,  1.,  1.,  0.,  0.])

    """
    columns = [observed_deaths_col, censored_col]
    N = event_table[columns].sum().sum()
    T = np.empty(N)
    C = np.empty(N)
    i = 0
    for event_time, row in event_table.iterrows():
        n = row[columns].sum()
        T[i:i + n] = event_time
        C[i:i + n] = np.r_[np.ones(row[columns[0]]), np.zeros(row[columns[1]])]
        i += n

    return T, C


def datetimes_to_durations(start_times, end_times, fill_date=datetime.today(), freq='D', dayfirst=False, na_values=None):
    """
    This is a very flexible function for transforming arrays of start_times and end_times
    to the proper format for lifelines: duration and event observation arrays.

    Parameters:
        start_times: an array, series or dataframe of start times. These can be strings, or datetimes.
        end_times: an array, series or dataframe of end times. These can be strings, or datetimes.
                   These values can be None, or an empty string, which corresponds to censorship.
        fill_date: the date to use if end_times is a None or empty string. This corresponds to last date
                  of observation. Anything after this date is also censored. Default: datetime.today()
        freq: the units of time to use.  See pandas 'freq'. Default 'D' for days.
        day_first: convert assuming European-style dates, i.e. day/month/year.
        na_values : list of values to recognize as NA/NaN. Ex: ['', 'NaT']

    Returns:
        T: a array of floats representing the durations with time units given by freq.
        C: a boolean array of event observations: 1 if death observed, 0 else.

    """
    fill_date = pd.to_datetime(fill_date)
    freq_string = 'timedelta64[%s]' % freq
    start_times = pd.Series(start_times).copy()
    end_times = pd.Series(end_times).copy()

    C = ~(pd.isnull(end_times).values | end_times.isin(na_values or [""]))
    end_times[~C] = fill_date
    start_times_ = to_datetime(start_times, dayfirst=dayfirst)
    end_times_ = to_datetime(end_times, dayfirst=dayfirst, coerce=True)

    deaths_after_cutoff = end_times_ > fill_date
    C[deaths_after_cutoff] = False

    T = (end_times_ - start_times_).values.astype(freq_string).astype(float)
    if (T < 0).sum():
        warnings.warn("Warning: some values of start_times are after end_times")
    return T, C.values


def l1_log_loss(event_times, predicted_event_times, event_observed=None):
    """
    Calculates the l1 log-loss of predicted event times to true event times for *non-censored*
    individuals only.

    1/N \sum_{i} |log(t_i) - log(q_i)|

    Parameters:
      event_times: a (n,) array of observed survival times.
      predicted_event_times: a (n,) array of predicted survival times.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.

    Returns:
      l1-log-loss: a scalar
    """
    if event_observed is None:
        event_observed = np.ones_like(event_times)

    ix = event_observed.astype(bool)
    return np.abs(np.log(event_times[ix]) - np.log(predicted_event_times[ix])).mean()


def l2_log_loss(event_times, predicted_event_times, event_observed=None):
    """
    Calculates the l2 log-loss of predicted event times to true event times for *non-censored*
    individuals only.

    1/N \sum_{i} (log(t_i) - log(q_i))**2

    Parameters:
      event_times: a (n,) array of observed survival times.
      predicted_event_times: a (n,) array of predicted survival times.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.

    Returns:
      l2-log-loss: a scalar
    """
    if event_observed is None:
        event_observed = np.ones_like(event_times)

    ix = event_observed.astype(bool)
    return np.power(np.log(event_times[ix]) - np.log(predicted_event_times[ix]), 2).mean()


def concordance_index(event_times, predicted_event_times, event_observed=None):
    """
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The concordance index is a value between 0 and 1 where,
    0.5 is the expected result from random predictions,
    1.0 is perfect concordance and,
    0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Score is usually 0.6-0.7 for survival models.

    See:
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.

    Parameters:
      event_times: a (n,) array of observed survival times.
      predicted_event_times: a (n,) array of predicted survival times.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.

    Returns:
      c-index: a value between 0 and 1.
    """
    event_times = np.array(event_times, dtype=float)
    predicted_event_times = np.array(predicted_event_times, dtype=float)

    # Allow for (n, 1) or (1, n) arrays
    if event_times.ndim == 2 and (event_times.shape[0] == 1 or
                                  event_times.shape[1] == 1):
        # Flatten array
        event_times = event_times.ravel()
    # Allow for (n, 1) or (1, n) arrays
    if (predicted_event_times.ndim == 2 and
        (predicted_event_times.shape[0] == 1 or
         predicted_event_times.shape[1] == 1)):
        # Flatten array
        predicted_event_times = predicted_event_times.ravel()

    if event_times.shape != predicted_event_times.shape:
        raise ValueError("Event times and predictions must have the same shape")
    if event_times.ndim != 1:
        raise ValueError("Event times can only be 1-dimensional: (n,)")

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")
        event_observed = np.array(event_observed, dtype=float).ravel()

    return _concordance_index(event_times,
                              predicted_event_times,
                              event_observed)


def coalesce(*args):
    return next(s for s in args if s is not None)


def inv_normal_cdf(p):

    def AandS_approximation(p):
        # Formula 26.2.23 from A&S and help from John Cook ;)
        # http://www.johndcook.com/normal_cdf_inverse.html
        c_0 = 2.515517
        c_1 = 0.802853
        c_2 = 0.010328

        d_1 = 1.432788
        d_2 = 0.189269
        d_3 = 0.001308

        t = np.sqrt(-2 * np.log(p))

        return t - (c_0 + c_1 * t + c_2 * t ** 2) / (1 + d_1 * t + d_2 * t * t + d_3 * t ** 3)

    if p < 0.5:
        return -AandS_approximation(p)
    else:
        return AandS_approximation(1 - p)


def k_fold_cross_validation(fitters, df, duration_col, event_col=None,
                            k=5, evaluation_measure=concordance_index,
                            predictor="predict_expectation", predictor_kwargs={},
                            fitter_kwargs={}):
    """
    Perform cross validation on a dataset. If multiple models are provided,
    all models will train on each of the k subsets.

    fitter(s): one or several objects which possess a method:
                   fit(self, data, duration_col, event_col)
               Note that the last two arguments will be given as keyword arguments,
               and that event_col is optional. The objects must also have
               the "predictor" method defined below.
    df: a Pandas dataframe with necessary columns `duration_col` and `event_col`, plus
        other covariates. `duration_col` refers to the lifetimes of the subjects. `event_col`
        refers to whether the 'death' events was observed: 1 if observed, 0 else (censored).
    duration_col: the column in dataframe that contains the subjects lifetimes.
    event_col: the column in dataframe that contains the subject's death observation. If left
               as None, assumes all individuals are non-censored.
    k: the number of folds to perform. n/k data will be withheld for testing on.
    evaluation_measure: a function that accepts either (event_times, predicted_event_times),
                        or (event_times, predicted_event_times, event_observed)
                        and returns something (could be anything).
                        Default: statistics.concordance_index: (C-index)
                        between two series of event times
    predictor: a string that matches a prediction method on the fitter instances.
               For example, "predict_expectation" or "predict_percentile".
               Default is "predict_expectation"
               The interface for the method is:
                   predict(self, data, **optional_kwargs)
    fitter_kwargs: keyword args to pass into fitter.fit method
    predictor_kwargs: keyword args to pass into predictor-method.

    Returns:
        (k,1) list of scores for each fold. The scores can be anything.
    """
    # Make sure fitters is a list
    try:
        fitters = list(fitters)
    except TypeError:
        fitters = [fitters]
    # Each fitter has its own scores
    fitterscores = [[] for _ in fitters]

    n, d = df.shape
    df = df.copy()

    if event_col is None:
        event_col = 'E'
        df[event_col] = 1.

    df = df.reindex(np.random.permutation(df.index)).sort_values(event_col)

    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]

    testing_columns = df.columns.difference([duration_col, event_col])

    for i in range(1, k + 1):

        ix = assignments == i
        training_data = df.ix[~ix]
        testing_data = df.ix[ix]

        T_actual = testing_data[duration_col].values
        E_actual = testing_data[event_col].values
        X_testing = testing_data[testing_columns]

        for fitter, scores in zip(fitters, fitterscores):
            # fit the fitter to the training data
            fitter.fit(training_data, duration_col=duration_col,
                       event_col=event_col, **fitter_kwargs)
            T_pred = getattr(fitter, predictor)(X_testing, **predictor_kwargs).values

            try:
                scores.append(evaluation_measure(T_actual, T_pred, E_actual))
            except TypeError:
                scores.append(evaluation_measure(T_actual, T_pred))

    # If a single fitter was given as argument, return a single result
    if len(fitters) == 1:
        return fitterscores[0]
    else:
        return fitterscores


def normalize(X, mean=None, std=None):
    '''
    Normalize X. If mean OR std is None, normalizes
    X to have mean 0 and std 1.
    '''
    if mean is None or std is None:
        mean = X.mean(0)
        std = X.std(0)
    return (X - mean) / std


def unnormalize(X, mean, std):
    '''
    Reverse a normalization. Requires the original mean and
    standard deviation of the data set.
    '''
    return X * std + mean


def epanechnikov_kernel(t, T, bandwidth=1.):
    M = 0.75 * (1 - ((t - T) / bandwidth) ** 2)
    M[abs((t - T)) >= bandwidth] = 0
    return M


def significance_code(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ' '


def ridge_regression(X, Y, c1=0.0, c2=0.0, offset=None):
    """
    Also known as Tikhonov regularization. This solves the minimization problem:

    min_{beta} ||(beta X - Y)||^2 + c1||beta||^2 + c2||beta - offset||^2

    One can find more information here: http://en.wikipedia.org/wiki/Tikhonov_regularization

    Parameters:
        X: a (n,d) numpy array
        Y: a (n,) numpy array
        c1: a scalar
        c2: a scalar
        offset: a (d,) numpy array.

    Returns:
        beta_hat: the solution to the minimization problem.
        V = (X*X^T + (c1+c2)I)^{-1} X^T

    """
    n, d = X.shape
    X = X.astype(float)
    penalizer_matrix = (c1 + c2) * np.eye(d)

    if offset is None:
        offset = np.zeros((d,))

    V_1 = inv(np.dot(X.T, X) + penalizer_matrix)
    V_2 = (np.dot(X.T, Y) + c2 * offset)
    beta = np.dot(V_1, V_2)

    return beta, np.dot(V_1, X.T)


def _smart_search(minimizing_function, n, *args):
    from scipy.optimize import fmin_powell
    x = np.ones(n)
    return fmin_powell(minimizing_function, x, args=args, disp=False)


def _additive_estimate(events, timeline, _additive_f, _additive_var, reverse):
    """
    Called to compute the Kaplan Meier and Nelson-Aalen estimates.

    """
    if reverse:
        events = events.sort_index(ascending=False)
        at_risk = events['entrance'].sum() - events['removed'].cumsum().shift(1).fillna(0)

        deaths = events['observed']

        estimate_ = np.cumsum(_additive_f(at_risk, deaths)).sort_index().shift(-1).fillna(0)
        var_ = np.cumsum(_additive_var(at_risk, deaths)).sort_index().shift(-1).fillna(0)
    else:
        deaths = events['observed']
        at_risk = events['at_risk']
        estimate_ = np.cumsum(_additive_f(at_risk, deaths))
        var_ = np.cumsum(_additive_var(at_risk, deaths))

    timeline = sorted(timeline)
    estimate_ = estimate_.reindex(timeline, method='pad').fillna(0)
    var_ = var_.reindex(timeline, method='pad')
    var_.index.name = 'timeline'
    estimate_.index.name = 'timeline'

    return estimate_, var_


def _preprocess_inputs(durations, event_observed, timeline, entry):
    """
    Cleans and confirms input to what lifelines expects downstream
    """

    n = len(durations)
    durations = np.asarray(durations).reshape((n,))

    # set to all observed if event_observed is none
    if event_observed is None:
        event_observed = np.ones(n, dtype=int)
    else:
        event_observed = np.asarray(event_observed).reshape((n,)).copy().astype(int)

    if entry is not None:
        entry = np.asarray(entry).reshape((n,))

    event_table = survival_table_from_events(durations, event_observed, entry)
    if timeline is None:
        timeline = event_table.index.values
    else:
        timeline = np.asarray(timeline)

    return durations, event_observed, timeline.astype(float), entry, event_table


def _get_index(X):
    if isinstance(X, pd.DataFrame):
        index = list(X.index)
    else:
        # If it's not a dataframe, order is up to user
        index = list(range(X.shape[0]))
    return index


class _BTree(object):

    """A simple balanced binary order statistic tree to help compute the concordance.

    When computing the concordance, we know all the values the tree will ever contain. That
    condition simplifies this tree a lot. It means that instead of crazy AVL/red-black shenanigans
    we can simply do the following:

    - Store the final tree in flattened form in an array (so node i's children are 2i+1, 2i+2)
    - Additionally, store the current size of each subtree in another array with the same indices
    - To insert a value, just find its index, increment the size of the subtree at that index and
      propagate
    - To get the rank of an element, you add up a bunch of subtree counts

    """

    def __init__(self, values):
        """
        Parameters:
            values: List of sorted (ascending), unique values that will be inserted.
        """
        self._tree = self._treeify(values)
        self._counts = np.zeros_like(self._tree, dtype=int)

    @staticmethod
    def _treeify(values):
        """Convert the np.ndarray `values` into a complete balanced tree.

        Assumes `values` is sorted ascending. Returns a list `t` of the same length in which t[i] >
        t[2i+1] and t[i] < t[2i+2] for all i."""
        if len(values) == 1:  # this case causes problems later
            return values
        tree = np.empty_like(values)
        # Tree indices work as follows:
        # 0 is the root
        # 2n+1 is the left child of n
        # 2n+2 is the right child of n
        # So we now rearrange `values` into that format...

        # The first step is to remove the bottom row of leaves, which might not be exactly full
        last_full_row = int(np.log2(len(values) + 1) - 1)
        len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
        if len_ragged_row > 0:
            bottom_row_ix = np.s_[:2 * len_ragged_row:2]
            tree[-len_ragged_row:] = values[bottom_row_ix]
            values = np.delete(values, bottom_row_ix)

        # Now `values` is length 2**n - 1, so can be packed efficiently into a tree
        # Last row of nodes is indices 0, 2, ..., 2**n - 2
        # Second-last row is indices 1, 5, ..., 2**n - 3
        # nth-last row is indices (2**n - 1)::(2**(n+1))
        values_start = 0
        values_space = 2
        values_len = 2 ** last_full_row
        while values_start < len(values):
            tree[values_len - 1:2 * values_len - 1] = values[values_start::values_space]
            values_start += int(values_space / 2)
            values_space *= 2
            values_len = int(values_len / 2)
        return tree

    def insert(self, value):
        """Insert an occurrence of `value` into the btree."""
        i = 0
        n = len(self._tree)
        while i < n:
            cur = self._tree[i]
            self._counts[i] += 1
            if value < cur:
                i = 2 * i + 1
            elif value > cur:
                i = 2 * i + 2
            else:
                return
        raise ValueError("Value %s not contained in tree."
                         "Also, the counts are now messed up." % value)

    def __len__(self):
        return self._counts[0]

    def rank(self, value):
        """Returns the rank and count of the value in the btree."""
        i = 0
        n = len(self._tree)
        rank = 0
        count = 0
        while i < n:
            cur = self._tree[i]
            if value < cur:
                i = 2 * i + 1
                continue
            elif value > cur:
                rank += self._counts[i]
                # subtract off the right tree if exists
                nexti = 2 * i + 2
                if nexti < n:
                    rank -= self._counts[nexti]
                    i = nexti
                    continue
                else:
                    return (rank, count)
            else:  # value == cur
                count = self._counts[i]
                lefti = 2 * i + 1
                if lefti < n:
                    nleft = self._counts[lefti]
                    count -= nleft
                    rank += nleft
                    righti = lefti + 1
                    if righti < n:
                        count -= self._counts[righti]
                return (rank, count)
        return (rank, count)


def _concordance_index(event_times, predicted_event_times, event_observed):
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.

    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = _BTree(np.unique(died_pred))
    num_pairs = 0
    num_correct = 0
    num_tied = 0

    def handle_pairs(truth, pred, first_ix):
        """
        Handle all pairs that exited at the same time as truth[first_ix].

        Returns:
          (pairs, correct, tied, next_ix)
          new_pairs: The number of new comparisons performed
          new_correct: The number of comparisons correctly predicted
          next_ix: The next index that needs to be handled
        """
        next_ix = first_ix
        while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
            next_ix += 1
        pairs = len(times_to_compare) * (next_ix - first_ix)
        correct = 0
        tied = 0
        for i in range(first_ix, next_ix):
            rank, count = times_to_compare.rank(pred[i])
            correct += rank
            tied += count

        return (pairs, correct, tied, next_ix)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died
                                  or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(censored_truth, censored_pred, censored_ix)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored
                                or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(died_truth, died_pred, died_ix)
            for pred in died_pred[died_ix:next_ix]:
                times_to_compare.insert(pred)
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied

    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")

    return (num_correct + num_tied / 2) / num_pairs


def _naive_concordance_index(event_times, predicted_event_times, event_observed):
    """
    Fallback, simpler method to compute concordance.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    def valid_comparison(time_a, time_b, event_a, event_b):
        """True if times can be compared."""
        if time_a == time_b:
            # Ties are only informative if exactly one event happened
            return event_a != event_b
        elif event_a and event_b:
            return True
        elif event_a and time_a < time_b:
            return True
        elif event_b and time_b < time_a:
            return True
        else:
            return False

    def concordance_value(time_a, time_b, pred_a, pred_b):
        if pred_a == pred_b:
            # Same as random
            return 0.5
        elif pred_a < pred_b:
            return (time_a < time_b) or (time_a == time_b and event_a and not event_b)
        else:  # pred_a > pred_b
            return (time_a > time_b) or (time_a == time_b and not event_a and event_b)

    paircount = 0.0
    csum = 0.0

    for a in range(0, len(event_times)):
        time_a = event_times[a]
        pred_a = predicted_event_times[a]
        event_a = event_observed[a]
        # Don't want to double count
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_event_times[b]
            event_b = event_observed[b]

            if valid_comparison(time_a, time_b, event_a, event_b):
                paircount += 1.0
                csum += concordance_value(time_a, time_b, pred_a, pred_b)

    if paircount == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return csum / paircount
