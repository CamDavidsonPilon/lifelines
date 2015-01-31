# -*- coding: utf-8 -*-
from __future__ import print_function, division

from datetime import datetime

import numpy as np
from numpy.linalg import inv
import pandas as pd
from pandas import to_datetime

from lifelines._utils import concordance_index as _cindex


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

    # 100 times faster to calculate in Fortran
    return _cindex(event_times,
                   predicted_event_times,
                   event_observed)


def coalesce(*args):
    return next(s for s in args if s is not None)


def group_survival_table_from_events(groups, durations, event_observed, birth_times, limit=-1):
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
    n = max(groups.shape)
    assert n == max(durations.shape) == max(event_observed.shape) == max(birth_times.shape), "inputs must be of the same length."
    groups, durations, event_observed, birth_times = map(lambda x: pd.Series(np.reshape(x, (n,))), [groups, durations, event_observed, birth_times])
    unique_groups = groups.unique()

    # set first group
    g = unique_groups[0]
    ix = (groups == g)
    T = durations[ix]
    C = event_observed[ix]
    B = birth_times[ix]

    g_name = str(g)
    data = survival_table_from_events(T, C, B,
                                      columns=['removed:' + g_name, "observed:" + g_name, 'censored:' + g_name, 'entrance' + g_name])
    for g in unique_groups[1:]:
        ix = groups == g
        T = durations[ix]
        C = event_observed[ix]
        B = birth_times[ix]
        g_name = str(g)
        data = data.join(survival_table_from_events(T, C, B,
                                                    columns=['removed:' + g_name, "observed:" + g_name, 'censored:' + g_name, 'entrance' + g_name]),
                         how='outer')
    data = data.fillna(0)
    # hmmm pandas its too bad I can't do data.ix[:limit] and leave out the if.
    if int(limit) != -1:
        data = data.ix[:limit]
    return unique_groups, data.filter(like='removed:'), data.filter(like='observed:'), data.filter(like='censored:')


def survival_table_from_events(death_times, event_observed, birth_times=None,
                               columns=["removed", "observed", "censored", "entrance"], weights=None):
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
        #input
        survival_table_from_events( waltonT, np.ones_like(waltonT)) #available in test suite

        #output

                  removed  observed  censored  entrance
        event_at
        0               0         0         0        11
        6               1         1         0         0
        7               2         2         0         0
        9               3         3         0         0
        13              3         3         0         0
        15              2         2         0         0

    """
    death_times = np.asarray(death_times)
    if birth_times is None:
        birth_times = min(0, death_times.min()) * np.ones(death_times.shape[0])
    else:
        birth_times = np.asarray(birth_times)
        if np.any(birth_times > death_times):
            raise ValueError('birth time must be less than time of death.')

    # deal with deaths and censorships
    df = pd.DataFrame(death_times, columns=["event_at"])
    df[columns[0]] = 1 if weights is None else weights
    df[columns[1]] = np.asarray(event_observed)
    death_table = df.groupby("event_at").sum()
    death_table[columns[2]] = (death_table[columns[0]] - death_table[columns[1]]).astype(int)

    # deal with late births
    births = pd.DataFrame(birth_times, columns=['event_at'])
    births[columns[3]] = 1
    births_table = births.groupby('event_at').sum()

    event_table = death_table.join(births_table, how='outer', sort=True).fillna(0)  # http://wesmckinney.com/blog/?p=414
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
                  of observation. Anything above this date is also censored. Default: datetime.today()
        freq: the units of time to use.  See pandas 'freq'. Default 'D' for days.
        day_first: convert assuming European-style dates, i.e. day/month/year.
        na_values : Additional string to recognize as NA/NaN. Ex: ['']

    Returns:
        T: a array of floats representing the durations with time units given by freq.
        C: a boolean array of event observations: 1 if death observed, 0 else.

    """
    freq_string = 'timedelta64[%s]' % freq
    start_times = pd.Series(start_times).copy()
    end_times = pd.Series(end_times).copy()
    start_times_ = to_datetime(start_times, dayfirst=dayfirst)

    C = ~(pd.isnull(end_times).values | (end_times == "") | (end_times == na_values))
    end_times[~C] = fill_date
    """
    c =  (to_datetime(end_times, dayfirst=dayfirst, coerce=True) > fill_date)
    end_times[c] = fill_date
    C += c
    """
    end_times_ = to_datetime(end_times, dayfirst=dayfirst, coerce=True)

    T = (end_times_ - start_times_).map(lambda x: x.astype(freq_string).astype(float))
    if (T < 0).sum():
        print("Warning: some values of start_times are before end_times")
    return T.values, C.values


class StatError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


def inv_normal_cdf(p):
    if p < 0.5:
        return -AandS_approximation(p)
    else:
        return AandS_approximation(1 - p)


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


def k_fold_cross_validation(fitter, df, duration_col, event_col=None,
                            k=5, evaluation_measure=concordance_index, predictor="predict_median",
                            predictor_kwargs={}):
    """
    Perform cross validation on a dataset.

    fitter: either an instance of AalenAdditiveFitter or CoxPHFitter.
    df: a Pandas dataframe with necessary columns `duration_col` and `event_col`, plus
        other covariates. `duration_col` refers to the lifetimes of the subjects. `event_col`
        refers to whether the 'death' events was observed: 1 if observed, 0 else (censored).
    duration_col: the column in dataframe that contains the subjects lifetimes.
    event_col: the column in dataframe that contains the subject's death observation. If left
                as None, assumes all individuals are non-censored.
    k: the number of folds to perform. n/k data will be withheld for testing on.
    evaluation_measure: a function that accepts either (event_times, predicted_event_times),
                  or (event_times, predicted_event_times, event_observed) and returns a scalar value.
                  Default: statistics.concordance_index: (C-index) between two series of event times
    predictor: a string that matches a prediction method on the fitter instances. For example,
            "predict_expectation" or "predict_percentile". Default is "predict_median"
    predictor_kwargs: keyward args to pass into predictor.

    Returns:
        (k,1) array of scores for each fold.
    """
    n, d = df.shape
    scores = np.zeros((k,))
    df = df.copy()

    if event_col is None:
        event_col = 'E'
        df[event_col] = 1.

    df = df.reindex(np.random.permutation(df.index)).sort(event_col)

    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]

    testing_columns = df.columns - [duration_col, event_col]

    for i in range(1, k + 1):

        ix = assignments == i
        training_data = df.ix[~ix]
        testing_data = df.ix[ix]

        T_actual = testing_data[duration_col].values
        E_actual = testing_data[event_col].values
        X_testing = testing_data[testing_columns]

        # fit the fitter to the training data
        fitter.fit(training_data, duration_col=duration_col, event_col=event_col)
        T_pred = getattr(fitter, predictor)(X_testing, **predictor_kwargs).values

        try:
            scores[i - 1] = evaluation_measure(T_actual, T_pred, E_actual)
        except TypeError:
            scores[i - 1] = evaluation_measure(T_actual, T_pred)

    return scores


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


def _line_search(minimizing_function, x, delta_x, *args, **kwargs):
    log_min = kwargs.get('log_min', -5)
    log_max = kwargs.get('log_max', -1)
    ts = 10 ** np.linspace(log_min, log_max, 5)
    out = map(lambda t: minimizing_function(x - t * delta_x, *args), ts)
    return ts[np.argmin(out)]


def _smart_search(minimizing_function, n, *args):
    from scipy.optimize import fmin_powell
    x = np.ones(n)
    return fmin_powell(minimizing_function, x, args=args, disp=False)

