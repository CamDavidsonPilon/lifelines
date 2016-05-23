# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.integrate import trapz

from lifelines.fitters import BaseFitter
from lifelines.utils import _get_index, inv_normal_cdf, epanechnikov_kernel, \
    ridge_regression as lr, qth_survival_times, coalesce

from lifelines.utils.progress_bar import progress_bar
from lifelines.plotting import fill_between_steps


class AalenAdditiveFitter(BaseFitter):

    """
    This class fits the regression model:

    hazard(t)  = b_0(t) + b_t(t)*x_1 + ... + b_N(t)*x_N

    that is, the hazard rate is a linear function of the covariates.

    Parameters:
      fit_intercept: If False, do not attach an intercept (column of ones) to the covariate matrix. The
        intercept, b_0(t) acts as a baseline hazard.
      alpha: the level in the confidence intervals.
      coef_penalizer: Attach a L2 penalizer to the size of the coeffcients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the absolute value of c_{i,t}. Recommended, even if a small value.
      smoothing_penalizer: Attach a L2 penalizer to difference between adjacent (over time) coefficents. For
        example, this shrinks the absolute value of c_{i,t} - c_{i,t+1}.
      nn_cumulative_hazard: If True, forces the negative values in cumulative hazards to be 0 instead. Default True.

    """

    def __init__(self, fit_intercept=True, alpha=0.95, coef_penalizer=0.5, smoothing_penalizer=0., nn_cumulative_hazard=True):
        if not (0 < alpha <= 1.):
            raise ValueError('alpha parameter must be between 0 and 1.')
        if coef_penalizer < 0 or smoothing_penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")

        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_penalizer = coef_penalizer
        self.smoothing_penalizer = smoothing_penalizer
        self.nn_cumulative_hazard = nn_cumulative_hazard

    def fit(self, dataframe, duration_col, event_col=None,
            timeline=None, id_col=None, show_progress=True):
        """
        Perform inference on the coefficients of the Aalen additive model.

        Parameters:
            dataframe: a pandas dataframe, with covariates and a duration_col and a event_col.

                static covariates:
                    one row per individual. duration_col refers to how long the individual was
                      observed for. event_col is a boolean: 1 if individual 'died', 0 else. id_col
                      should be left as None.

                time-varying covariates:
                    For time-varying covariates, an id_col is required to keep track of individuals'
                    changing covariates. individual should have a unique id. duration_col refers to how
                    long the individual has been  observed to up to that point. event_col refers to if
                    the event (death) occured in that  period. Censored individuals will not have a 1.
                    For example:

                        +----+---+---+------+------+
                        | id | T | E | var1 | var2 |
                        +----+---+---+------+------+
                        |  1 | 1 | 0 |    0 |    1 |
                        |  1 | 2 | 0 |    0 |    1 |
                        |  1 | 3 | 0 |    4 |    3 |
                        |  1 | 4 | 1 |    8 |    4 |
                        |  2 | 1 | 0 |    1 |    1 |
                        |  2 | 2 | 0 |    1 |    2 |
                        |  2 | 3 | 0 |    1 |    2 |
                        +----+---+---+------+------+

            duration_col: specify what the duration column is called in the dataframe
            event_col: specify what the event column is called in the dataframe.
                       If left as None, treat all individuals as non-censored.
            timeline: reformat the estimates index to a new timeline.
            id_col: (only for time-varying covariates) name of the id column in the dataframe
            progress_bar: include a fancy progress bar =)

        Returns:
          self, with new methods like plot, smoothed_hazards_ and properties like cumulative_hazards_
        """

        if id_col is None:
            self._fit_static(dataframe, duration_col, event_col, timeline, show_progress)
        else:
            self._fit_varying(dataframe, duration_col, event_col, id_col, timeline, show_progress)

        return self

    def _fit_static(self, dataframe, duration_col, event_col=None,
                    timeline=None, show_progress=True):
        """
        Perform inference on the coefficients of the Aalen additive model.

        Parameters:
            dataframe: a pandas dataframe, with covariates and a duration_col and a event_col.
                      one row per individual. duration_col refers to how long the individual was
                      observed for. event_col is a boolean: 1 if individual 'died', 0 else. id_col
                      should be left as None.

            duration_col: specify what the duration column is called in the dataframe
            event_col: specify what the event occurred column is called in the dataframe
            timeline: reformat the estimates index to a new timeline.
            progress_bar: include a fancy progress bar!

        Returns:
          self, with new methods like plot, smoothed_hazards_ and properties like cumulative_hazards_
        """

        from_tuples = pd.MultiIndex.from_tuples
        df = dataframe.copy()

        # set unique ids for individuals
        id_col = 'id'
        ids = np.arange(df.shape[0])
        df[id_col] = ids

        # if the regression should fit an intercept
        if self.fit_intercept:
            df['baseline'] = 1.

        # if no event_col is specified, assume all non-censorships
        if event_col:
            c = df[event_col].values
            del df[event_col]
        else:
            c = np.ones_like(ids)

        # each individual should have an ID of time of leaving study
        C = pd.Series(c, dtype=bool, index=ids)
        T = pd.Series(df[duration_col].values, index=ids)

        df = df.set_index(id_col)

        ix = T.argsort()
        T, C = T.iloc[ix], C.iloc[ix]

        del df[duration_col]
        n, d = df.shape
        columns = df.columns

        # initialize dataframe to store estimates
        non_censorsed_times = list(T[C].iteritems())
        n_deaths = len(non_censorsed_times)

        hazards_ = pd.DataFrame(np.zeros((n_deaths, d)), columns=columns,
                                index=from_tuples(non_censorsed_times)).swaplevel(1, 0)

        variance_ = pd.DataFrame(np.zeros((n_deaths, d)), columns=columns,
                                 index=from_tuples(non_censorsed_times)).swaplevel(1, 0)

        # initialize loop variables.
        previous_hazard = np.zeros((d,))
        progress = progress_bar(n_deaths)
        to_remove = []
        t = T.iloc[0]
        i = 0

        for id, time in T.iteritems():  # should be sorted.

            if t != time:
                assert t < time
                # remove the individuals from the previous loop.
                df.iloc[to_remove] = 0.
                to_remove = []
                t = time

            to_remove.append(id)
            if C[id] == 0:
                continue

            relevant_individuals = (ids == id)
            assert relevant_individuals.sum() == 1.

            # perform linear regression step.
            try:
                v, V = lr(df.values, relevant_individuals, c1=self.coef_penalizer, c2=self.smoothing_penalizer, offset=previous_hazard)
            except LinAlgError:
                print("Linear regression error. Try increasing the penalizer term.")

            hazards_.ix[time, id] = v.T
            variance_.ix[time, id] = V[:, relevant_individuals][:, 0] ** 2
            previous_hazard = v.T

            # update progress bar
            if show_progress:
                i += 1
                progress.update(i)

        # print a new line so the console displays well
        if show_progress:
            print()

        # not sure this is the correct thing to do.
        self.hazards_ = hazards_.groupby(level=0).sum()
        self.cumulative_hazards_ = self.hazards_.cumsum()
        self.variance_ = variance_.groupby(level=0).sum()

        if timeline is not None:
            self.hazards_ = self.hazards_.reindex(timeline, method='ffill')
            self.cumulative_hazards_ = self.cumulative_hazards_.reindex(timeline, method='ffill')
            self.variance_ = self.variance_.reindex(timeline, method='ffill')
            self.timeline = timeline
        else:
            self.timeline = self.hazards_.index.values.astype(float)

        self.data = dataframe
        self.durations = T
        self.event_observed = C
        self._compute_confidence_intervals()

        return

    def _fit_varying(self, dataframe, duration_col="T", event_col="E",
                     id_col=None, timeline=None, show_progress=True):

        from_tuples = pd.MultiIndex.from_tuples
        df = dataframe.copy()

        # if the regression should fit an intercept
        if self.fit_intercept:
            df['baseline'] = 1.

        # each individual should have an ID of time of leaving study
        df = df.set_index([duration_col, id_col])

        # if no event_col is specified, assume all non-censorships
        if event_col is None:
            event_col = 'E'
            df[event_col] = 1

        C_panel = df[[event_col]].to_panel().transpose(2, 1, 0)
        C = C_panel.minor_xs(event_col).sum().astype(bool)
        T = (C_panel.minor_xs(event_col).notnull()).cumsum().idxmax()

        del df[event_col]
        n, d = df.shape

        # so this is a problem line. bfill performs a recursion which is
        # really not scalable. Plus even for modest datasets, this eats a lot of memory.
        # Plus is bfill the correct thing to choose? It's forward looking...
        wp = df.to_panel().bfill().fillna(0)

        # initialize dataframe to store estimates
        non_censorsed_times = list(T[C].iteritems())
        columns = wp.items
        hazards_ = pd.DataFrame(np.zeros((len(non_censorsed_times), d)),
                                columns=columns, index=from_tuples(non_censorsed_times))

        variance_ = pd.DataFrame(np.zeros((len(non_censorsed_times), d)),
                                 columns=columns, index=from_tuples(non_censorsed_times))

        previous_hazard = np.zeros((d,))
        ids = wp.minor_axis.values
        progress = progress_bar(len(non_censorsed_times))

        # this makes indexing times much faster
        wp = wp.swapaxes(0, 1, copy=False).swapaxes(1, 2, copy=False)

        for i, (id, time) in enumerate(non_censorsed_times):

            relevant_individuals = (ids == id)
            assert relevant_individuals.sum() == 1.

            # perform linear regression step.
            try:
                v, V = lr(wp[time].values, relevant_individuals, c1=self.coef_penalizer, c2=self.smoothing_penalizer, offset=previous_hazard)
            except LinAlgError:
                print("Linear regression error. Try increasing the penalizer term.")

            hazards_.ix[id, time] = v.T
            variance_.ix[id, time] = V[:, relevant_individuals][:, 0] ** 2
            previous_hazard = v.T

            # update progress bar
            if show_progress:
                progress.update(i)

        # print a new line so the console displays well
        if show_progress:
            print()

        ordered_cols = df.columns  # to_panel() mixes up my columns

        self.hazards_ = hazards_.groupby(level=1).sum()[ordered_cols]
        self.cumulative_hazards_ = self.hazards_.cumsum()[ordered_cols]
        self.variance_ = variance_.groupby(level=1).sum()[ordered_cols]

        if timeline is not None:
            self.hazards_ = self.hazards_.reindex(timeline, method='ffill')
            self.cumulative_hazards_ = self.cumulative_hazards_.reindex(timeline, method='ffill')
            self.variance_ = self.variance_.reindex(timeline, method='ffill')
            self.timeline = timeline
        else:
            self.timeline = self.hazards_.index.values.astype(float)

        self.data = wp

        self.durations = T
        self.event_observed = C
        self._compute_confidence_intervals()

        return

    def smoothed_hazards_(self, bandwidth=1):
        """
        Using the epanechnikov kernel to smooth the hazard function, with sigma/bandwidth

        """
        return pd.DataFrame(np.dot(epanechnikov_kernel(self.timeline[:, None], self.timeline, bandwidth), self.hazards_.values),
                            columns=self.hazards_.columns, index=self.timeline)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf(1 - (1 - self.alpha) / 2)
        n = self.timeline.shape[0]
        d = self.cumulative_hazards_.shape[1]
        index = [['upper'] * n + ['lower'] * n, np.concatenate([self.timeline, self.timeline])]

        self.confidence_intervals_ = pd.DataFrame(np.zeros((2 * n, d)),
                                                  index=index,
                                                  columns=self.cumulative_hazards_.columns
                                                  )

        self.confidence_intervals_.ix['upper'] = self.cumulative_hazards_.values + \
            alpha2 * np.sqrt(self.variance_.cumsum().values)

        self.confidence_intervals_.ix['lower'] = self.cumulative_hazards_.values - \
            alpha2 * np.sqrt(self.variance_.cumsum().values)
        return

    def predict_cumulative_hazard(self, X, id_col=None):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the hazard rates for the individuals
        """
        if id_col is not None:
            # see https://github.com/CamDavidsonPilon/lifelines/issues/38
            raise NotImplementedError

        n, d = X.shape

        cols = _get_index(X)
        if isinstance(X, pd.DataFrame):
            order = self.cumulative_hazards_.columns
            order = order.drop('baseline') if self.fit_intercept else order
            X_ = X[order].values.copy()
        else:
            X_ = X.copy()
        X_ = X_ if not self.fit_intercept else np.c_[X_, np.ones((n, 1))]
        individual_cumulative_hazards_ = pd.DataFrame(np.dot(self.cumulative_hazards_, X_.T), index=self.timeline, columns=cols)

        if self.nn_cumulative_hazard:
            individual_cumulative_hazards_[individual_cumulative_hazards_ < 0.] = 0.

        return individual_cumulative_hazards_

    def predict_survival_function(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the survival functions for the individuals
        """
        return np.exp(-self.predict_cumulative_hazard(X))

    def predict_percentile(self, X, p=0.5):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the median lifetimes for the individuals.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions
        """
        index = _get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X)[index])

    def predict_median(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the median lifetimes for the individuals
        """
        return self.predict_percentile(X, 0.5)

    def predict_expectation(self, X):
        """
        Compute the expected lifetime, E[T], using covarites X.

        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the expected lifetimes for the individuals
        """
        index = _get_index(X)
        t = self.cumulative_hazards_.index
        return pd.DataFrame(trapz(self.predict_survival_function(X)[index].values.T, t), index=index)

    def plot(self, ix=None, iloc=None, columns=[], legend=True, **kwargs):
        """"
        A wrapper around plotting. Matplotlib plot arguments can be passed in, plus:

          ix: specify a time-based subsection of the curves to plot, ex:
                   .plot(ix=slice(0.,10.)) will plot the time values between t=0. and t=10.
          iloc: specify a location-based subsection of the curves to plot, ex:
                   .plot(iloc=slice(0,10)) will plot the first 10 time points.
          columns: If not empty, plot a subset of columns from the cumulative_hazards_. Default all.
          legend: show legend in figure.

        """
        from matplotlib import pyplot as plt

        def shaded_plot(ax, x, y, y_upper, y_lower, **kwargs):
            base_line, = ax.plot(x, y, drawstyle='steps-post', **kwargs)
            fill_between_steps(x, y_lower, y2=y_upper, ax=ax, alpha=0.25,
                               color=base_line.get_color(), linewidth=1.0)

        assert (ix is None or iloc is None), 'Cannot set both ix and iloc in call to .plot'

        get_method = "ix" if ix is not None else "iloc"
        if iloc == ix is None:
            user_submitted_ix = slice(0, None)
        else:
            user_submitted_ix = ix if ix is not None else iloc
        get_loc = lambda df: getattr(df, get_method)[user_submitted_ix]

        if len(columns) == 0:
            columns = self.cumulative_hazards_.columns

        if 'ax' in kwargs:
            # don't use a .get here, as the default parameter will be called. In this case,
            # plt.figure().add_subplot(111), which instantiates a new window
            ax = kwargs['ax']
        else:
            ax = plt.figure().add_subplot(111)

        x = get_loc(self.cumulative_hazards_).index.values.astype(float)

        for column in columns:
            y = get_loc(self.cumulative_hazards_[column]).values
            y_upper = get_loc(self.confidence_intervals_[column].ix['upper']).values
            y_lower = get_loc(self.confidence_intervals_[column].ix['lower']).values
            shaded_plot(ax, x, y, y_upper, y_lower, label=kwargs.get('label', column))

        if legend:
            ax.legend()

        return ax
