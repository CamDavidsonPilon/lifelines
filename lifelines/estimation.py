from __future__ import print_function, division

import numpy as np
from numpy.linalg import LinAlgError, inv
from numpy import dot
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import beta
from scipy import stats

from lifelines.plotting import plot_estimate, plot_regressions
from lifelines.utils import survival_table_from_events, inv_normal_cdf, quadrature, \
                            epanechnikov_kernel, StatError, coalesce
from lifelines.progress_bar import progress_bar


class NelsonAalenFitter(object):

    """
    Class for fitting the Nelson-Aalen estimate for the cumulative hazard.

    NelsonAalenFitter( alpha=0.95, nelson_aalen_smoothing=True)

    alpha: The alpha value associated with the confidence intervals.
    nelson_aalen_smoothing: If the event times are naturally discrete (like discrete years, minutes, etc.)
      then it is advisable to turn this parameter to False. See [1], pg.84.

    """

    def __init__(self, alpha=0.95, nelson_aalen_smoothing=True):
        self.alpha = alpha
        self.nelson_aalen_smoothing = nelson_aalen_smoothing
        if self.nelson_aalen_smoothing:
            self._variance_f = self._variance_f_smooth
            self._additive_f = self._additive_f_smooth
        else:
            self._variance_f = self._variance_f_discrete
            self._additive_f = self._additive_f_discrete

    def fit(self, durations, event_observed=None, timeline=None, entry=None, label='NA-estimate', alpha=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          event_observed: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.

        Returns:
          self, with new properties like 'cumulative_hazard_'.

        """

        v = preprocess_inputs(durations, event_observed, timeline, entry)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v

        cumulative_hazard_, cumulative_sq_ = _additive_estimate(self.event_table, self.timeline,
                                                                self._additive_f, self._variance_f)

        # esimates
        self.cumulative_hazard_ = pd.DataFrame(cumulative_hazard_, columns=[label])
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha if alpha else self.alpha)
        self._cumulative_sq = cumulative_sq_

        # estimation functions
        self.predict = _predict(self, "cumulative_hazard_", label)
        self.subtract = _subtract(self, "cumulative_hazard_")
        self.divide = _divide(self, "cumulative_hazard_")

        # plotting
        self.plot = plot_estimate(self, "cumulative_hazard_")
        self.plot_cumulative_hazard = self.plot
        self.plot_hazard = plot_estimate(self, 'hazard_')

        return self

    def _bounds(self, cumulative_sq_, alpha):
        alpha2 = inv_normal_cdf(1 - (1 - alpha) / 2)
        df = pd.DataFrame(index=self.timeline)
        name = self.cumulative_hazard_.columns[0]
        df["%s_upper_%.2f" % (name, self.alpha)] = self.cumulative_hazard_.values * \
            np.exp(alpha2 * np.sqrt(cumulative_sq_) / self.cumulative_hazard_.values)
        df["%s_lower_%.2f" % (name, self.alpha)] = self.cumulative_hazard_.values * \
            np.exp(-alpha2 * np.sqrt(cumulative_sq_) / self.cumulative_hazard_.values)
        return df

    def _variance_f_smooth(self, population, deaths):
        df = pd.DataFrame({'N': population, 'd': deaths})
        return df.apply(lambda N_d: np.sum((1. / (N_d[0] - i) ** 2 for i in range(int(N_d[1])))), axis=1)

    def _variance_f_discrete(self, population, deaths):
        return 1. * (population - deaths) * deaths / population ** 3

    def _additive_f_smooth(self, population, deaths):
        df = pd.DataFrame({'N': population, 'd': deaths})
        return df.apply(lambda N_d: np.sum((1. / (N_d[0] - i) for i in range(int(N_d[1])))), axis=1)

    def _additive_f_discrete(self, population, deaths):
        return (1. * deaths / population).replace([np.inf], 0)

    def smoothed_hazard_(self, bandwidth):
        """
        Parameters:
          bandwidth: the bandwith used in the Epanechnikov kernel.

        Returns:
          a DataFrame of the smoothed hazard
        """
        timeline = self.timeline
        cumulative_hazard_name = self.cumulative_hazard_.columns[0]
        hazard_name = "smoothed-" + cumulative_hazard_name
        hazard_ = self.cumulative_hazard_.diff().fillna(self.cumulative_hazard_.iloc[0])
        C = (hazard_[cumulative_hazard_name] != 0.0).values
        return pd.DataFrame( 1./(2*bandwidth)*np.dot(epanechnikov_kernel(timeline[:, None], timeline[C][None,:], bandwidth), hazard_.values[C,:]),
                             columns=[hazard_name], index=timeline)

    def smoothed_hazard_confidence_intervals_(self, bandwidth, hazard_=None):
        """
        Parameter:
          bandwidth: the bandwith to use in the Epanechnikov kernel.
          hazard_: a computed (n,) numpy array of estimated hazard rates. If none, uses naf.smoothed_hazard_
        """
        if hazard_ == None:
            hazard_ = self.smoothed_hazard_(bandwidth).values[:, 0]

        timeline = self.timeline
        alpha2 = inv_normal_cdf(1 - (1 - self.alpha) / 2)
        name = "smoothed-" + self.cumulative_hazard_.columns[0]
        self._cumulative_sq.iloc[0] = 0
        var_hazard_ = self._cumulative_sq.diff().fillna(self._cumulative_sq.iloc[0])
        C = (var_hazard_.values != 0.0)  # only consider the points with jumps
        std_hazard_ = np.sqrt(1./(2*bandwidth**2)*np.dot(epanechnikov_kernel(timeline[:, None], timeline[C][None,:], bandwidth)**2, var_hazard_.values[C]))
        values = {
            "%s_upper_%.2f" % (name, self.alpha): hazard_ * np.exp(alpha2 * std_hazard_ / hazard_),
            "%s_lower_%.2f" % (name, self.alpha): hazard_ * np.exp(-alpha2 * std_hazard_ / hazard_)
        }
        return pd.DataFrame(values, index=timeline)

    def __repr__(self):
        try:
            s = """<lifelines.NelsonAalenFitter: fitted with %d observations, %d censored>""" % (
                self.event_observed.shape[0], (1-self.event_observed).sum())
        except AttributeError as e:
            s = """<lifelines.NelsonAalenFitter>"""
        return s


class KaplanMeierFitter(object):

    """
    Class for fitting the Kaplan-Meier estimate for the survival function.

    KaplanMeierFitter( alpha=0.95)

    alpha: The alpha value associated with the confidence intervals.

    """


    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def fit(self, durations, event_observed=None, timeline=None, entry=None, label='KM-estimate', alpha=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          event_observed: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.

        Returns:
          self, with new properties like 'survival_function_'.

        """
        v = preprocess_inputs(durations, event_observed, timeline, entry)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v

        log_survival_function, cumulative_sq_ = _additive_estimate(self.event_table, self.timeline,
                                                                   self._additive_f, self._additive_var)

        if entry is not None:
            #a serious problem with KM is that when the sample size is small and there are too few early
            # truncation times, it may happen that is the number of patients at risk and the number of deaths is the same.
            # we adjust for this using the Breslow-Fleming-Harrington estimator
            n = self.event_table.shape[0]
            net_population = (self.event_table['entrance'] - self.event_table['removed']).cumsum()
            if net_population.iloc[:int(n/2)].min() == 0:
                ix = net_population.iloc[:int(n/2)].argmin()
                raise StatError("""There are too few early truncation times and too many events. S(t)==0 for all t>%.1f. Recommend BFH estimator."""%ix)


        # estimation
        self.survival_function_ = pd.DataFrame(np.exp(log_survival_function), columns=[label])
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha if alpha else self.alpha)
        self.median_ = median_survival_times(self.survival_function_)

        # estimation methods
        self.predict = _predict(self, "survival_function_", label)
        self.subtract = _subtract(self, "survival_function_")
        self.divide = _divide(self, "survival_function_")

        # plotting functions
        self.plot = plot_estimate(self, "survival_function_")
        self.plot_survival_function = self.plot
        return self

    def _bounds(self, cumulative_sq_, alpha):
        # See http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes2.pdfg
        alpha2 = inv_normal_cdf((1. + alpha) / 2.)
        df = pd.DataFrame(index=self.timeline)
        name = self.survival_function_.columns[0]
        v = np.log(self.survival_function_.values)
        df["%s_upper_%.2f" % (name, self.alpha)] = np.exp(-np.exp(np.log(-v) + alpha2 * np.sqrt(cumulative_sq_) / v))
        df["%s_lower_%.2f" % (name, self.alpha)] = np.exp(-np.exp(np.log(-v) - alpha2 * np.sqrt(cumulative_sq_) / v))
        return df

    def _additive_f(self, population, deaths):
        np.seterr(invalid='ignore')
        return (np.log(population - deaths) - np.log(population))

    def _additive_var(self, population, deaths):
        np.seterr(divide='ignore')
        return (1. * deaths / (population * (population - deaths))).replace([np.inf], 0)

    def __repr__(self):
        try:
            s = """<lifelines.KaplanMeierFitter: fitted with %d observations, %d censored>""" % (
                self.event_observed.shape[0], (1-self.event_observed).sum())
        except AttributeError as e:
            s = """<lifelines.KaplanMeierFitter>"""
        return s


class BreslowFlemingHarringtonFitter(object):

    """
    Class for fitting the Breslow-Fleming-Harrington estimate for the survival function. This estimator
    is a biased estimator of the survival function but is more stable when the popualtion is small and
    there are too few early truncation times, it may happen that is the number of patients at risk and
    the number of deaths is the same.

    Mathematically, the NAF estimator is the negative logarithm of the BFH estimator.

    BreslowFlemingHarringtonFitter(alpha=0.95)

    alpha: The alpha value associated with the confidence intervals.

    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def fit(self, durations, event_observed=None, timeline=None, entry=None, label='BFH-estimate', alpha=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          event_observed: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.

        Returns:
          self, with new properties like 'survival_function_'.

        """
        naf = NelsonAalenFitter(self.alpha)
        naf.fit(durations, event_observed=event_observed, timeline=timeline, label=label, entry=entry)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = \
                naf.durations, naf.event_observed, naf.timeline, naf.entry, naf.event_table

        # estimation
        self.survival_function_ = np.exp(-naf.cumulative_hazard_)
        self.confidence_interval_ = np.exp(-naf.confidence_interval_)
        self.median_ = median_survival_times(self.survival_function_)

        # estimation methods
        self.predict = _predict(self, "survival_function_", label)
        self.subtract = _subtract(self, "survival_function_")
        self.divide = _divide(self, "survival_function_")

        # plotting functions
        self.plot = plot_estimate(self, "survival_function_")
        self.plot_survival_function = self.plot
        return self

    def __repr__(self):
        try:
            s = """<lifelines.BreslowFlemingHarringtonFitter: fitted with %d observations, %d censored>""" % (
                self.event_observed.shape[0], (1-self.event_observed).sum())
        except AttributeError as e:
            s = """<lifelines.BreslowFlemingHarringtonFitter>"""
        return s


class BayesianFitter(object):
    """
    If you have small data, and KM feels too uncertain, you can use the BayesianFitter to
    generate sample survival functions. The algorithm is:

    S_i(T) = \Prod_{t=0}^T (1 - p_t)

    where p_t ~ Beta( 0.01 + d_t, 0.01 + n_t - d_t), d_t is the number of deaths and n_t is the size of the
    population at risk at time t. The prior is a Beta(0.01, 0.01) for each time point (high values led to a
    high bias).

    Parameters:
        samples: the number of sample survival functions to return.

    """

    def __init__(self, samples = 100):
        self.beta = beta
        self.samples = samples

    def fit(self, durations, censorship=None, timeline=None, entry=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          censorship: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if censorship==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)

        Returns:
          self, with new properties like 'sample_survival_functions_'.
        """
        v = preprocess_inputs(durations, censorship, timeline, entry)
        self.durations, self.censorship, self.timeline, self.entry, self.event_table = v

        self.sample_survival_functions_ = self.generate_sample_path(self.samples)

        return self

    def plot(self, **kwargs):
        kwargs['alpha'] = coalesce(kwargs.get('alpha'), 15./self.samples)
        kwargs['legend'] = False
        kwargs['c'] = coalesce( kwargs.get('c'), kwargs.get('color'), '#348ABD')
        ax = self.sample_survival_functions_.plot(**kwargs)
        return ax

    def generate_sample_path(self, n=1):

        deaths = self.event_table['observed']
        population = self.event_table['entrance'].cumsum() - self.event_table['removed'].cumsum().shift(1).fillna(0)
        d = deaths.shape[0]
        samples = 1. - beta( 0.01 + deaths, 0.01 + population - deaths, size=(n, d))
        sample_paths = pd.DataFrame( np.exp(np.log(samples).cumsum(1)).T, index = self.timeline )
        return sample_paths


class AalenAdditiveFitter(object):
    """
    This class fits the regression model:

    hazard(t)  = b_0(t) + b_t(t)*x_1 + ... + b_N(t)*x_N

    that is, the hazard rate is a linear function of the covariates. Currently the covariates
    must be time independent.

    Parameters:
      fit_intercept: If False, do not attach an intercept (column of ones) to the covariate matrix. The
        intercept, b_0(t) acts as a baseline hazard.
      alpha: the level in the confidence intervals.
      penalizer: Attach a L2 penalizer to the regression. This improves stability of the estimates
       and controls high correlation between covariates.

    """

    def __init__(self, fit_intercept=True, alpha=0.95, penalizer=0.5):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.penalizer = penalizer
        assert penalizer >= 0, "penalizer must be >= 0."


    def fit(self, dataframe, duration_col="T", event_col="E", timeline=None, id_col=None, show_progress=True):
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
            event_col: specify what the event occurred column is called in the dataframe
            timeline: reformat the estimates index to a new timeline.
            id_col: (only for time-varying covariates) name of the id column in the dataframe
            progress_bar: include a fancy progress bar!


        Returns:
          self, with new methods like plot, smoothed_hazards_ and properties like cumulative_hazards_
        """
        from_tuples = pd.MultiIndex.from_tuples
        df = dataframe.copy()

        #only for time-indp. covariates
        if id_col is None:
            df['id'] = np.arange(df.shape[0])
            id_col = 'id'

        #if the regression should fit an intercept
        if self.fit_intercept:
            df['baseline'] = 1.

        #each individual should have an ID of time of leaving study
        df = df.set_index([id_col, duration_col])

        C_panel = df[[event_col]].to_panel().transpose(1,2,0)
        C = C_panel.minor_xs(event_col).sum().astype(bool)
        T = (C_panel.minor_xs(event_col).notnull()).cumsum().idxmax()

        del df[event_col]
        n,d = df.shape


        wp = df.to_panel().transpose(1,2,0).bfill().fillna(0) #bfill will cause problems later, plus it is slow.

        non_censorsed_times = T[C].iteritems()

        hazards_ = pd.DataFrame( np.zeros((len(non_censorsed_times),d)),
                        columns = df.columns, index = from_tuples(non_censorsed_times))

        variance_  = pd.DataFrame( np.zeros((len(non_censorsed_times),d)),
                        columns = df.columns, index = from_tuples(non_censorsed_times))

        #initializes the penalizer matrix
        penalizer = self.penalizer*np.eye(d)
        ids = wp.items
        progress = progress_bar(len(non_censorsed_times))
        #wp = wp.transpose(1,0,2)

        for i,(id, time) in enumerate(non_censorsed_times):

            relevant_individuals = (ids==id)
            assert relevant_individuals.sum() == 1.

            #X = wp[time].values
            X = wp.major_xs(time).values.T

            #perform linear regression step.
            try:
                V = dot(inv(dot(X.T, X) + penalizer), X.T)
            except LinAlgError:
                print("Linear regression error. Try increasing the penalizer term.")

            v = dot(V, 1.0*relevant_individuals )

            hazards_.ix[id, time]  = v.T
            variance_.ix[id, time] = V[:, relevant_individuals][:,0]**2

            #update progress bar
            if show_progress:
                progress.update(i)

        #print a new line so the console displays well
        if show_progress:
            print()

        #not sure this is the correct thing to do.
        self.hazards_ = hazards_.groupby(level=1).sum()
        self.cumulative_hazards_= self.hazards_.cumsum()
        self.variance_ = variance_.groupby(level=1).sum()

        if timeline is not None:
            self.hazards_ = self.hazards_.reindex(timeline, method='ffill')
            self.cumulative_hazards_ = self.cumulative_hazards_.reindex(timeline, method='ffill')
            self.variance_= self.variance_.reindex(timeline, method='ffill')
            self.timeline = timeline
        else:
            self.timeline = self.hazards_.index.values.astype(float)

        self.data = wp

        self.durations = T
        self.event_observed = C
        self._compute_confidence_intervals()
        self.plot = plot_regressions(self)

        return self

    def smoothed_hazards_(self, bandwidth=1):
        """
        Using the epanechnikov kernel to smooth the hazard function, with sigma/bandwidth

        """
        return pd.DataFrame( np.dot(epanechnikov_kernel(self.timeline[:, None], self.timeline, bandwidth), self.hazards_.values),
                             columns=self.hazards_.columns, index=self.timeline)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf(1 - (1 - self.alpha) / 2)
        n = self.timeline.shape[0]
        d = self.cumulative_hazards_.shape[1]
        index = [['upper'] * n + ['lower'] * n, np.concatenate([self.timeline, self.timeline])]

        self.confidence_intervals_ = pd.DataFrame( np.zeros((2 * n, d)),
                                                    index=index,
                                                    columns=self.cumulative_hazards_.columns
                                                  )

        self.confidence_intervals_.ix['upper'] = self.cumulative_hazards_.values + \
                alpha2 * np.sqrt(self.variance_.cumsum().values)

        self.confidence_intervals_.ix['lower'] = self.cumulative_hazards_.values - \
                alpha2 * np.sqrt(self.variance_.cumsum().values)
        return

    def predict_cumulative_hazard(self, X, columns=None, id_col=None):
        """
        X: a (n,d) covariate matrix

        Returns the hazard rates for the individuals
        """
        if id_col is not None:
            #see https://github.com/CamDavidsonPilon/lifelines/issues/38
            raise NotImplementedError

        n, d = X.shape
        try:
            X_ = X.values.copy()
        except:
            X_ = X.copy()
        X_ = X.copy() if not self.fit_intercept else np.c_[X.copy(), np.ones((n, 1))]
        return pd.DataFrame(np.dot(self.cumulative_hazards_, X_.T), index=self.timeline, columns=columns)

    def predict_survival_function(self, X, columns=None):
        """
        X: a (n,d) covariate matrix

        Returns the survival functions for the individuals
        """
        return np.exp(-self.predict_cumulative_hazard(X, columns=columns))

    def predict_median(self, X):
        """
        X: a (n,d) covariate matrix
        Returns the median lifetimes for the individuals
        """
        return median_survival_times(self.predict_survival_function(X))

    def predict_expectation(self, X):
        """
        Compute the expected lifetime, E[T], using covarites X.
        """
        t = self.cumulative_hazards_.index
        return quadrature(self.predict_survival_function(X).values.T, t)

    def __repr__(self):
        try:
            s = """<lifelines.AalenAdditiveFitter: fitted with %d observations, %d censored>""" % (
                self.event_observed.shape[0], (1.-self.event_observed).sum())
        except AttributeError as e:
            s = """<lifelines.AalenAdditiveFitter>"""
        return s


class RobustROSFitter(object):
    '''
    Class to implement the Robust regression-on-order statistics (ROS)
    method outlined in Nondetects and Data Analysis by Dennis R. Helsel
    (2005) to estimate the left-censored (non-detect) values of a
    dataset.

    Parameters
    ----------
    data : pandas DataFrame
        The censored dataset for which the non-detect values need to be
        estimated.

    result_col : optional string (default='res')
        The name of the column containing the numerical values of the
        dataset. Left-censored values should be set to the detection
        limit.

    censorship_col : optional string (default='cen')
        The name of the column containing indicating which observations
        are censored.
        `True` implies Left-censorship. `False` -> uncensored.

    Attributes
    ----------
    N_obs : int
        Total number of results in the dataset

    N_cen : int
        Total number of non-detect results in the dataset.

    cohn : pandas DataFrame
        A DataFrame of the unique detection limits found in `data` along
        with the `A`, `B`, `C`, and `PE` quantities computed by the
        estimation.

    data : pandas DataFrame
        An expanded version of the original dataset `data` passed the
        constructor included in the `modeled` column.

    debug : pandas DataFrame
        A full version of the `data` DataFrame that includes preliminary
        quantities.

    Example
    -------
    >>> from lifelines.estimation import RobustROS
    >>> ros = RobustROSFitter(myDataFrame, result_col='conc',
                              censorship_col='censored')
    >>> ros.fit()
    >>> print(ros.data)

    Notes
    -----
    It is inappropriate to replace specific left-censored values with
    the estimated values from this method. The estimated values
    (self.data['modeled']) should instead be used to refine descriptive
    statistics of the dataset as a whole.

    '''
    def __init__(self, data, result_col='res', censorship_col='cen'):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input `data` must be a pandas DataFrame")

        if not data.index.is_unique:
            raise ValueError("Index of input DataFrame `data` must be unique")

        if data[result_col].min() <= 0:
            raise ValueError('All result values of `data` must be positive')

        # rename the dataframe columns to the standard names
        # these will be used throughout ros.py when convenient
        newdata = data.rename(columns={
            result_col: 'res', censorship_col: 'cen'
        })

        # confirm a datatype real quick
        try:
            newdata.res = np.float64(newdata.res)
        except ValueError:
            raise ValueError('Result data is not uniformly numeric')

        # and get the basic info
        self.N_obs = newdata.shape[0]
        self.N_cen = newdata[newdata.cen].shape[0]

        # sort the data
        self.data = _ros_sort(newdata, result_col='res', censorship_col='cen')

        # create a dataframe of detection limits and their parameters
        # used in the ROS estimation
        self.cohn = self._get_cohn_numbers()

    def _get_cohn_numbers(self):
        '''
        Computes the Cohn numbers for the delection limits in the dataset.
        '''

        def _A(row):
            '''
            Helper function to compute the `A` quantity.
            '''
            # index of results above the lower DL
            above = self.data.res >= row['lower']

            # index of results below the upper DL
            below = self.data.res < row['upper']

            # index of non-detect results
            detect = self.data.cen == False

            # return the number of results where all condictions are True
            return self.data[above & below & detect].shape[0]

        def _B(row):
            '''
            Helper function to compute the `B` quantity
            '''
            # index of data less than the lower DL
            less_than = self.data.res < row['lower']

            # index of data less than or equal to the lower DL
            less_thanequal = self.data.res <= row['lower']

            # index of detects, non-detects
            detect = self.data.cen == False
            nondet = self.data.cen == True

            # number results less than or equal to lower DL and non-detect
            LTE_nondets = self.data[less_thanequal & nondet].shape[0]

            # number of results less than lower DL and detected
            LT_detects = self.data[less_than & detect].shape[0]

            # return the sum
            return LTE_nondets + LT_detects

        def _C(row):
            '''
            Helper function to compute the `C` quantity
            '''
            censored_below = self.data.res[self.data.cen] == row['lower']
            return censored_below.sum()

        # unique values
        cohn = pd.unique(self.data.res[self.data.cen])

        # if there is a results smaller than the minimum detection limit,
        # add that value to the array
        if cohn.shape[0] > 0:
            if self.data.res.min() < cohn.min():
                cohn = np.hstack([self.data.res.min(), cohn])

            # create a dataframe
            cohn = pd.DataFrame(cohn, columns=['DL'])

            # copy the cohn in two columns. offset the 2nd (upper) column
            cohn['lower'] = cohn['DL']
            if cohn.shape[0] > 1:
                cohn['upper'] = cohn.DL.shift(-1).fillna(value=np.inf)
            else:
                cohn['upper'] = np.inf

            # compute A, B, and C
            cohn['A'] = cohn.apply(_A, axis=1)
            cohn['B'] = cohn.apply(_B, axis=1)
            cohn['C'] = cohn.apply(_C, axis=1)

            # add an extra row
            cohn = cohn.reindex(range(cohn.shape[0]+1))

            # add the 'PE' column, initialize with zeros
            cohn['PE'] = 0.0

        else:
            dl_cols = ['DL', 'lower', 'upper', 'A', 'B', 'C', 'PE']
            cohn = pd.DataFrame(np.empty((0,7)), columns=dl_cols)

        return cohn

    def fit(self):
        '''
        Estimates the values of the censored data
        '''
        def _ros_DL_index(row):
            '''
            Helper function to create an array of indices for the
            detection  limits (self.cohn) corresponding to each
            data point
            '''
            DLIndex = np.zeros(len(self.data.res))
            if self.cohn.shape[0] > 0:
                index, = np.where(self.cohn['DL'] <= row['res'])
                DLIndex = index[-1]
            else:
                DLIndex = 0

            return DLIndex

        def _ros_plotting_pos(row):
            '''
            Helper function to compute the ROS'd plotting position
            '''
            dl_1 = self.cohn.iloc[row['DLIndex']]
            dl_2 = self.cohn.iloc[row['DLIndex']+1]
            if row['cen']:
                return (1 - dl_1['PE']) * row['Rank']/(dl_1['C']+1)
            else:
                return (1 - dl_1['PE']) + (dl_1['PE'] - dl_2['PE']) * \
                        row['Rank'] / (dl_1['A']+1)

        def _select_modeled(row):
            '''
            Helper fucntion to select "final" data from original detects
            and estimated non-detects
            '''
            if row['cen']:
                return row['modeled_data']
            else:
                return row['res']

        def _select_half_DL(row):
            '''
            Helper function to select half cohn when there are
            too few detections
            '''
            if row['cen']:
                return 0.5 * row['res']
            else:
                return row['res']

        # create a DLIndex column that references self.cohn
        self.data['DLIndex'] = self.data.apply(_ros_DL_index, axis=1)

        # compute the ranks of the data
        self.data['Rank'] = 1
        groupcols = ['DLIndex', 'cen', 'Rank']
        rankgroups = self.data[groupcols].groupby(by=['DLIndex', 'cen'])
        self.data['Rank'] = rankgroups.transform(lambda x: x.cumsum())

        # detect/non-detect selectors
        detect_selector = self.data.cen == False
        nondet_selector = self.data.cen == True

        # if there are no non-detects, just spit everything back out
        if self.N_cen == 0:
            self.data['modeled'] = self.data['res']

        # if there are too few detects, use half DL
        elif self.N_obs - self.N_cen < 2 or self.N_cen/self.N_obs > 0.8:
            self.data['modeled'] = self.data.apply(_select_half_DL, axis=1)

        # in most cases, actually use the MR method to estimate NDs
        else:
            # compute the PE values
            for j in self.cohn.index[:-1][::-1]:
                self.cohn.iloc[j]['PE'] = self.cohn.iloc[j+1]['PE'] + \
                   self.cohn.iloc[j]['A'] / \
                   (self.cohn.iloc[j]['A'] + self.cohn.iloc[j]['B']) * \
                   (1 - self.cohn.loc[j+1]['PE'])

            # compute the plotting position of the data (uses the PE stuff)
            self.data['plot_pos'] = self.data.apply(_ros_plotting_pos, axis=1)

            # correctly sort the plotting positions of the ND data:
            ND_plotpos = self.data['plot_pos'][self.data.cen]
            ND_plotpos.values.sort()
            self.data['plot_pos'][self.data.cen] = ND_plotpos

            # estimate a preliminary value of the Z-scores
            self.data['Zprelim'] = stats.norm.ppf(self.data['plot_pos'])

            # fit a line to the logs of the detected data
            fit = stats.linregress(self.data['Zprelim'][detect_selector],
                                   np.log(self.data['res'][detect_selector]))

            # save the fit params to an attribute
            self.fit = fit

            # pull out the slope and intercept for use later
            slope, intercept = fit[:2]

            # model the data based on the best-fit curve
            self.data['modeled_data'] = np.exp(
                slope*self.data['Zprelim'][nondet_selector] + intercept
            )

            # select out the final data
            self.data['modeled'] = self.data.apply(_select_modeled, axis=1)

        # create the debug attribute as a copy of the self.data attribute
        self.debug = self.data.copy(deep=True)

        # select out only the necessary columns for data
        self.data = self.data[['modeled', 'res', 'cen']]

        return self

    def plot(self, ax=None, show_raw=True, raw_kwds={}, model_kwds={},
             leg_kwds={}, ylog=True):
        '''
        Generate a QQ plot of the raw (censored) and modeled data.

        Parameters
        ----------
        ax : optional matplotlib Axes
            The axis on which the figure will be drawn. If no specified
            a new one is created.

        show_raw : optional boolean (default = True)
            Toggles on (True) or off (False) the drawing of the censored
            quantiles.

        raw_kwds : optional dict
            Plotting parameters for the censored data. Passed directly
            to `ax.plot`.

        model_kwds : optional dict
            Plotting parameters for the modeled data. Passed directly to
            `ax.plot`.

        leg_kwds : optional dict
            Optional kwargs for the legend, which is only drawn if
            `show_raw` is True. Passed directly to `ax.legend`.

        ylog : optional boolean (default = True)
            Toggles the logarthmic scale of the y-axis.

        Returns
        -------
        ax : matplotlib Axes

        '''
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # legend options
        leg_params = {
            'loc': 'upper left',
            'fontsize': 8
        }
        leg_params.update(leg_kwds)

        # modeled data
        mod_symbols = {
            'marker': 'o',
            'markersize': 6,
            'markeredgewidth': 1.0,
            'markeredgecolor': 'none',
            'markerfacecolor': 'CornflowerBlue',
            'linestyle': 'none',
            'label': 'Modeled data',
            'alpha': 0.87
        }
        mod_symbols.update(model_kwds)
        osm_mod, osr_mod = stats.probplot(self.data['modeled'], fit=False)
        ax.plot(osm_mod, osr_mod, **mod_symbols)

        # raw data
        if show_raw:
            raw_symbols = {
                'marker': 's',
                'markersize': 6,
                'markeredgewidth': 1.0,
                'markeredgecolor': '0.35',
                'markerfacecolor': 'none',
                'linestyle': 'none',
                'label': 'Censored data',
                'alpha': 0.70
            }
            raw_symbols.update(raw_kwds)
            osm_raw, osr_raw = stats.probplot(self.data['res'], fit=False)
            ax.plot(osm_raw, osr_raw, **raw_symbols)
            ax.legend(**leg_params)

        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Observations')
        if ylog:
            ax.set_yscale('log')

        return ax


#### Utils ####
def _subtract(self, estimate):
    class_name = self.__class__.__name__
    doc_string = """
        Subtract the %s of two %s objects.

        Parameters:
          other: an %s fitted instance.

        """ % (estimate, class_name, class_name)

    def subtract(other):
        self_estimate = getattr(self, estimate)
        other_estimate = getattr(other, estimate)
        return self_estimate.reindex(other_estimate.index, method='ffill') - \
            other_estimate.reindex(self_estimate.index, method='ffill')

    subtract.__doc__ = doc_string
    return subtract


def _divide(self, estimate):
    class_name = self.__class__.__name__
    doc_string = """
        Divide the %s of two %s objects.

        Parameters:
          other: an %s fitted instance.

        """ % (estimate, class_name, class_name)

    def divide(other):
        self_estimate = getattr(self, estimate)
        other_estimate = getattr(other, estimate)
        return self_estimate.reindex(other_estimate.index, method='ffill') / \
            other_estimate.reindex(self_estimate.index, method='ffill')

    divide.__doc__ = doc_string
    return divide


def _predict(self, estimate, label):
    doc_string =       """
      Predict the %s at certain times

      Parameters:
        time: an array of times to predict the value of %s at
      """ % (estimate, estimate)

    def predict(time):
        return map(lambda t: getattr(self, estimate).ix[:t].iloc[-1][label], time)

    predict.__doc__ = doc_string
    return predict


def preprocess_inputs(durations, event_observed, timeline, entry ):

    n = len(durations)
    durations = np.asarray(durations).reshape((n,))

    # set to all observed if event_observed is none
    if event_observed is None:
        event_observed = np.ones(n, dtype=int)
    else:
        event_observed = np.asarray(event_observed).reshape((n,)).copy().astype(int)

    if entry is None:
        entry = np.zeros(n)
    else:
        entry = np.asarray(entry).reshape((n,))

    event_table = survival_table_from_events(durations, event_observed, entry)

    if timeline is None:
        timeline = event_table.index.values
    else:
        timeline = np.asarray(timeline)

    return durations, event_observed, timeline.astype(float), entry, event_table


def _additive_estimate(events, timeline, _additive_f, _additive_var):
    """
    Called to compute the Kaplan Meier and Nelson-Aalen estimates.

    """

    deaths = events['observed']
    population = events['entrance'].cumsum() - events['removed'].cumsum().shift(1).fillna(0) #slowest line here.
    estimate_ = np.cumsum(_additive_f(population, deaths))
    var_ = np.cumsum(_additive_var(population, deaths))

    estimate_ = estimate_.reindex(timeline, method='pad').fillna(0)
    var_ = var_.reindex(timeline, method='pad')
    var_.index.name = 'timeline'
    estimate_.index.name = 'timeline'

    return estimate_, var_


def qth_survival_times(q, survival_functions):
    """
    This can be done much better.

    Parameters:
      q: a float between 0 and 1.
      survival_functions: a (n,d) dataframe or numpy array.
        If dataframe, will return index values (actual times)
        If numpy array, will return indices.

    Returns:
      v: an array containing the first times the value was crossed.
        np.inf if infinity.
    """
    assert 0. <= q <= 1., "q must be between 0. and 1."
    sv_b = (1.0 * (survival_functions < q)).cumsum() > 0
    try:
        v = sv_b.idxmax(0)
        v[sv_b.iloc[-1,:] == 0] = np.inf
    except:
        v = sv_b.argmax(0)
        v[sv_b[-1,:] == 0] = np.inf
    return v


def median_survival_times(survival_functions):
    return qth_survival_times(0.5, survival_functions)


def asymmetric_epanechnikov_kernel(q, x):
    return (64 * (2 - 4 * q + 6 * q * q - 3 * q ** 3) + 240 * (1 - q) ** 2 * x) / ((1 + q) ** 4 * (19 - 18 * q + 3 * q ** 2))


def _ros_sort(dataframe, result_col='res', censorship_col='cen'):
    '''
    This function prepares a dataframe for ROS. It sorts ascending with
    left-censored observations on top.

    Parameters
    ----------
    dataframe : a pandas dataframe with results and qualifiers.
        The qualifiers of the dataframe must have two states:
        detect and non-detect.
    result_col (default = 'res') : name of the column in the dataframe
        that contains result values.
    censorship_col(default = 'cen') : name of the column in the dataframe
        that indicates that a result is left-censored.
        (i.e., True -> censored, False -> uncensored)

    Output
    ------
    Sorted pandas DataFrame.
    '''
    # separate detects from non-detects
    nondetects = dataframe[dataframe[censorship_col]].sort(columns=result_col)
    detects = dataframe[~dataframe[censorship_col]].sort(columns=result_col)

    return nondetects.append(detects)

"""
References:
[1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

"""
