# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from numpy.linalg import solve, norm, inv, LinAlgError
from numpy import dot, exp
from numpy.random import beta
from scipy.integrate import trapz
import scipy.stats as stats
import pandas as pd

from lifelines.plotting import plot_estimate, plot_regressions
from lifelines.utils import survival_table_from_events, inv_normal_cdf, \
    epanechnikov_kernel, StatError, coalesce, normalize, significance_code
from lifelines.utils import ridge_regression as lr
from lifelines.progress_bar import progress_bar
from lifelines.utils import concordance_index


class BaseFitter(object):

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifelines.%s: fitted with %d observations, %d censored>""" % (
                classname, self.event_observed.shape[0], (1 - self.event_observed).sum())
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s


class NelsonAalenFitter(BaseFitter):

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

    def fit(self, durations, event_observed=None, timeline=None, entry=None,
            label='NA_estimate', alpha=None, ci_labels=None):
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
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>

        Returns:
          self, with new properties like 'cumulative_hazard_'.

        """

        v = preprocess_inputs(durations, event_observed, timeline, entry)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v

        cumulative_hazard_, cumulative_sq_ = _additive_estimate(self.event_table, self.timeline,
                                                                self._additive_f, self._variance_f, False)

        # esimates
        self._label = label
        self.cumulative_hazard_ = pd.DataFrame(cumulative_hazard_, columns=[self._label])
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha if alpha else self.alpha, ci_labels)
        self._cumulative_sq = cumulative_sq_

        # estimation functions
        self.predict = _predict(self, "cumulative_hazard_", self._label)
        self.subtract = _subtract(self, "cumulative_hazard_")
        self.divide = _divide(self, "cumulative_hazard_")

        # plotting
        self.plot = plot_estimate(self, "cumulative_hazard_")
        self.plot_cumulative_hazard = self.plot
        self.plot_hazard = plot_estimate(self, 'hazard_')

        return self

    def _bounds(self, cumulative_sq_, alpha, ci_labels):
        alpha2 = inv_normal_cdf(1 - (1 - alpha) / 2)
        df = pd.DataFrame(index=self.timeline)

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, self.alpha), "%s_lower_%.2f" % (self._label, self.alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."
        self.ci_labels = ci_labels

        df[ci_labels[0]] = self.cumulative_hazard_.values * \
            np.exp(alpha2 * np.sqrt(cumulative_sq_) / self.cumulative_hazard_.values)
        df[ci_labels[1]] = self.cumulative_hazard_.values * \
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
        hazard_name = "differenced-" + cumulative_hazard_name
        hazard_ = self.cumulative_hazard_.diff().fillna(self.cumulative_hazard_.iloc[0])
        C = (hazard_[cumulative_hazard_name] != 0.0).values
        return pd.DataFrame(1. / (2 * bandwidth) * np.dot(epanechnikov_kernel(timeline[:, None], timeline[C][None,:], bandwidth), hazard_.values[C,:]),
                            columns=[hazard_name], index=timeline)

    def smoothed_hazard_confidence_intervals_(self, bandwidth, hazard_=None):
        """
        Parameter:
          bandwidth: the bandwith to use in the Epanechnikov kernel.
          hazard_: a computed (n,) numpy array of estimated hazard rates. If none, uses naf.smoothed_hazard_
        """
        if hazard_ is None:
            hazard_ = self.smoothed_hazard_(bandwidth).values[:, 0]

        timeline = self.timeline
        alpha2 = inv_normal_cdf(1 - (1 - self.alpha) / 2)
        self._cumulative_sq.iloc[0] = 0
        var_hazard_ = self._cumulative_sq.diff().fillna(self._cumulative_sq.iloc[0])
        C = (var_hazard_.values != 0.0)  # only consider the points with jumps
        std_hazard_ = np.sqrt(1. / (2 * bandwidth ** 2) * np.dot(epanechnikov_kernel(timeline[:, None], timeline[C][None,:], bandwidth) ** 2, var_hazard_.values[C]))
        values = {
            self.ci_labels[0]: hazard_ * np.exp(alpha2 * std_hazard_ / hazard_),
            self.ci_labels[1]: hazard_ * np.exp(-alpha2 * std_hazard_ / hazard_)
        }
        return pd.DataFrame(values, index=timeline)


class KaplanMeierFitter(BaseFitter):

    """
    Class for fitting the Kaplan-Meier estimate for the survival function.

    KaplanMeierFitter( alpha=0.95)

    alpha: The alpha value associated with the confidence intervals.

    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def fit(self, durations, event_observed=None, timeline=None, entry=None, label='KM_estimate',
            alpha=None, left_censorship=False, ci_labels=None):
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
          left_censorship: True if durations and event_observed refer to left censorship events. Default False
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>


        Returns:
          self, with new properties like 'survival_function_'.

        """
        # if the user is interested in left-censorship, we return the cumulative_density_, no survival_function_,
        estimate_name = 'survival_function_' if not left_censorship else 'cumulative_density_'
        v = preprocess_inputs(durations, event_observed, timeline, entry)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v
        self._label = label
        self.alpha = alpha if alpha else self.alpha
        log_survival_function, cumulative_sq_ = _additive_estimate(self.event_table, self.timeline,
                                                                   self._additive_f, self._additive_var,
                                                                   left_censorship)

        if entry is not None:
            # a serious problem with KM is that when the sample size is small and there are too few early
            # truncation times, it may happen that is the number of patients at risk and the number of deaths is the same.
            # we adjust for this using the Breslow-Fleming-Harrington estimator
            n = self.event_table.shape[0]
            net_population = (self.event_table['entrance'] - self.event_table['removed']).cumsum()
            if net_population.iloc[:int(n / 2)].min() == 0:
                ix = net_population.iloc[:int(n / 2)].argmin()
                raise StatError("""There are too few early truncation times and too many events. S(t)==0 for all t>%.1f. Recommend BreslowFlemingHarringtonFitter.""" % ix)

        # estimation
        setattr(self, estimate_name, pd.DataFrame(np.exp(log_survival_function), columns=[self._label]))
        self.__estimate = getattr(self, estimate_name)
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], ci_labels)
        self.median_ = median_survival_times(self.__estimate)

        # estimation methods
        self.predict = _predict(self, estimate_name, label)
        self.subtract = _subtract(self, estimate_name)
        self.divide = _divide(self, estimate_name)

        # plotting functions
        self.plot = plot_estimate(self, estimate_name)
        setattr(self, "plot_" + estimate_name, self.plot)
        return self

    def _bounds(self, cumulative_sq_, ci_labels):
        # See http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes2.pdfg
        alpha2 = inv_normal_cdf((1. + self.alpha) / 2.)
        df = pd.DataFrame(index=self.timeline)
        v = np.log(self.__estimate.values)

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, self.alpha), "%s_lower_%.2f" % (self._label, self.alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = np.exp(-np.exp(np.log(-v) + alpha2 * np.sqrt(cumulative_sq_) / v))
        df[ci_labels[1]] = np.exp(-np.exp(np.log(-v) - alpha2 * np.sqrt(cumulative_sq_) / v))
        return df

    def _additive_f(self, population, deaths):
        np.seterr(invalid='ignore')
        return (np.log(population - deaths) - np.log(population))

    def _additive_var(self, population, deaths):
        np.seterr(divide='ignore')
        return (1. * deaths / (population * (population - deaths))).replace([np.inf], 0)

    def conditional_time_to(self):
        """
        Return a DataFrame, with index equal to survival_function_, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        indivual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.

        Returns:
            conditional_time_to_: DataFrame, with index equal to survival_function_

        """
        age = self.survival_function_.index.values[:, None]
        columns = ['%s - Conditional time remaining to event' % self._label]
        return pd.DataFrame(qth_survival_times(self.survival_function_[self._label] * 0.5, self.survival_function_).T.sort(ascending=False).values,
                            index=self.survival_function_.index,
                            columns=columns) - age


class BreslowFlemingHarringtonFitter(BaseFitter):

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

    def fit(self, durations, event_observed=None, timeline=None, entry=None,
            label='BFH_estimate', alpha=None, ci_labels=None):
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
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>


        Returns:
          self, with new properties like 'survival_function_'.

        """
        naf = NelsonAalenFitter(self.alpha)
        naf.fit(durations, event_observed=event_observed, timeline=timeline, label=label, entry=entry, ci_labels=ci_labels)
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


class BayesianFitter(BaseFitter):

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

    def __init__(self, samples=300):
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
        kwargs['alpha'] = coalesce(kwargs.pop('alpha', None), 0.05)
        kwargs['legend'] = False
        kwargs['c'] = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), '#348ABD')
        ax = self.sample_survival_functions_.plot(**kwargs)
        return ax

    def generate_sample_path(self, n=1):
        deaths = self.event_table['observed']
        population = self.event_table['entrance'].cumsum() - self.event_table['removed'].cumsum().shift(1).fillna(0)
        d = deaths.shape[0]
        samples = 1. - beta(0.01 + deaths, 0.01 + population - deaths, size=(n, d))
        sample_paths = pd.DataFrame(np.exp(np.log(samples).cumsum(1)).T, index=self.timeline)
        return sample_paths


class AalenAdditiveFitter(BaseFitter):

    """
    This class fits the regression model:

    hazard(t)  = b_0(t) + b_t(t)*x_1 + ... + b_N(t)*x_N

    that is, the hazard rate is a linear function of the covariates.

    Parameters:
      fit_intercept: If False, do not attach an intercept (column of ones) to the covariate matrix. The
        intercept, b_0(t) acts as a baseline hazard.
      alpha: the level in the confidence intervals.
      penalizer: Attach a L2 penalizer to the regression. This improves stability of the estimates
       and controls high correlation between covariates. Recommended, even if a small value.

    """

    def __init__(self, fit_intercept=True, alpha=0.95, coef_penalizer=0.5, smoothing_penalizer=0.):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_penalizer = coef_penalizer
        self.smoothing_penalizer = smoothing_penalizer
        assert coef_penalizer >= 0 and smoothing_penalizer >= 0, "penalizer must be >= 0."

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
            max_unique_durations: memory can be an issue if there are too many
              unique durations. If the max is surpassed, max_unique_durations bins
              will be used.

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
        self.plot = plot_regressions(self)

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
        self.plot = plot_regressions(self)

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

        cols = get_index(X)
        if isinstance(X, pd.DataFrame):
            order = self.cumulative_hazards_.columns
            order = order.drop('baseline') if self.fit_intercept else order
            X_ = X[order].values.copy()
        else:
            X_ = X.copy()
        X_ = X_ if not self.fit_intercept else np.c_[X_, np.ones((n, 1))]
        return pd.DataFrame(np.dot(self.cumulative_hazards_, X_.T), index=self.timeline, columns=cols)

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
        index = get_index(X)
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
        index = get_index(X)
        t = self.cumulative_hazards_.index
        return pd.DataFrame(trapz(self.predict_survival_function(X)[index].values.T, t), index=index)

    def predict(self, X):
        return self.predict_median(X)


class CoxPHFitter(BaseFitter):

    """
    This class implements fitting Cox's proportional hazard model:

    h(t|x) = h_0(t)*exp(x'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      tie_method: specify how the fitter should deal with ties. Currently only
        'Efron' is available.
      normalize: substract the mean and divide by standard deviation of each covariate
        in the input data before performing any fitting.
    """

    def __init__(self, alpha=0.95, tie_method='Efron', normalize=True):
        self.alpha = alpha
        self.normalize = normalize
        if tie_method != 'Efron':
            raise NotImplementedError("Only Efron is available atm.")
        self.tie_method = tie_method

    def _get_efron_values(self, X, beta, T, E, include_likelihood=False):
        """
        Calculates the first and second order vector differentials,
        with respect to beta. If 'include_likelihood' is True, then
        the log likelihood is also calculated. This is omitted by default
        to speed up the fit.

        Note that X, T, E are assumed to be sorted on T!

        Parameters:
            X: (n,d) numpy array of observations.
            beta: (1, d) numpy array of coefficients.
            T: (n) numpy array representing observed durations.
            E: (n) numpy array representing death events.

        Returns:
            hessian: (d, d) numpy array,
            gradient: (1, d) numpy array
            log_likelihood: double, if include_likelihood=True
        """

        n, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros((1, d))
        log_lik = 0

        # Init risk and tie sums to zero
        x_tie_sum = np.zeros((1, d))
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))
        risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

        # Init number of ties
        tie_count = 0

        # Iterate backwards to utilize recursive relationship
        for i, (ti, ei) in reversed(list(enumerate(zip(T, E)))):
            # Doing it like this to preserve shape
            xi = X[i:i + 1]
            # Calculate phi values
            phi_i = exp(dot(xi, beta))
            phi_x_i = dot(phi_i, xi)
            phi_x_x_i = np.dot(xi.T, xi) * phi_i

            # Calculate sums of Risk set
            risk_phi += phi_i
            risk_phi_x += phi_x_i
            risk_phi_x_x += phi_x_x_i
            # Calculate sums of Ties, if this is an event
            if ei:
                x_tie_sum += xi
                tie_phi += phi_i
                tie_phi_x += phi_x_i
                tie_phi_x_x += phi_x_x_i

                # Keep track of count
                tie_count += 1

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                # Only censored with current time, move on
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            partial_gradient = np.zeros((1, d))

            for l in range(tie_count):
                c = l / tie_count

                denom = (risk_phi - c * tie_phi)
                z = (risk_phi_x - c * tie_phi_x)

                if denom == 0:
                    # Can't divide by zero
                    raise ValueError("Denominator was zero")

                # Gradient
                partial_gradient += z / denom
                # Hessian
                a1 = (risk_phi_x_x - c * tie_phi_x_x) / denom
                # In case z and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = dot(z.T / denom, z / denom)

                hessian -= (a1 - a2)

                if include_likelihood:
                    log_lik -= np.log(denom)

            # Values outside tie sum
            gradient += x_tie_sum - partial_gradient
            if include_likelihood:
                log_lik += dot(x_tie_sum, beta).ravel()

            # reset tie values
            tie_count = 0
            x_tie_sum = np.zeros((1, d))
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            tie_phi_x_x = np.zeros((d, d))

        if include_likelihood:
            return hessian, gradient, log_lik.ravel()[0]
        else:
            return hessian, gradient

    def _newton_rhaphson(self, X, T, E, initial_beta=None, step_size=1.,
                         epsilon=10e-5, show_progress=True, include_likelihood=False):
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            X: (n,d) numpy array of observations.
            T: (n) numpy array representing observed durations.
            E: (n) numpy array representing death events.
            initial_beta: (1,d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0.001 to determine a starting step size in NR algorithm.
            epsilon: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.
            include_likelihood: saves the final log-likelihood to the CoxPHFitter under _log_likelihood.

        Returns:
            beta: (1,d) numpy array.
        """
        assert epsilon <= 1., "epsilon must be less than or equal to 1."
        n, d = X.shape

        # Enforce numpy arrays
        X = np.array(X)
        T = np.array(T)
        E = np.array(E)

        # Want as bools
        E = E.astype(bool)

        # make sure betas are correct size.
        if initial_beta is not None:
            assert initial_beta.shape == (d, 1)
            beta = initial_beta
        else:
            beta = np.zeros((d, 1))

        # Method of choice is just efron right now
        if self.tie_method == 'Efron':
            get_gradients = self._get_efron_values
        else:
            raise NotImplementedError("Only Efron is available atm.")

        i = 1
        converging = True
        # 50 iterations steps with N-R is a lot.
        # Expected convergence is ~10 steps
        while converging and i < 50 and step_size > 0.001:
            output = get_gradients(X, beta, T, E,
                                   include_likelihood=include_likelihood)
            # Do not override hessian and gradient in case of garbage
            h, g = output[:2]

            delta = solve(-h, step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ValueError("delta contains nan value(s). Convergence halted.")
            # Only allow small steps
            if norm(delta) > 10:
                step_size *= 0.5
                continue

            beta += delta
            # Save these as pending result
            hessian, gradient = h, g

            if norm(delta) < epsilon:
                converging = False

            if ((i % 10) == 0) and show_progress:
                print("Iteration %d: delta = %.5f" % (i, norm(delta)))
            i += 1

        self._hessian_ = hessian
        self._score_ = gradient
        if include_likelihood:
            self._log_likelihood = output[2]
        if show_progress:
            print("Convergence completed after %d iterations." % (i))
        return beta

    def fit(self, df, duration_col, event_col=None,
            show_progress=False, initial_beta=None, include_likelihood=False):
        """
        Fit the Cox Propertional Hazard model to a dataset. Tied survival times
        are handled using Efron's tie-method.

        Parameters:
          df: a Pandas dataframe with necessary columns `duration_col` and
             `event_col`, plus other covariates. `duration_col` refers to
             the lifetimes of the subjects. `event_col` refers to whether
             the 'death' events was observed: 1 if observed, 0 else (censored).
          duration_col: the column in dataframe that contains the subjects'
             lifetimes.
          event_col: the column in dataframe that contains the subjects' death
             observation. If left as None, assume all individuals are non-censored.
          show_progress: since the fitter is iterative, show convergence
             diagnostics.
          initial_beta: initialize the starting point of the iterative
             algorithm. Default is the zero vector.
          include_likelihood: saves the final log-likelihood to the CoxPHFitter under
             the property _log_likelihood.


        Returns:
            self, with additional properties: hazards_

        """
        df = df.copy()
        # Sort on time
        df.sort(duration_col, inplace=True)

        # Extract time and event
        T = df[duration_col]
        del df[duration_col]
        if event_col is None:
            E = pd.Series(np.ones(df.shape[0]), index=df.index)
        else:
            E = df[event_col]
            del df[event_col]

        # Store original non-normalized data
        self.data = df

        if self.normalize:
            # Need to normalize future inputs as well
            self._norm_mean = df.mean(0)
            self._norm_std = df.std(0)
            df = normalize(df)

        E = E.astype(bool)
        self._check_values(df)

        hazards_ = self._newton_rhaphson(df, T, E, initial_beta=initial_beta,
                                         show_progress=show_progress,
                                         include_likelihood=include_likelihood)

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns,
                                     index=['coef'])
        self.confidence_intervals_ = self._compute_confidence_intervals()

        self.durations = T
        self.event_observed = E

        self.baseline_hazard_ = self._compute_baseline_hazard()
        self.baseline_cumulative_hazard_ = self.baseline_hazard_.cumsum()
        self.baseline_survival_ = exp(-self.baseline_cumulative_hazard_)
        return self

    def _check_values(self, X):
        low_var = (X.var(0) < 10e-5)
        if low_var.any():
            cols = str(list(X.columns[low_var]))
            print("Warning: column(s) %s have very low variance.\
 This may harm convergence." % cols)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1. + self.alpha) / 2.)
        se = self._compute_standard_errors()
        hazards = self.hazards_.values
        return pd.DataFrame(np.r_[hazards - alpha2 * se,
                                  hazards + alpha2 * se],
                            index=['lower-bound', 'upper-bound'],
                            columns=self.hazards_.columns)

    def _compute_standard_errors(self):
        se = np.sqrt(inv(-self._hessian_).diagonal())
        return pd.DataFrame(se[None,:],
                            index=['se'], columns=self.hazards_.columns)

    def _compute_z_values(self):
        return (self.hazards_.ix['coef'] /
                self._compute_standard_errors().ix['se'])

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper"""

        df = pd.DataFrame(index=self.hazards_.columns)
        df['coef'] = self.hazards_.ix['coef'].values
        df['exp(coef)'] = exp(self.hazards_.ix['coef'].values)
        df['se(coef)'] = self._compute_standard_errors().ix['se'].values
        df['z'] = self._compute_z_values()
        df['p'] = self._compute_p_values()
        df['lower %.2f' % self.alpha] = self.confidence_intervals_.ix['lower-bound'].values
        df['upper %.2f' % self.alpha] = self.confidence_intervals_.ix['upper-bound'].values
        return df

    def print_summary(self, c_index=True):
        """
        Print summary statistics describing the fit.

        c_index: If set to False, will not print the concordance index

        """
        df = self.summary
        # Significance codes last
        df[''] = [significance_code(p) for p in df['p']]

        # Print information about data first
        print('n={}, number of events={}'.format(self.data.shape[0],
                                                 np.where(self.event_observed)[0].shape[0]),
              end='\n\n')
        print(df.to_string(float_format=lambda f: '{:.3e}'.format(f)))
        # Significance code explanation
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ",
              end='\n\n')
        if c_index:
            print("Concordance = {:.3f}"
                  .format(concordance_index(self.durations,
                                            -self.predict_partial_hazard(self.data).values.ravel(),
                                            self.event_observed)))
        return

    def predict_partial_hazard(self, X):
        """
        X: a (n,d) covariate matrix

        If covariates were normalized during fitting, they are normalized
        in the same way here.

        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to \exp{\beta X}
        """
        index = get_index(X)

        if isinstance(X, pd.DataFrame):
            order = self.hazards_.columns
            X = X[order]

        if self.normalize:
            # Assuming correct ordering and number of columns
            X = normalize(X, self._norm_mean.values, self._norm_std.values)

        return pd.DataFrame(exp(np.dot(X, self.hazards_.T)), index=index)

    def predict_cumulative_hazard(self, X):
        """
        X: a (n,d) covariate matrix

        Returns the cumulative hazard for the individuals.
        """
        v = self.predict_partial_hazard(X)
        s_0 = self.baseline_survival_
        col = get_index(X)
        return pd.DataFrame(-np.dot(np.log(s_0), v.T), index=self.baseline_survival_.index, columns=col)

    def predict_survival_function(self, X):
        """
        X: a (n,d) covariate matrix

        Returns the survival functions for the individuals
        """
        return exp(-self.predict_cumulative_hazard(X))

    def predict_percentile(self, X, p=0.5):
        """
        X: a (n,d) covariate matrix
        Returns the median lifetimes for the individuals.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions
        """
        index = get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X)[index])

    def predict_median(self, X):
        """
        X: a (n,d) covariate matrix
        Returns the median lifetimes for the individuals
        """
        return self.predict_percentile(X, 0.5)

    def predict_expectation(self, X):
        """
        Compute the expected lifetime, E[T], using covarites X.
        """
        index = get_index(X)
        v = self.predict_survival_function(X)[index]
        return pd.DataFrame(trapz(v.values.T, v.index), index=index)

    def predict(self, X):
        return self.predict_median(X)

    def _compute_baseline_hazard(self):
        # http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
        ind_hazards = self.predict_partial_hazard(self.data).values

        event_table = survival_table_from_events(self.durations.values,
                                                 self.event_observed.values)

        baseline_hazard_ = pd.DataFrame(np.zeros((event_table.shape[0], 1)),
                                        index=event_table.index,
                                        columns=['baseline hazard'])

        for t, s in event_table.iterrows():
            less = np.array(self.durations >= t)
            if ind_hazards[less].sum() == 0:
                v = 0
            else:
                v = (s['observed'] / ind_hazards[less].sum())
            baseline_hazard_.ix[t] = v

        return baseline_hazard_


def get_index(X):
    if isinstance(X, pd.DataFrame):
        index = list(X.index)
    else:
        # If it's not a dataframe, order is up to user
        index = list(range(X.shape[0]))
    return index


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
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        return self_estimate.reindex(new_index, method='ffill') - \
            other_estimate.reindex(new_index, method='ffill')

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
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        return self_estimate.reindex(new_index, method='ffill') / \
            other_estimate.reindex(new_index, method='ffill')

    divide.__doc__ = doc_string
    return divide


def _predict(self, estimate, label):
    doc_string = """
      Predict the %s at certain times

      Parameters:
        time: an array of times to predict the value of %s at
      """ % (estimate, estimate)

    def predict(time):
        return [getattr(self, estimate).ix[:t].iloc[-1][label] for t in time]

    predict.__doc__ = doc_string
    return predict


def preprocess_inputs(durations, event_observed, timeline, entry):

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


def _additive_estimate(events, timeline, _additive_f, _additive_var, reverse):
    """
    Called to compute the Kaplan Meier and Nelson-Aalen estimates.

    """
    if reverse:
        events = events.sort_index(ascending=False)
        population = events['entrance'].sum() - events['removed'].cumsum().shift(1).fillna(0)
        deaths = events['observed'].shift(1).fillna(0)
        estimate_ = np.cumsum(_additive_f(population, deaths)).ffill().sort_index()
        var_ = np.cumsum(_additive_var(population, deaths)).ffill().sort_index()
    else:
        deaths = events['observed']
        population = events['entrance'].cumsum() - events['removed'].cumsum().shift(1).fillna(0)  # slowest line here.
        estimate_ = np.cumsum(_additive_f(population, deaths))
        var_ = np.cumsum(_additive_var(population, deaths))

    timeline = sorted(timeline)
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
      v: if d==1, returns a float, np.inf if infinity.
         if d > 1, an DataFrame containing the first times the value was crossed.

    """
    q = pd.Series(q)
    assert (q <= 1).all() and (0 <= q).all(), 'q must be between 0 and 1'
    survival_functions = pd.DataFrame(survival_functions)
    if survival_functions.shape[1] == 1 and q.shape == (1,):
        return survival_functions.apply(lambda s: qth_survival_time(q[0], s)).ix[0]
    else:
        return pd.DataFrame({q_: survival_functions.apply(lambda s: qth_survival_time(q_, s)) for q_ in q})


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


def asymmetric_epanechnikov_kernel(q, x):
    return (64 * (2 - 4 * q + 6 * q * q - 3 * q ** 3) + 240 * (1 - q) ** 2 * x) / ((1 + q) ** 4 * (19 - 18 * q + 3 * q ** 2))

"""
References:
[1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

"""
