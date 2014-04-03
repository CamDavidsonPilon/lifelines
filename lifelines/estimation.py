import numpy as np
from numpy.linalg import LinAlgError, inv, pinv
from scipy.sparse.linalg import inv
from scipy import sparse
from numpy import dot
import pandas as pd

from lifelines.plotting import plot_estimate, plot_regressions
from lifelines.utils import survival_table_from_events, basis, inv_normal_cdf, quadrature, epanechnikov_kernel, StatError


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

    def fit(self, durations, censorship=None, timeline=None, entry=None, label='NA-estimate', alpha=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          censorship: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if censorship==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.

        Returns:
          self, with new properties like 'cumulative_hazard_'.

        """

        v = preprocess_inputs(durations, censorship, timeline, entry)
        self.durations, self.censorship, self.timeline, self.entry, self.event_table = v

        cumulative_hazard_, cumulative_sq_ = _additive_estimate(
            self.event_table, self.timeline,
            self._additive_f, self._variance_f)

        # esimates
        self.cumulative_hazard_ = pd.DataFrame(
            cumulative_hazard_, columns=[label])
        self.confidence_interval_ = self._bounds(
            cumulative_sq_[:, None], alpha if alpha else self.alpha)
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
            np.exp(alpha2 * np.sqrt(cumulative_sq_)
                   / self.cumulative_hazard_.values)
        df["%s_lower_%.2f" % (name, self.alpha)] = self.cumulative_hazard_.values * \
            np.exp(-alpha2 * np.sqrt(cumulative_sq_)
                   / self.cumulative_hazard_.values)
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
        hazard_ = self.cumulative_hazard_.diff().fillna(
            self.cumulative_hazard_.iloc[0])
        C = (hazard_[cumulative_hazard_name] != 0.0).values
        return pd.DataFrame(
            1. / (2 * bandwidth) * np.dot(epanechnikov_kernel(
                timeline[
                    :, None], timeline[
                        C][None, :], bandwidth), hazard_.values[C, :]),
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
        var_hazard_ = self._cumulative_sq.diff().fillna(
            self._cumulative_sq.iloc[0])
        C = (var_hazard_.values != 0.0)  # only consider the points with jumps
        std_hazard_ = np.sqrt(1. / (2 * bandwidth ** 2) * np.dot(
            epanechnikov_kernel(timeline[:, None], timeline[C][None, :], bandwidth) ** 2, var_hazard_.values[C]))
        values = {
            "%s_upper_%.2f" % (name, self.alpha): hazard_ * np.exp(alpha2 * std_hazard_ / hazard_),
            "%s_lower_%.2f" % (name, self.alpha): hazard_ * np.exp(-alpha2 * std_hazard_ / hazard_)
        }
        return pd.DataFrame(values, index=timeline)

    def __repr__(self):
        try:
            s = """<lifelines.NelsonAalenFitter: fitted with %d observations, %d censored>""" % (
                self.censorship.shape[0], (1 - self.censorship).sum())
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

    def fit(self, durations, censorship=None, timeline=None, entry=None, label='KM-estimate', alpha=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          censorship: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if censorship==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.

        Returns:
          self, with new properties like 'survival_function_'.

        """
        v = preprocess_inputs(durations, censorship, timeline, entry)
        self.durations, self.censorship, self.timeline, self.entry, self.event_table = v

        log_survival_function, cumulative_sq_ = _additive_estimate(
            self.event_table, self.timeline,
            self._additive_f, self._additive_var)

        if entry is not None:
            # a serious problem with KM is that when the sample size is small and there are too few early
            # truncation times, it may happen that is the number of patients at risk and the number of deaths is the same.
            # we adjust for this using the Breslow-Fleming-Harrington estimator
            n = self.event_table.shape[0]
            net_population = (
                self.event_table['entrance'] - self.event_table['removed']).cumsum()
            if net_population.iloc[:int(n / 2)].min() == 0:
                ix = net_population.iloc[:int(n / 2)].argmin()
                raise StatError(
                    """There are too few early truncation times and too many events. S(t)==0 for all t>%.1f. Recommend BFH estimator.""" % ix)

        # estimation
        self.survival_function_ = pd.DataFrame(
            np.exp(log_survival_function), columns=[label])
        self.confidence_interval_ = self._bounds(
            cumulative_sq_[:, None], alpha if alpha else self.alpha)
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
        # See
        # http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes2.pdfg
        alpha2 = inv_normal_cdf((1. + alpha) / 2.)
        df = pd.DataFrame(index=self.timeline)
        name = self.survival_function_.columns[0]
        v = np.log(self.survival_function_.values)
        df["%s_upper_%.2f" % (name, self.alpha)] = np.exp(
            -np.exp(np.log(-v) + alpha2 * np.sqrt(cumulative_sq_) / v))
        df["%s_lower_%.2f" % (name, self.alpha)] = np.exp(
            -np.exp(np.log(-v) - alpha2 * np.sqrt(cumulative_sq_) / v))
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
                self.censorship.shape[0], (1 - self.censorship).sum())
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

    def fit(self, durations, censorship=None, timeline=None, entry=None, label='BFH-estimate', alpha=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          censorship: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if censorship==None
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
        naf.fit(durations, censorship=censorship,
                timeline=timeline, label=label, entry=entry)
        self.durations, self.censorship, self.timeline, self.entry, self.event_table = \
            naf.durations, naf.censorship, naf.timeline, naf.entry, naf.event_table

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

    def __init__(self, fit_intercept=True, alpha=0.95, penalizer=0.0):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.penalizer = penalizer
        assert penalizer >= 0, "penalizer keyword must be >= 0."

    def fit(self, event_times, X, timeline=None, censorship=None, columns=None):
        """currently X is a static (n,d) array

        event_times: (n,1) array of event times
        X: (n,d) the design matrix, either a numpy matrix or DataFrame.
        timeline: (t,1) timepoints in ascending order
        censorship: (n,1) boolean array of censorships: True if observed, False if right-censored.
                    By default, assuming all are observed.

        Fits: self.cumulative_hazards_: a (t,d+1) dataframe of cumulative hazard coefficients
              self.hazards_: a (t,d+1) dataframe of hazard coefficients

        """
        # deal with the covariate matrix. Check if it is a dataframe or numpy
        # array
        n, d = X.shape

        # append a columns of ones for the baseline hazard
        ix = event_times.argsort(0)[:, 0]
        baseline = sparse.csc_matrix(np.ones((n, 1)))
        X = X.tocsc()
        X = sparse.hstack([X_[ix, :], baseline])
        sorted_event_times = event_times[ix, 0]

        # set the column's names of the dataframe.
        if columns is None:
            columns = range(d)
        else:
            columns = [c for c in columns]

        if self.fit_intercept:
            columns += ['baseline']

        # set the censorship events. 1 if the death was observed.
        if censorship is None:
            observed = np.ones(n, dtype=bool)
        else:
            observed = censorship[ix].reshape(n)

        # set the timeline -- this is used as DataFrame index in the results
        if timeline is None:
            timeline = sorted_event_times

        timeline = np.unique(timeline.astype(float))
        if timeline[0] > 0:
            timeline = np.insert(timeline, 0, 0.)

        unique_times = np.unique(timeline)
        zeros = np.zeros((timeline.shape[0], d + self.fit_intercept))
        self.cumulative_hazards_ = pd.DataFrame(
            zeros.copy(), index=unique_times, columns=columns)
        self.hazards_ = pd.DataFrame(
            np.zeros((event_times.shape[0], d + self.fit_intercept)), index=event_times[:, 0], columns=columns)
        self._variance = pd.DataFrame(
            zeros.copy(), index=unique_times, columns=columns)

        # create the penalizer matrix for L2 regression
        penalizer = sparrse.csr_matrix(
            self.penalizer * np.eye(d + self.fit_intercept))

        t_0 = sorted_event_times[0]
        cum_v = np.zeros((d + self.fit_intercept, 1))
        v = cum_v.copy()
        for i, time in enumerate(sorted_event_times):
            relevant_times = (t_0 < timeline) * (timeline <= time)
            if observed[i] == 0:
                X_[i, :] = 0
            try:
                V = inv((X.T.dot(X)) + penalizer).dot(X.T)
            except LinAlgError:
                print("shit happened")

            v = V.dot(basis(n, i))
            cum_v = cum_v + v
            self.cumulative_hazards_.ix[relevant_times] = self.cumulative_hazards_.ix[
                relevant_times].values + cum_v.T
            self.hazards_.iloc[i] = self.hazards_.iloc[i].values + v.T
            self._variance.ix[relevant_times] = self._variance.ix[
                relevant_times].values + dot(V[:, i][:, None], V[:, i][None, :]).diagonal()
            t_0 = time
            X[i, :] = 0

        # clean up last iteration
        relevant_times = (timeline > time)
        self.hazards_.iloc[i] = v.T
        try:
            self.cumulative_hazards_.ix[relevant_times] = cum_v.T
            self._variance.ix[relevant_times] = dot(
                V[:, i][:, None], V[:, i][None, :]).diagonal()
        except:
            pass
        self.timeline = timeline
        self.X = X
        self.censorship = censorship
        self.event_times = event_times
        self._compute_confidence_intervals()
        self.plot = plot_regressions(self)
        return self

    def smoothed_hazards_(self, bandwith=1):
        """
        Using the epanechnikov kernel to smooth the hazard function, with sigma/bandwith

        """
        C = self.censorship.astype(bool)
        return pd.DataFrame(
            np.dot(epanechnikov_kernel(
                self.timeline[
                    :, None], self.timeline[
                        C], bandwith), self.hazards_.values[C, :]),
            columns=self.hazards_.columns, index=self.timeline)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf(1 - (1 - self.alpha) / 2)
        n = self.timeline.shape[0]
        d = self.cumulative_hazards_.shape[1]
        index = [['upper'] * n + ['lower'] * n,
                 np.concatenate([self.timeline, self.timeline])]
        self.confidence_intervals_ = pd.DataFrame(
            np.zeros((2 * n, d)), index=index, columns=self.cumulative_hazards_.columns)
        self.confidence_intervals_.ix['upper'] = self.cumulative_hazards_.values + \
            alpha2 * np.sqrt(self._variance.cumsum().values)
        self.confidence_intervals_.ix['lower'] = self.cumulative_hazards_.values - \
            alpha2 * np.sqrt(self._variance.cumsum().values)
        return

    def predict_cumulative_hazard(self, X, columns=None):
        """
        X: a (n,d) covariate matrix

        Returns the hazard rates for the individuals
        """
        n, d = X.shape

        baseline = sparse.csc_matrix(np.ones((n, 1)))
        X = X.tocsc()
        X = sparse.hstack([X_[ix, :], baseline])
        sparse_hazards = sparse.csr_matrix(self.cumulative_hazards_)
        pred_hazards = pred_hazards.dot(X.T)
        return pd.DataFrame(pred_hazards.A, index=self.timeline, columns=columns)

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

# utils


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


def preprocess_inputs(durations, censorship, timeline, entry):

    n = len(durations)
    durations = np.asarray(durations).reshape((n,))

    # set to all observed if censorship is none
    if censorship is None:
        censorship = np.ones(n, dtype=int)
    else:
        censorship = np.asarray(censorship).reshape((n,)).copy().astype(int)

    if entry is None:
        entry = np.zeros(n)
    else:
        entry = np.asarray(entry).reshape((n,))

    event_table = survival_table_from_events(durations, censorship, entry)

    if timeline is None:
        timeline = event_table.index.values.copy()
    else:
        timeline = np.asarray(timeline)

    return durations, censorship, timeline.astype(float), entry, event_table


def _additive_estimate(events, timeline, _additive_f, _additive_var):
    """
    Called to compute the Kaplan Meier and Nelson-Aalen estimates.

    """

    deaths = events['observed']
    population = events['entrance'].cumsum() - events[
        'removed'].cumsum().shift(1).fillna(0)
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
        v[sv_b.iloc[-1, :] == 0] = np.inf
    except:
        v = sv_b.argmax(0)
        v[sv_b[-1, :] == 0] = np.inf
    return v


def median_survival_times(survival_functions):
    return qth_survival_times(0.5, survival_functions)


def asymmetric_epanechnikov_kernel(q, x):
    return (64 * (2 - 4 * q + 6 * q * q - 3 * q ** 3) + 240 * (1 - q) ** 2 * x) / ((1 + q) ** 4 * (19 - 18 * q + 3 * q ** 2))

"""
References:
[1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

"""
