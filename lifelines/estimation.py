import numpy as np
from numpy.linalg import LinAlgError, inv, pinv
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
                self.censorship.shape[0], (1-self.censorship).sum())
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
                self.censorship.shape[0], (1-self.censorship).sum())
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
        naf.fit(durations, censorship=censorship, timeline=timeline, label=label, entry=entry)
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
        assert penalizer >= 0, "penalizer must be >= 0."


    def fit(self, dataframe, duration_col="T", event_col="E", timeline=None, id_col=None):
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


        Returns:
          self, with new methods like plot, smoothed_hazards_ and properties like cumulative_hazards_
        """

        df = dataframe.copy()
        if id_col is None:
            #only for time-indp. covariates
            df['id'] = np.arange(df.shape[0])
            id_col = 'id'

        if self.fit_intercept:
            df['baseline'] = 1.

        df = df.set_index([id_col, duration_col])
  
        C_panel = df[[event_col]].to_panel().transpose(1,2,0)
        C = C_panel.minor_xs(event_col).sum().astype(bool)
        T = (C_panel.minor_xs(event_col).notnull()).cumsum().idxmax()

        del df[event_col]
        n,d = df.shape

        from_tuples = pd.MultiIndex.from_tuples
        wp = df.to_panel().transpose(1,2,0).bfill().fillna(0) #bfill will cause problems later

        non_censorsed_times = T[C].iteritems()
        hazards_ = pd.DataFrame( np.zeros((len(non_censorsed_times),d)), 
                        columns = df.columns, index = from_tuples(non_censorsed_times))

        variance_  = pd.DataFrame( np.zeros((len(non_censorsed_times),d)), 
                        columns = df.columns, index = from_tuples(non_censorsed_times))
        
        ids = wp.items
        penalizer = self.penalizer*np.eye(d)
        
        for i,(id, time) in enumerate(non_censorsed_times): 

            relevant_individuals = (ids==id)
            assert relevant_individuals.sum() == 1.

            X = wp.major_xs(time).T.values

            #perform linear regression step.
            try:
                V = np.dot(pinv(np.dot(X.T, X) + penalizer), X.T)
            except LinAlgError:
                print "Linear regression error. Try increasing the penalizer."
            v = np.dot(V, 1.0*relevant_individuals )

            hazards_.ix[id, time]  = v.T
            variance_.ix[id, time] = np.dot( V[:, relevant_individuals], V[:, relevant_individuals].T ).diagonal()

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
        self.event_occured = C
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

def preprocess_inputs(durations, censorship, timeline, entry ):

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
    population = events['entrance'].cumsum() - events['removed'].cumsum().shift(1).fillna(0)
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

"""
References:
[1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

"""
