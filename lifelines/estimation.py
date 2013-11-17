ƒimport numpy as np
from numpy.linalg import LinAlgError, inv, pinv
from numpy import dot
import pandas as pd

from lifelines.plotting import plot_dataframes
from lifelines.utils import dataframe_from_events_censorship, basis, inv_normal_cdf, quadrature

import pdb

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
        if nelson_aalen_smoothing:
          self._variance_f = self._variance_f_smooth
          self._additive_f = self._additive_f_smooth
        else:
          self._variance_f = self._variance_f_discrete
          self._additive_f = self._additive_f_discrete

    def fit(self, event_times,censorship=None, timeline=None, columns=['NA-estimate'],alpha=None):
        """
        Parameters:
          event_times: an (n,1) array of times that the death event occured at 
          timeline: return the best estimate at the values in timelines (postively increasing)
          columns: a length 1 array to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only. 

        Returns:
          DataFrame with index either event_times or timelines (if not None), with
          values as the NelsonAalen estimate
        """
        
        if censorship is None:
           self.censorship = np.ones_like(event_times, dtype=bool) #why boolean?
        else:
           self.censorship = censorship.copy().astype(bool)
        self.event_times = dataframe_from_events_censorship(event_times, self.censorship)

        if alpha is None:
            alpha = self.alpha

        if timeline is None:
           self.timeline = self.event_times.index.values.copy().astype(float)
        else:
           self.timeline = timeline
        self.cumulative_hazard_, cumulative_sq_ = _additive_estimate(self.event_times, 
                                                                     self.timeline, self._additive_f,
                                                                      columns, self._variance_f )
        self.confidence_interval_ = self._bounds(cumulative_sq_,alpha)
        self.plot = plot_dataframes(self, "cumulative_hazard_")

        return

    def _bounds(self, cumulative_sq_, alpha):
        alpha2 = inv_normal_cdf(1 - (1-alpha)/2)
        df = pd.DataFrame( index=self.timeline)
        name = self.cumulative_hazard_.columns[0]
        df["%s_upper_%.2f"%(name,self.alpha)] = self.cumulative_hazard_.values*np.exp(alpha2*np.sqrt(cumulative_sq_)/self.cumulative_hazard_.values )
        df["%s_lower_%.2f"%(name,self.alpha)] = self.cumulative_hazard_.values*np.exp(-alpha2*np.sqrt(cumulative_sq_)/self.cumulative_hazard_.values )
        return df

    def _variance_f_smooth(self, N, d):
        if N==d==0:
            return 0
        return np.sum([1./(N-i)**2 for i in range(int(d))])

    def _variance_f_discrete(self, N, d):
        if N==d==0:
            return 0
        return 1.*(N-d)*d/N**3

    def _additive_f_smooth(self, N, d):
        if N==d==0:
          return 0
        return np.sum([1./(N-i) for i in range(int(d))])

    def _additive_f_discrete(self, N,d ):
       #check it 0
       if N==d==0:
          return 0
       return 1.*d/N


class KaplanMeierFitter(object):
   
  def __init__(self, alpha=0.95):
       self.alpha = alpha

  def fit(self, event_times, censorship=None, timeline=None, columns=['KM-estimate'], alpha=None):
       """
       Parameters:
         event_times: an (n,1) array of times that the death event occured at 
         timeline: return the best estimate at the values in timelines (postively increasing)
         censorship: an (n,1) array of booleans -- True if the the death was observed, False if the event 
            was lost (right-censored). Defaults all True if censorship==None
         columns: a length 1 array to name the column of the estimate.
         alpha: the alpha value in the confidence intervals. Overrides the initializing
            alpha for this call to fit only. 

       Returns:
         DataFrame with index either event_times or timelines (if not None), with
         values under column_name with the KaplanMeier estimate
       """
       #set to all observed if censorship is none
       if censorship is None:
          self.censorship = np.ones_like(event_times, dtype=bool) #why boolean?
       else:
          self.censorship = censorship.copy()

       if not alpha:
          alpha = self.alpha

       self.event_times = dataframe_from_events_censorship(event_times, self.censorship)

       if timeline is None:
          self.timeline = self.event_times.index.values.copy()
       else:
          self.timeline = timeline
       log_surivial_function, cumulative_sq_ = _additive_estimate(self.event_times, 
                                                                  self.timeline, self._additive_f, 
                                                                   columns, self._variance_f )
       self.survival_function_ = np.exp(log_surivial_function)
       self.median_ = median_survival_times(self.survival_function_)
       self.confidence_interval_ = self._bounds(cumulative_sq_,alpha)
       self.plot = plot_dataframes(self, "survival_function_")
       return self

  def _additive_f(self, N, d):
      if N==d==0:
        return 0
      return np.log(1 - 1.*d/N)

  def _bounds(self, cumulative_sq_, alpha):
      # See http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes2.pdfg
      alpha2 = inv_normal_cdf((1.+ alpha)/2.)
      df = pd.DataFrame( index=self.timeline)
      name = self.survival_function_.columns[0]
      v = np.log(self.survival_function_.values)
      df["%s_upper_%.2f"%(name,self.alpha)] =  np.exp(-np.exp(np.log(-v)+alpha2*np.sqrt(cumulative_sq_)/v))
      df["%s_lower_%.2f"%(name,self.alpha)] =  np.exp(-np.exp(np.log(-v)-alpha2*np.sqrt(cumulative_sq_)/v))
      return df

  def _variance_f(self, N, d):
     if N==d==0:
        return 0
     return 1.*d/(N*(N-d))


def _additive_estimate(event_times, timeline, additive_f, columns, variance_f):
    """

    nelson_aalen_smoothing: see section 3.1.3 in Survival and Event History Analysis

    """
    timeline = timeline.astype(float)
    if timeline[0] > 0:
       timeline = np.insert(timeline,0,0.)

    n = timeline.shape[0]
    _additive_estimate_ = pd.DataFrame(np.zeros((n,1)), index=timeline, columns=columns)
    _additive_var = pd.DataFrame(np.zeros((n,1)), index=timeline)

    N = event_times["removed"].sum()
    t_0 = 0

    _additive_estimate_.ix[(timeline<t_0)]= 0
    _additive_var.ix[(timeline<t_0)]=0

    v = 0
    v_sq = 0
    for t, removed, observed_deaths, missing in event_times.itertuples():
        times = (t_0<=timeline)*(timeline<t)
        _additive_estimate_.ix[times] = v  
        _additive_var.ix[times] = v_sq
        N -= missing
        v += additive_f(N,observed_deaths)
        v_sq += variance_f(N,observed_deaths)
        N -= observed_deaths
        t_0 = t
    _additive_estimate_.ix[(timeline>=t)]=v
    _additive_var.ix[(timeline>=t)]=v_sq
    return _additive_estimate_, _additive_var


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

  def __init__(self,fit_intercept=True, alpha=0.95, penalizer = 0.0):
    self.fit_intercept = fit_intercept
    self.alpha = alpha
    self.penalizer = penalizer
    assert penalizer >= 0, "penalizer keyword must be >= 0."

  def fit(self, event_times, X, timeline = None, censorship=None, columns=None):
    """currently X is a static (n,d) array

    event_times: (n,1) array of event times
    X: (n,d) the design matrix, either a numpy matrix or DataFrame.  
    timeline: (t,1) timepoints in ascending order
    censorship: (n,1) boolean array of censorships: True if observed, False if right-censored.
                By default, assuming all are observed.

    Fits: self.cumulative_hazards_: a (t,d+1) dataframe of cumulative hazard coefficients
          self.hazards_: a (t,d+1) dataframe of hazard coefficients

    """
    #deal with the covariate matrix. Check if it is a dataframe or numpy array
    n,d = X.shape
    if type(X)==pd.core.frame.DataFrame:
      X_ = X.values.copy()
      if columns is None:
        columns = X.columns
    else:
      X_ = X.copy()

    # append a columns of ones for the baseline hazard
    ix = event_times.argsort(0)[:,0].copy()
    X_ = X_[ix,:].copy() if not self.fit_intercept else np.c_[ X_[ix,:].copy(), np.ones((n,1)) ]
    sorted_event_times = event_times[ix,0].copy()

    #set the column's names of the dataframe.
    if columns is None:
      columns = range(d) 
    else:
      columns = [c for c in columns] 

    if self.fit_intercept:
      columns += ['baseline']

    #set the censorship events. 1 if the death was observed.
    if censorship is None:
        observed = np.ones(n, dtype=bool)
    else:
        observed = censorship[ix].reshape(n)

    #set the timeline -- this is used as DataFrame index in the results
    if timeline is None:
        timeline = sorted_event_times.copy()

    timeline = np.unique(timeline.astype(float))
    if timeline[0] > 0:
       timeline = np.insert(timeline,0,0.)
    
    unique_times = np.unique(timeline)
    zeros = np.zeros((timeline.shape[0],d+self.fit_intercept))
    self.cumulative_hazards_ = pd.DataFrame(zeros.copy() , index=unique_times, columns = columns)
    self.hazards_ = pd.DataFrame(np.zeros((event_times.shape[0],d+self.fit_intercept)), index=event_times[:,0], columns = columns)
    self._variance = pd.DataFrame(zeros.copy(), index=unique_times, columns = columns)
    
    #create the penalizer matrix for L2 regression
    penalizer = self.penalizer*np.eye(d + self.fit_intercept)
    
    t_0 = sorted_event_times[0]
    cum_v = np.zeros((d+self.fit_intercept,1))
    v = cum_v.copy()
    for i,time in enumerate(sorted_event_times):
        relevant_times = (t_0<timeline)*(timeline<=time)
        if observed[i] == 0:
            X_[i,:] = 0
        try:
            V = dot(inv(dot(X_.T,X_) + penalizer), X_.T)
        except LinAlgError:
            #if penalizer > 0, this should not occur. But sometimes it does...
            V = dot(pinv(dot(X_.T,X_) + penalizer), X_.T)

        v = dot(V, basis(n,i))
        cum_v = cum_v + v
        self.cumulative_hazards_.ix[relevant_times] = self.cumulative_hazards_.ix[relevant_times].values + cum_v.T
        self.hazards_.iloc[i] = self.hazards_.iloc[i].values + v.T
        self._variance.ix[relevant_times] = self._variance.ix[relevant_times].values + dot( V[:,i][:,None], V[:,i][None,:] ).diagonal()
        t_0 = time
        X_[i,:] = 0

    #clean up last iteration
    relevant_times = (timeline>time)
    self.cumulative_hazards_.ix[relevant_times] = cum_v.T
    self.hazards_.iloc[i] = v.T
    self._variance.ix[relevant_times] = dot( V[:,i][:,None], V[:,i][None,:] ).diagonal()
    self.timeline = timeline
    self.X = X
    self.censorship = censorship
    self._compute_confidence_intervals()
    return self

  def smoothed_hazards_(self, bandwith=1):
        """
        Using the gaussian kernel to smooth the hazard function, with sigma/bandwith

        """
        C = self.censorship.astype(bool)
        return pd.DataFrame( np.dot(gaussian(self.timeline[:,None], self.timeline[C][None,:],bandwith), self.hazards_.values[C,:]), 
                columns=self.hazards_.columns, index=self.timeline)

  def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf(1 - (1-self.alpha)/2)
        n = self.timeline.shape[0]
        d = self.cumulative_hazards_.shape[1]
        index = [['upper']*n+['lower']*n, np.concatenate( [self.timeline, self.timeline] ) ]
        self.confidence_intervals_ = pd.DataFrame(np.zeros((2*n,d)),index=index)
        self.confidence_intervals_.ix['upper'] = self.cumulative_hazards_.values + alpha2*np.sqrt(self._variance.values)
        self.confidence_intervals_.ix['lower'] = self.cumulative_hazards_.values - alpha2*np.sqrt(self._variance.values)
        return 

  def predict_cumulative_hazard(self, X, columns=None):
    """
    X: a (n,d) covariate matrix

    Returns the hazard rates for the individuals
    """
    n,d = X.shape
    try:
      X_ = X.values.copy()
    except:
      X_ = X.copy()
    X_ = X.copy() if not self.fit_intercept else np.c_[ X.copy(), np.ones((n,1)) ]
    return pd.DataFrame(np.dot(self.cumulative_hazards_, X_.T), index=self.timeline, columns=columns )
  
  def predict_survival_function(self,X, columns=None):
    """
    X: a (n,d) covariate matrix

    Returns the survival functions for the individuals
    """
    return np.exp(-self.predict_cumulative_hazard(X, columns=columns))

  def predict_median(self,X):
    """
    X: a (n,d) covariate matrix
    Returns the median lifetimes for the individuals
    """
    return median_survival_times(self.predict_survival_function(X))

  def predict_expectation(self,X):
    """
    Compute the expected lifetime, E[T], using covarites X.
    """
    t = self.cumulative_hazards_.index
    return quadrature(self.predict_survival_function(X).values.T, t)

#utils
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
    assert 0.<=q<=1., "q must be between 0. and 1."
    sv_b = (1.0*(survival_functions < q)).cumsum() > 0
    try:
        v = sv_b.idxmax(0)
        v[sv_b.iloc[-1,:]==0] = -1
    except:
        v = sv_b.argmax(0)
        v[sv_b[-1,:]==0] = np.inf
    return v

def median_survival_times(survival_functions):
    return qth_survival_times(0.5, survival_functions)

def gaussian(t,T,sigma=1.):
    return 1./np.sqrt(np.pi*2.*sigma**2)*np.exp(-0.5*(t-T)**2/sigma**2)

def ipcw(target_event_times, target_censorship, predicted_event_times ):
    pass

"""
References:
[1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

"""
