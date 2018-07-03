from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import warnings

from lifelines.fitters import UnivariateFitter
from lifelines.utils import _preprocess_inputs, inv_normal_cdf
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter

class AalenJohansenFitter(UnivariateFitter):
    """
    Class for fitting the Aalen-Johansen estimate for the cumulative incidence function in a competing 
    risks framework.
    
    AalenJohansenFitter(alpha=0.95,jitter_level=0.00001,seed=None)
    
    Aalen-Johansen cannot deal with tied times. We can get around this by randomy jittering the event times 
    slightly. This will be done automatically and generate a warning.
    """
    def fit(self,durations, event_observed, event_indicator, timeline=None, entry=None, label='AJ_estimate', 
            alpha=None, ci_labels=None, weights=None):
        """
        Parameters:
          duration: an array or pd.Series of length n -- duration of subject was observed for 
          event_observed: an array, or pd.Series, of length n. Integer indicator of distinct events. Must be 
             only positive integers, where 0 indicates censoring.
          event_indicator: integer -- indicator for event of interest. All other integers are considered competing events
             Ex) event_observed contains 0,1,2 with 0:censored, 1:lung cancer, 2:death. If event_indicator=1, then deaht (2)
             is considered a competing event. The returned cumulative incidence function corresponds to risk of lung cancer
          timeline: return the best estimate at the values in timelines (postively increasing)
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated (not left-censored) observations. If None, all members of the population
             were born at time 0.
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
          weights: n array, or pd.Series, of length n, if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns:
          self, with new properties like 'cumulative_incidence_'.
        
        """
        #First, we check to see if tied event times, if so then need to randomly jitter the data
        event_times = durations.loc[event_observed!=0].copy() #Only care about tied event times
        if np.sum(durations.duplicated()) > 0: #raise warning if duplicated times, then randomly jitter times
            warnings.warn('''Tied event times were detected. The Aalen-Johansen estimator cannot handle tied event times. 
                To resolve ties, data is randomly jittered.''',Warning)  
            durations = self._jitter(durations=durations,event=event_observed)
        
        #Setting competing risk label, it is based on the event indicator specified
        cmprisk_label = 'CIF_'+str(int(event_indicator))
        overall_events = np.where(event_observed>0,1,0) #create indicator for any event occurring
        
        #Fit overall Kaplan Meier for any event types
        km = KaplanMeierFitter().fit(durations,event_observed=overall_events,timeline=timeline,entry=entry,weights=weights)
        aj = km.event_table
        aj['overall_survival'] = km.survival_function_
        aj['lagged_overall_survival'] = aj['overall_survival'].shift()
        
        #Determine discrete time hazards for event type specified by event_ind
        event_spec = np.where(event_observed==event_indicator,1,0)
        event_spec_proc = _preprocess_inputs(durations=durations,event_observed=event_spec,timeline=timeline,entry=entry,weights=weights)
        event_spec_times = event_spec_proc[-1]['observed']
        self.label_cmprisk = 'observed_'+str(int(event_indicator))
        event_spec_times = event_spec_times.rename(self.label_cmprisk)
        aj = pd.concat([aj,event_spec_times],axis=1)
        aj = aj.reset_index().copy()        
        aj['discrete_hazard'] = (aj[self.label_cmprisk]) / (aj['at_risk'])
        aj['stepsize'] = aj['lagged_overall_survival'] * aj['discrete_hazard'] * aj[self.label_cmprisk] #Calculating jump size
        aj[cmprisk_label] = aj['stepsize'].cumsum() #Calculating Cumulative Incidence Function
        aj.loc[0,cmprisk_label] = 0 #Setting initial CIF to be zero
        aj = aj.set_index('event_at')
        
        #Setting attributes
        alpha = alpha if alpha else self.alpha
        self._label = label
        self.event_table = aj[['removed','observed',self.label_cmprisk,'censored','entrance','at_risk']] #Event table, like KaplanMeier
        self.cumulative_density_ = aj[cmprisk_label] #Technically, cumulative incidence, but labeled as density to keep consistency with Kaplan Meier
        self.variance, self.confidence_interval_ = self._bounds(aj['lagged_overall_survival'],alpha=alpha,ci_labels=ci_labels)
        return self
    
    def _jitter(self,durations,event,jitter_level=0.00001,seed=None):
        '''Determine extent to jitter tied event times. Automatically called by fit if tied event times are detected
        '''
        #Setting inital seed
        if jitter_level <= 0:
            raise ValueError('The jitter level is less than zero, please select a jitter value greater than 0')

        if seed != None:
            np.random.seed(seed)
        
        event_time = durations.loc[event!=0].copy()
        #Determining whether to randomly shift event times up or down
        mark = np.random.choice([-1,1],size=event_time.shape[0])
        #Determining extent to jitter event times up or down
        shift = np.random.uniform(size=event_time.shape[0])*jitter_level
        #Jittering times
        event_time += mark*shift
        durations_jitter = event_time.align(durations)[0].fillna(durations)
        
        #Recursive call if event times are still tied after jitter
        if np.sum(event_time.duplicated()) > 0: 
            return self._jitter(self,durations=durations_jitter,event=event,jitter_level=jitter_level,seed=seed)
        else:
            return durations_jitter
    
    def _bounds(self,lagged_survival,alpha,ci_labels):
        '''To be added...  I need to try and find an exponential version, so it matches the Kaplan Meier.
        Currently, I only have a formula (based on Greenwoods) on pg411 of "Modelling Survival Data in Medical 
        Research" David Collett 3rd Edition
        
        Var(F_j) = sum((F_j(t) - F_j(t_i))**2 * d/(n*(n-d) + S(t_i-1)**2 * ((d*(n-d))/n**3) + 
                    -2 * sum((F_j(t) - F_j(t_i)) * S(t_i-1) * (d/n**2)
        
        Confidence intervals are obtained using the delta method transformation of SE(log(-log(F_j))). This ensures
        that the confidence intervals all lie between 0 and 1. 
        
        SE(log(-log(F_j) = SE(F_j) / (F_j * absolute(log(F_j)))
        
        '''
        df = self.event_table.copy()
        df['Ft'] = self.cumulative_density_
        df['lagS'] = lagged_survival.fillna(1)
        all_vars = []
        for i,r in df.iterrows():
            sf = df.loc[df.index<=r.name].copy()
            sf['part1'] = ((r['Ft'] - sf['Ft'])**2) * (sf['observed'] / (sf['at_risk']*(sf['at_risk'] - sf['observed'])))
            sf['part2'] = ((sf['lagS'])**2) * ((sf[self.label_cmprisk] * ((sf['at_risk']-sf[self.label_cmprisk])/(sf['at_risk']**3))))
            sf['part3'] = (r['Ft'] - sf['Ft']) * sf['lagS'] * (sf[self.label_cmprisk] / (sf['at_risk']**2))
            variance = (sf['part1'].sum()) + (sf['part2'].sum()) - 2*(sf['part3'].sum())
            all_vars.append(variance)
        df['variance'] = all_vars
        
        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."
        
        #Calculating Confidence Intervals
        df['F_transformed'] = np.log(-np.log(df['Ft']))
        df['se'] = np.sqrt(df['variance'])
        df['se_transformed'] = df['se'] / (df['Ft'] * np.absolute(np.log(df['Ft'])))
        zalpha = inv_normal_cdf((1. + alpha) / 2.)
        df[ci_labels[0]] = np.exp(-np.exp(df['F_transformed']+zalpha*df['se_transformed']))
        df[ci_labels[1]] = np.exp(-np.exp(df['F_transformed']-zalpha*df['se_transformed']))
        return df['variance'],df[ci_labels]
                


