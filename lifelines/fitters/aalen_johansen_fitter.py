from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd

from lifelines.fitters import KaplanMeierFitter
from lifelines.utils import _preprocess_inputs


class AalenJohansenFitter():
    """
    Class for fitting the Aalen-Johansen estimate for the cumulative incidence function in a competing 
    risks framework.
    
    AalenJohansenFitter()
    
    """
    def jitter(event_time,jitter_level=0.00001,seed=None):
        '''Aalen-Johansen cannot deal with tied times. We can get around this
        by randomy jittering the event times slightly. 
        
        Parameters:
            event_time: array or pd.Series with time of event/censoring
            jitter_level: how much jittering to apply to data 
        
        Returns:
            pd.Series of jittered times
        '''
        if seed != None:
            np.random.seed(seed)
        jf = pd.DataFrame()
        jf['t'] = event_time
        jf['mark'] = -1 * np.random.binomial(1,0.5,size=len(jf)) #half increase, half decrease
        jf.loc[jf['mark']==0,'mark'] = 1 
        jf['randon_uniform'] = np.random.uniform(size=len(df)) #random amount to jitter 
        jf['t2'] = jf['t'] + (jf['mark']*jf['randon_uniform']*jitter_level) #jittering times
        return jf['t2'] #return Pandas Series of jittered times

    
    def fit(self,durations,event_observed,event_ind,timeline=None,entry=None,weights=None):
        """
        Parameters:
          duration: an array or pd.Series of length n -- duration of subject was observed for 
          event_observed: an array, or pd.Series, of length n. Integer indicator of distinct events. Must be 
             only positive integers, where 0 indicates censoring.
          event_ind: integer -- indicator for event of interest. All other integers indicatin events in event observed
             will be considered as competing events
          timeline: return the best estimate at the values in timelines (postively increasing)
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated (not left-censored) observations. If None, all members of the population
             were born at time 0.
          weights: weights for event
        """
        if np.sum(durations.duplicated()) > 0: #raise error if duplicated times
            raise ValueError('Aalen-Johansen cannot be estimated with tied events')
        cmprisk_label = 'CDF_'+str(int(event_ind))
        overall_events = np.where(event_observed>0,1,0) #create indicator for any event occurring
        km = KaplanMeierFitter().fit(durations,event_observed=overall_events,timeline=timeline,entry=entry) #Fit KM for survival times
        aj = km.event_table #extract out KM info
        aj['Overall_S'] = km.survival_function_
        event_spec = np.where(event_observed==event_ind,1,0) #extract observed event counts for each unique event
        event_spec_proc = _preprocess_inputs(durations=durations,event_observed=event_spec,timeline=timeline,entry=entry,weights=weights)
        event_spec_times = event_spec_proc[-1]['observed']
        label_cmprisk = 'observed_'+str(int(event_ind))
        event_spec_times.rename(label_cmprisk,inplace=True)
        aj = pd.concat([aj,event_spec_times],axis=1)
        aj['Overall_S_lag'] = aj['Overall_S'].shift() #creating lagged survival times
        aj.reset_index(inplace=True)
        aj['h-h'] = -np.log(aj['Overall_S']) + np.log(aj['Overall_S_lag']) #calculating discrete time hazard
        aj.loc[0,'h-h'] = 0 #setting initial discrete time hazard to zero
        aj['stepsize'] = aj['Overall_S_lag'] * aj['h-h'] * aj[label_cmprisk]
        aj[cmprisk_label] = aj['stepsize'].cumsum()
        aj.loc[0,cmprisk_label] = 0
        aj.set_index('event_at',inplace=True) 
        self.event_table = aj[['removed','observed',cmprisk_label,'censored','entrance','at_risk']] #setting self items
        self.cumulative_density_ = aj[cmprisk_label]
        return self 


