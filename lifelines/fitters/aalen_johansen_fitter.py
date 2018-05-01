from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import warnings

from lifelines.fitters import UnivariateFitter
from lifelines.utils import _preprocess_inputs
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter

class AalenJohansenFitter(UnivariateFitter):
    """
    Class for fitting the Aalen-Johansen estimate for the cumulative incidence function in a competing 
    risks framework.
    
    AalenJohansenFitter(alpha=0.95,jitter_level=0.00001,seed=None)
    
    Aalen-Johansen cannot deal with tied times. We can get around this by randomy jittering the event times 
    slightly. This will be done automatically and generate a warning.
    """
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
        #First, we check to see if tied event times, if so then need to randomly jitter the data
        if np.sum(durations.duplicated()) > 0: #raise warning if duplicated times, then randomly jitter times
            warnings.warn('''Tied event times were detected. The Aalen-Johansen estimator cannot handle tied event times. 
                To resolve ties, data is randomly jittered.''',Warning)
            durations = _jitter(event_time=durations)
        
        #Setting competing risk label, it is based on the event indicator specified
        cmprisk_label = 'CIF_'+str(int(event_ind))
        overall_events = np.where(event_observed>0,1,0) #create indicator for any event occurring
        
        #Fit overall Kaplan Meier for any event types
        km = KaplanMeierFitter().fit(durations,event_observed=overall_events,timeline=timeline,entry=entry)
        aj = km.event_table
        aj['overall_survival'] = km.survival_function_
        aj['lagged_overall_survival'] = aj['overall_survival'].shift()
        
        #Determine discrete time hazards for event type specified by event_ind
        event_spec = np.where(event_observed==event_ind,1,0)
        event_spec_proc = _preprocess_inputs(durations=durations,event_observed=event_spec,timeline=timeline,entry=entry,weights=weights)
        event_spec_times = event_spec_proc[-1]['observed']
        label_cmprisk = 'observed_'+str(int(event_ind))
        event_spec_times = event_spec_times.rename(label_cmprisk)
        aj = pd.concat([aj,event_spec_times],axis=1)
        aj = aj.reset_index().copy()        
        aj['discrete_hazard'] = -np.log(aj['overall_survival']) + np.log(aj['lagged_overall_survival'])
        aj.loc[0,'h-h'] = 0 #setting initial discrete time hazard to zero
        aj['stepsize'] = aj['lagged_overall_survival'] * aj['discrete_hazard'] * aj[label_cmprisk] #Calculating jump size
        aj[cmprisk_label] = aj['stepsize'].cumsum() #Calculating Cumulative Incidence Function
        aj.loc[0,cmprisk_label] = 0 #Setting initial CIF to be zero
        aj = aj.set_index('event_at')
        
        #Setting attributes
        self.event_table = aj[['removed','observed',label_cmprisk,'censored','entrance','at_risk']] #Event table, like KaplanMeier
        self.cumulative_density_ = aj[cmprisk_label] #Technically, cumulative incidence, but labeled as density to keep consistency with Kaplan Meier
        return self
    
    def _jitter(self,event_time,jitter_level=0.00001,seed=None):
        '''Determine extent to jitter tied event times. Automatically called by fit if tied event times are detected
        '''
        #Setting inital seed
        if seed != None:
            np.random.seed(seed)
            
        #Determining whether to randomly shift event times up or down
        mark = np.random.choice([-1,1],size=event_time.shape[0])
        
        #Determining extent to jitter event times up or down
        shift = np.random.uniform(size=jf.shape[0])*jitter_level)
        return event_time + mark*shift #Return randomly jittered event times

    
    def _bounds(self,alpha):
        '''To be added...  I need to try and find an exponential version, so it matches the Kaplan Meier.
        Currently, I only have a formula (based on Greenwoods) on pg411 of "Modelling Survival Data in Medical 
        Research" David Collett 3rd Edition'''
        return aj
        


