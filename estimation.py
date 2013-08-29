#estimation tests
import pandas as pd
import numpy as np
import pdb




def _additive_estimate(event_times, timeline, additive_f,initial):

    n = timeline.shape[0]
    _additive_estimate_ = pd.DataFrame(np.zeros((n,1)), index=timeline)

    N = event_times.shape[0]
    t_0 = event_times[0]
    _additive_estimate_.ix[(timeline<t_0)]=initial

    v = initial

    for i,t in enumerate(event_times):
        v += additive_f(N,i)
        _additive_estimate_.ix[ (t_0<timeline)*(timeline<=t) ] = v  
        t_0 = t

    _additive_estimate_.ix[(timeline>t)]=v
    return _additive_estimate_


class NelsonAalenFitter(object):
  
    def __init__(self):
        pass

    def additive_f(self, N,i ):
       #check it 0
       return 1./(N-i)

    def fit(self, event_times, timeline=None):
        """
        event_times: an (n,1) array of times that the death event occured at 
        timeline: return the best estimate at the values in timelines (postively increasing)

        Returns:
          DataFrame with index either event_times or timelines (if not None), with
          values as the NelsonAalen estimate
        """
        #need to sort event_times
        self.event_times = np.sort(event_times,0)

        if timeline is None:
          self.timeline = self.event_times[:,0].copy()
        else:
          self.timeline = timeline

        self.cumulative_hazard_ = _additive_estimate(self.event_times, self.timeline, self.additive_f, 0 )
        return


class KaplanMeierFitter(object):
   
  def __init__(self):
      pass

  def fit(self, event_times, timeline=None):
       """
       event_times: an (n,1) array of times that the death event occured at 
       timeline: return the best estimate at the values in timelines (postively increasing)

       Returns:
         DataFrame with index either event_times or timelines (if not None), with
         values as the NelsonAalen estimate
       """
       #need to sort event_times
       self.event_times = np.sort(event_times,0)

       if timeline is None:
          self.timeline = self.event_times[:,0].copy()
       else:
          self.timeline = timeline


       self.survival_function_ = np.exp(_additive_estimate(self.event_times, self.timeline, self.additive_f, 0 ) )
       return self

  def additive_f(self,N,i):
      return np.log(1 - 1./(N-i))