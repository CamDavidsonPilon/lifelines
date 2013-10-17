import numpy as np
from itertools import combinations

from lifelines.utils import dataframe_from_events_censorship, inv_normal_cdf,normal_cdf

import pdb



def multi_logrank_test( event_durations, groups, censorship=None, t_0=-1, alpha=0.95):
  """
  This uses the berferonni? correction

  """
  assert event_durations.shape[0] == groups.shape[0], "event_durations must be the same shape as groups"
  
  if censorship is None:
      censorship = np.ones((event_durations.shape[0],1))

  data = pd.DataFrame( np.c_[groups, event_durations, censorship], columns=['groups','durations','censorship']).groupby('group')

  new_alpha = 1 - (1. - alpha)/(n*(n-1)/2.) #correction accounting for n pairwise comparisons.


  raise Exception
  #iterate over all combinations. 
    


def logrank_test(event_times_A, event_times_B, censorship_A=None, censorship_B=None, alpha=0.95, t_0=-1):
  """
  Measures and reports on whether two intensity processes are different. That is, given two 
  event series, determines whether the data generating processes are statistically different. 

  Pre lifelines 0.2: this returned a test statistic. 
  Post lifelines 0.2: this returns the results of the entire test. 

  See Survival and Event Analysis, page 108. This implicitly uses the log-rank weights.

  Parameters:
    event_times_X: a (nx1) array of event durations (birth to death,...) for the population.
    t_0: the period under observation, -1 for all time.

  Returns 
    summary: a print-friendly string detailing the results of the test.
    p: the p-value
    the test result: True if reject the null, (pendantically None if inconclusive)
  """
  if censorship_A is None:
    censorship_A = np.ones((event_times_A.shape[0], 1))  
  if censorship_B is None:
    censorship_B = np.ones((event_times_B.shape[0], 1))

  if t_0 == -1: 
    t_0 = np.max([event_times_A.max(), event_times_B.max()])
  
  event_times_AB = dataframe_from_events_censorship( np.append(event_times_A,event_times_B),
                                                     np.append( censorship_A, censorship_B) )

  event_times_A = dataframe_from_events_censorship( event_times_A, censorship_A)
  event_times_B = dataframe_from_events_censorship( event_times_B, censorship_B)

  #Poisson Process of total deaths observed 
  N_dot = event_times_AB[["observed"]].cumsum() 

  #susceptible population remaining
  Y_dot = event_times_AB["removed"].sum() - event_times_AB["removed"].cumsum() 

  #susceptible population remaining in subpopulations
  Y_1 = event_times_A["removed"].sum() - event_times_A["removed"].cumsum() 
  Y_2 = event_times_B["removed"].sum() - event_times_B["removed"].cumsum()

  v = 0
  v_sq = 0
  y_1 = Y_1.iloc[0]
  y_2 = Y_2.iloc[0]
  n_t_ = 0
  for t, n_t in N_dot.ix[N_dot.index<=t_0].itertuples():
    try:
      #sorta a nasty hack to check of the time is not in the 
      # data. Could be done better.
      y_1 = Y_1.loc[t]
    except KeyError:
      pass  
    except IndexError:
      pass
    try:  
      y_2 = Y_2.loc[t]
    except KeyError:
      pass
    except IndexError:
      pass
    y_dot = Y_dot.loc[t]
    if y_dot != 0:
      delta = n_t - n_t_
      v += 1.*y_1*delta/y_dot
      if y_dot >= 2:
         v_sq += (1.*y_2*y_1*delta)*(y_dot - delta)/(y_dot**2)/(y_dot-1)
      n_t_ = n_t

  E_1 = v
  N_1 = event_times_A[["observed"]].sum()[0]
  Z_1 = N_1 - E_1
  U = Z_1/np.sqrt(v_sq) #this is approx normal under null.
  
  test_result, p_value = z_test(U,alpha)
  summary = pretty_print_summary(test_result, p_value, U, t_0=t_0, test='logrank', alpha=alpha)
  return summary, p_value, test_result


def p_value_normal(U):
  return 2*normal_cdf(-np.abs(U))

def z_test(Z, alpha):
  """
  Pendantically returns None, p-value if test is inconclusive, else returns True, p-value.
  """
  p = p_value_normal(Z)
  if (Z < inv_normal_cdf((1.-alpha)/2.) ) or (Z > inv_normal_cdf((1+alpha)/2)):
    #reject null
    return True, p #TODO
  else:
    #cannot reject null
    return None, p

def pretty_print_summary( test_results, p_value, test_statistic, **kwargs):
  """
  kwargs are experiment meta-data. 
  """
  HEADER = "   __ p-value ___|__ test statistic __|__ test results __"


  s = "Results\n"
  meta_data = pretty_print_meta_data( kwargs )
  s += meta_data + "\n"
  s += HEADER + "\n"
  s += "         %.5f |            %.3f |   %s   "%(p_value, test_statistic, test_results)
  return s

def pretty_print_meta_data(dictionary):
  s = ""
  for k,v in dictionary.iteritems():
      s=  s + "   " + k.__str__() +  ": " + v.__str__() + "\n"
  return s
