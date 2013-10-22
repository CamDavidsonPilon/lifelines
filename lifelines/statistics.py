import numpy as np
from scipy import stats
from lifelines.utils import group_event_series


def multi_logrank_test( event_durations, groups, censorship=None, alpha=0.95, t_0=-1, bonferroni=True):
  """
  Parameters:
    event_durations: a (n,) numpy array the (partial) lifetimes of all individuals
    groups: a (n,) numpy array of unique group labels for each individual.
    censorship: a (n,) numpy array of censorship events: 1 if observed death, 0 if censored. Defaults
        to all observed.
    alpha: the level of signifiance desired.
    t_0: the final time to compare the series' up to. Defaults to all.
    bonferroni: If true, uses the Bonferroni correction to compare the M=n(n-1)/2 pairs, i.e alpha = alpha/M
          [http://en.wikipedia.org/wiki/Bonferroni_correction]

  Returns:
    summary: a print-friendly summary of the statistical test
    p_value: the p-value
    test_result: True if reject the null, (pendantically) None if we can't reject the null.

  """
  assert event_durations.shape[0] == groups.shape[0], "event_durations must be the same shape as groups"
  
  if censorship is None:
      censorship = np.ones((event_durations.shape[0],1))

  unique_groups, rm, obs, _ = group_event_series(groups, event_durations, censorship,t_0)
  n_groups = unique_groups.shape[0]
  m = n_groups*(n_groups-1)/2

  if bonferroni:
    alpha = 1- (1-alpha)/m

  #compute the factors needed
  N_j = obs.sum(0).values
  n_ij = ( rm.sum(0).values - rm.cumsum(0).shift(1).fillna(0) )
  d_i = obs.sum(1)
  n_i = rm.values.sum() - rm.sum(1).cumsum().shift(1).fillna(0)
  ev = n_ij.mul(d_i/n_i, axis='index').sum(0)
  Z_j =  N_j - ev

  assert abs(Z_j.sum()) < 10e-8, "Sum is not zero." #this should move to a test eventually.

  #compute covariance matrix
  V_ = n_ij.mul(np.sqrt(d_i)/n_i, axis='index').fillna(1)
  V = -np.dot(V_.T, V_)
  ix = np.arange(n_groups)
  V[ix, ix] = V[ix, ix] + ev

  #take the first n-1 groups 
  U = Z_j.ix[:-1].dot(np.linalg.inv(V[:-1,:-1]).dot(Z_j.ix[:-1])) #Z.T*inv(V)*Z

  #compute the p-values and tests
  test_result, p_value = chisq_test(U, n_groups-1, alpha)
  summary = pretty_print_summary(test_result, p_value, U, t_0=t_0, test='logrank', 
                                 alpha=alpha, null_distribution='chi squared', 
                                 df=n_groups-1, use_bonferonni=bonferroni)

  return summary, p_value, test_result


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
    censorship_A = np.ones(event_times_A.shape[0])  
  if censorship_B is None:
    censorship_B = np.ones(event_times_B.shape[0])

  event_times = np.r_[event_times_A, event_times_B]
  groups = np.r_[np.zeros(event_times_A.shape[0]), np.ones(event_times_B.shape[0])]
  censorship = np.r_[censorship_A,censorship_B]
  return multi_logrank_test( event_times, groups, censorship,alpha=alpha, t_0=t_0)



def chisq_test(U, degrees_freedom, alpha):
  p_value = 1 - stats.chi2.cdf(U, degrees_freedom)
  if p_value < 1-alpha:
    return True, p_value
  else:
    return None, p_value

def pretty_print_summary( test_results, p_value, test_statistic, **kwargs):
  """
  kwargs are experiment meta-data. 
  """
  HEADER = "   __ p-value ___|__ test statistic __|__ test results __"


  s = "Results\n"
  meta_data = pretty_print_meta_data( kwargs )
  s += meta_data + "\n"
  s += HEADER + "\n"
  s += "         %.5f |              %.3f |     %s   "%(p_value, test_statistic, test_results)
  return s

def pretty_print_meta_data(dictionary):
  s = ""
  for k,v in dictionary.iteritems():
      s=  s + "   " + k.__str__().replace('_', ' ') +  ": " + v.__str__() + "\n"
  return s
