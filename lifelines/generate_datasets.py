#lib to create fake survival datasets
import numpy as np
from numpy import random
import pandas as pd

class coeff_func(object):
  """This is a decorator class used later to construct nice names"""
  def __init__(self, f):
    self.f = f

  def __call__(self, *args, **kwargs):
    def __repr__():
      s = self.f.__doc__.replace("alpha", "%.2f"%kwargs["alpha"]).replace("beta","%.2f"%kwargs["beta"])
      return s
    self.__doc__ = __repr__()
    self.__repr__ = __repr__
    self.__str__ = __repr__
    return self.f(*args, **kwargs)


@coeff_func
def exp_comp_(t,alpha=1,beta =1):
  """beta*(1 - np.exp(-alpha*(t-beta)))"""
  return beta*(1 - np.exp(-alpha*np.maximum(0,t-10*beta)))
@coeff_func
def log_(t,alpha=1, beta=1):
  """beta*np.log(alpha*(t-beta)+1)"""
  return beta*np.log(alpha*np.maximum(0,t-10*beta)+1)
@coeff_func
def inverseSq_(t,alpha=1,beta=1):
    """beta/(t+alpha+1)**(0.5)"""
    return beta/(t+alpha+1)**(0.5)
@coeff_func
def periodic_(t, alpha=1,beta=1):
    """abs(0.5*beta*sin(0.1*alpha*t + alpha*beta))"""
    return 0.5*beta*np.sin(0.1*alpha*t)
@coeff_func
def constant_(t,alpha=1,beta=1):
    """beta"""
    return beta

FUNCS = [exp_comp_, log_, inverseSq_, constant_, periodic_ ] 


def right_censor_lifetimes(lifetimes, max_ , min_ = 0):
  """
  Right censor the deaths, uniformly
    lifetimes: (n,) array of positive random variables
    max_: the max time a censorship can occur
    min_: the min time a censorship can occur 

  Returns 
    The actual observations including uniform right censoring, and
    D_i (observed death or did not)

  I think this is deprecated
  """
  n = lifetimes.shape[0]
  u = min_ + (max_-min_)*random.rand(n)
  observations = np.minimum( u, lifetimes)
  return observations, lifetimes == observations


### Models with covariates

def generate_covariates(n, d, n_binary=0, p=0.5):
    """
    n: the number of instances, integer
    d: the dimension of the covarites, integer
    binary: a float between 0 and d the represents the binary covariates 
    p: in binary, the probability of 1

    returns (n, d+1)
    """
    assert ( n_binary>=0 and n_binary<=d ), "binary must be between 0 and d"
    covariates = np.zeros((n,d+1))
    covariates[:, :d-n_binary] = random.exponential(1, size=(n, d-n_binary))
    covariates[:, d-n_binary:-1 ] =  random.binomial(1, p, size=(n, n_binary))
    covariates[:,-1] = np.ones(n)
    return covariates



def constant_coefficients(d, timelines, constant=False, independent=0):
    """
    Proportional hazards model.

    d: the dimension of the dataset
    timelines: the observational times
    constant: True for constant coefficients 
    independent: the number of coffients to set to 0 (covariate is ind of survival), or 
      a list of covariates to make indepent.

    returns a matrix (t,d+1) of coefficients
    """
    return time_varying_coefficients(d, timelines, constant=True, independent=independent, randgen=random.normal )


def time_varying_coefficients(d, timelines, constant=False, independent=0, randgen= random.exponential):
    """
    Time vary coefficients

    d: the dimension of the dataset
    timelines: the observational times
    constant: True for constant coefficients 
    independent: the number of coffients to set to 0 (covariate is ind of survival), or 
      a list of covariates to make indepent.
    randgen: how scalar coefficients (betas) are sampled.

    returns a matrix (t,d+1) of coefficients
    """
    t = timelines.shape[0]
    try:
      a = np.arange(d)
      random.shuffle(a)
      independent = a[:independent]
    except IndexError:
      pass

    n_funcs = len(FUNCS)
    coefficients = np.zeros((t,d))
    data_generators = []
    for i in range(d):
       f = FUNCS[random.randint(0,n_funcs)] if not constant else constant_
       if i in independent:
          beta = 0
       else:
          beta = randgen((1-constant)*0.5/d)
       coefficients[:,i] = f(timelines,alpha=randgen(2000.0/t), beta=beta)
       data_generators.append(f.__doc__)

    df_coefficients = pd.DataFrame( coefficients, columns = data_generators, index = timelines)
    return df_coefficients


def generate_hazard_rates(n,d, timelines, constant=False, independent=0, n_binary=0, model="aalen"):
  """
    n: the number of instances
    d: the number of covariates
    lifelines: the observational times
    constant: make the coeffients constant (not time dependent)
    n_binary: the number of binary covariates
    model: from ["aalen", "cox"]

  Returns:s
    hazard rates: (t,n) dataframe,
    coefficients: (t,d+1) dataframe of coefficients, 
    covarites: (n,d) dataframe

  """
  covariates = generate_covariates(n,d, n_binary=n_binary)
  if model=="aalen":
      coefficients = time_varying_coefficients(d+1,timelines, independent=independent, constant=constant)
      hazard_rates = np.dot( covariates, coefficients.T )
      return pd.DataFrame(hazard_rates.T,index=timelines), coefficients, pd.DataFrame(covariates)
  elif model=="cox":
      covariates = covariates[:,:-1]
      coefficients = constant_coefficients(d, timelines, independent)
      baseline = time_varying_coefficients(1, timelines)
      hazard_rates =np.exp(np.dot( covariates, coefficients.T ))*baseline[baseline.columns[0]].values
      coefficients[ "baseline: "+ baseline.columns[0] ] = baseline.values
      return pd.DataFrame(hazard_rates.T,index=timelines),coefficients, pd.DataFrame(covariates)
  else:
    raise Exception

def generate_random_lifetimes(hazard_rates,timelines, size = 1, censor=None):
  """ 
  Based on the hazard rates, compute random variables from the survival function
    hazard_rates: (n,t) array of hazard rates
    timelines: (t,) the observation times
    size: the number to return, per hardard rate 
    censor: If True, adds uniform censoring between timelines.max() and  0
            If a postive number, censors all events above that value.
            If (n,) np.array >=0 , censor elementwise. 


  Returns:
    survival_times: (size,n) array of random variables.
    (optional) censorship: if censor is true, returns (size,n) array with bool True 
       if the death was observed (not right-censored)
  """
  n = hazard_rates.shape[1]
  survival_times = np.empty((n,size))
  cumulative_hazards = cumulative_quadrature(hazard_rates.values.T, timelines)

  for i in range(size):
     u = random.rand(n,1)
     e = -np.log(u)
     v = (e - cumulative_hazards) < 0
     cross = v.argmax(1)
     survival_times[:,i] = timelines[cross]
     survival_times[cross==0,i] = np.inf

  if censor is not None:
      if type(censor)==bool:
        T = timelines.max()
        rv = T*random.uniform(size=survival_times.shape)
      else:
        rv = censor

      observed = np.less_equal(survival_times, rv)
      survival_times = np.minimum(rv, survival_times)
      return survival_times.T, observed.T
  else:
      return survival_times

def generate_observational_matrix(n,d, timelines, constant=False, independent=0, n_binary=0, model="aalen"):
  hz, coeff, covariates = generate_hazard_rates(n,d, timelines, constant=False, independent=0, n_binary=0, model="aalen")
  R = generate_random_lifetimes(hz,timelines)
  covariates["event_at"] = R.T
  return covariates.sort("event_at"), pd.DataFrame(cumulative_quadrature(coeff.values.T,timelines).T, columns = coeff.columns, index = timelines )

def cumulative_quadrature(fx, x):
  """Boss algo"""

  lower_x = x[0]
  cum_integral = np.empty((fx.shape[0],x.shape[0]))
  cum_integral[:,0] = 0
  for i in range(1,x.shape[0]):
    if (x[i-1]-lower_x) == 0:
        cum_integral[:,i] = 0.5*(fx[:,i-1] + fx[:,i])*(x[i] - lower_x)/i
        continue
    cum_integral[:,i] = ( cum_integral[:,i-1]*( (i-1)/(x[i-1]-lower_x) ) + 0.5*fx[:,i-1] + 0.5*fx[:,i])*( (x[i] - lower_x)/i )
  return cum_integral

def construct_survival_curves(hazard_rates, timelines):
  """
  Given hazard rates, reconstruct the survival curves 
    hazard_rates: (n,t) array
    timelines: (t,) the observational times 

  Returns:t
    survial curves, (n,t) array
  """
  #pdb.set_trace()
  cumulative_hazards = cumulative_quadrature(hazard_rates.values.T, timelines).T
  return pd.DataFrame( np.exp(-cumulative_hazards), index=timelines)



