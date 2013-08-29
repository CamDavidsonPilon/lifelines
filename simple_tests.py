#quick tests

import numpy as np
from generate_datasets import *

T = np.linspace(0,100,1000)

def int_sv_equals_exp_T():
  """
  Integral of SV curve = E[T]
  """
  #create a survival curve
  hz, coefs, covart = generate_hazard_rates(1, 10, T)
  sv = construct_survival_curves(hz, T)
  ET = cumulative_quadrature(sv.values.T, T )[-1]
  rv = generate_random_lifetimes(hz, T, size=1000)
  print "ET: %.3f, rv.mean(): %.3f"%(ET.mean(), rv.mean())

