"""
python -m lifelines.tests.test_suit

"""
import unittest
import numpy as np
import numpy.testing as npt
from collections import Counter
import matplotlib.pyplot as plt
import pdb

from ..estimation import KaplanMeierFitter, NelsonAalenFitter, AalenAdditiveFitter
from ..statistics import intensity_test
from ..generate_datasets import *
from ..plotting import plot_lifetimes

LIFETIMES = np.array([2,4,4,4,5,7,10,11,11,12])
CENSORSHIP = np.array([1,1,0,1,0,1,1,1,1,0])
N = len(LIFETIMES)

class MiscTests(unittest.TestCase):

    def test_quadratic_integration_identity(self):
      #integrate x between 0 and 10:
      x = np.linspace(0,10,1000)[:,None]
      answer = 0.5*x**2
      approx_answer = cumulative_quadrature( x.T, x).T
      npt.assert_almost_equal(answer, approx_answer, decimal=5)


    def test_quadratic_integration_exp(self):
      #integrate exp(x) between 0 and 4:
      x = np.linspace(0,4,1000)[:,None]
      answer = np.exp(x) -1.
      approx_answer = cumulative_quadrature( np.exp(x).T, x).T
      #pdb.set_trace()
      npt.assert_almost_equal(answer, approx_answer, decimal=4)

class StatisticalTests(unittest.TestCase):

  def setUp(self):
      self.lifetimes = Counter(LIFETIMES)
      self.km = self.kaplan_meier()
      self.kmc = self.kaplan_meier(censor=True)
      self.na = self.nelson_aalen()
      self.nac = self.nelson_aalen(censor=True)

  def test_kaplan_meier(self):
      kmf = KaplanMeierFitter()
      kmf.fit(LIFETIMES)
      npt.assert_almost_equal(kmf.survival_function_.values, self.km )
  
  def test_nelson_aalen(self):
      naf = NelsonAalenFitter()
      naf.fit(LIFETIMES)
      npt.assert_almost_equal(naf.cumulative_hazard_.values, self.na )
  
  def test_censor_nelson_aalen(self):
      naf = NelsonAalenFitter()
      naf.fit(LIFETIMES, censorship=CENSORSHIP)
      npt.assert_almost_equal(naf.cumulative_hazard_.values, self.nac )
  
  def test_censor_kaplan_meier(self):
      kmf = KaplanMeierFitter()
      kmf.fit(LIFETIMES, censorship = CENSORSHIP)
      npt.assert_almost_equal(kmf.survival_function_.values, self.kmc )

  def test_equal_intensity(self):
      data1 = np.random.exponential(5, size=(200,1))
      data2 = np.random.exponential(5, size=(200,1))
      U = intensity_test(data1, data2)
      self.assertTrue( np.abs(U)<=1.96)

  def test_unequal_intensity(self):
      data1 = np.random.exponential(5, size=(200,1))
      data2 = np.random.exponential(1, size=(200,1))
      U = intensity_test(data1, data2)
      self.assertTrue( np.abs(U)>=1.96)

  def test_unequal_intensity_censorship(self):
      data1 = np.random.exponential(5, size=(200,1))
      data2 = np.random.exponential(1, size=(200,1))
      censorA = np.random.binomial(1,0.5, size=(200,1))
      censorB = np.random.binomial(1,0.5, size=(200,1))
      U = intensity_test(data1, data2, censorship_A = censorA, censorship_B = censorB)
      self.assertTrue( np.abs(U)>=1.96)

  def kaplan_meier(self, censor=False):
      km = np.zeros((len(self.lifetimes.keys()),1))
      ordered_lifetimes = np.sort( self.lifetimes.keys())
      v = 1.
      n = N*1.0
      for i,t in enumerate(ordered_lifetimes):
        if censor:
           ix = LIFETIMES == t
           c = sum(1-CENSORSHIP[ix])
           n -=  c
           if n!=0:
              v *= ( 1-(self.lifetimes.get(t)-c)/n )
           n -= self.lifetimes.get(t) - c
        else:
           v *= ( 1-self.lifetimes.get(t)/n )
           n -= self.lifetimes.get(t)
        km[i] = v
      return km

  def nelson_aalen(self, censor = False):
      na = np.zeros((len(self.lifetimes.keys()),1))
      ordered_lifetimes = np.sort( self.lifetimes.keys())
      v = 0.
      n = N*1.0
      for i,t in enumerate(ordered_lifetimes):
        if censor:
           ix = LIFETIMES == t
           c = sum(1-CENSORSHIP[ix])
           n -=  c
           if n!=0:
              v += ( (self.lifetimes.get(t)-c)/n )
           n -= self.lifetimes.get(t) - c
        else:
           v += ( self.lifetimes.get(t)/n )
           n -= self.lifetimes.get(t)
        na[i] = v
      return na

class PlottingTests(unittest.TestCase):

  def test_kmf_plotting(self):
      data1 = np.random.exponential(5, size=(200,1))
      data2 = np.random.exponential(5, size=(200,1))
      kmf = KaplanMeierFitter()
      kmf.fit(data1)
      ax = kmf.plot()
      kmf.fit(data2)
      kmf.plot(ax=ax, c="#A60628")
      return 

  def test_naf_plotting(self):
      data1 = np.random.exponential(5, size=(200,1))
      data2 = np.random.exponential(1, size=(200,1))
      naf = NelsonAalenFitter()
      naf.fit(data1)
      ax = naf.plot()
      naf.fit(data2)
      naf.plot(ax=ax, c="#A60628")
      return 

  def test_plot_lifetimes_calendar(self):
      plt.figure()
      t = np.linspace(0, 20, 1000)
      hz, coef, covrt =generate_hazard_rates(1,5, t) 
      N = 20
      current = 10
      birthtimes = current*np.random.uniform(size=(N,))
      T, C= generate_random_lifetimes(hz, t, size=N, censor=current - birthtimes )
      plot_lifetimes(T, censorship=C, birthtimes=birthtimes)

  def test_plot_lifetimes_relative(self):
      plt.figure()
      t = np.linspace(0, 20, 1000)
      hz, coef, covrt =generate_hazard_rates(1,5, t) 
      N = 20
      T, C= generate_random_lifetimes(hz, t, size=N, censor=True )
      plot_lifetimes(T, censorship=C)

if __name__ == '__main__':
    unittest.main()
