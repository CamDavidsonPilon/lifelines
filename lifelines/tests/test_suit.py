"""
python -m lifelines.tests.test_suit

"""
import unittest
import numpy as np
import numpy.testing as npt
from collections import Counter

from ..estimation import KaplanMeierFitter, NelsonAalenFitter, AalenAdditiveFitter
from ..statistics import intensity_test

LIFETIMES = np.array([2,4,4,4,5,7,10,11,11,12])
CENSORSHIP = np.array([1,1,0,1,0,1,1,1,1,0])
N = len(LIFETIMES)

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
      censorA = np.random.binomial(200,0.5, size=(200,1))
      censorB = np.random.binomial(200,0.5, size=(200,1))
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


if __name__ == '__main__':
    unittest.main()
