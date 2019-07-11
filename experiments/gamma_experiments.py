# -*- coding: utf-8 -*-
from autograd import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter, ParametericUnivariateFitter
from lifelines.utils.gamma import gammainc, gammaincc
from lifelines import GeneralizedGammaFitter


from lifelines.datasets import load_waltons

df = load_waltons()

T = np.arange(1, 100)
# using another minimization algo, my best values are:
"""


<lifelines.GeneralizedGammaFitter: fitted with 99 observations, 0 censored>
number of subjects = 99
  number of events = 99
    log-likelihood = -471.299
        hypothesis = mu_ != 10.6449, sigma_ != 5.09977, lambda_ != 21.497

---
         coef  se(coef)  lower 0.95  upper 0.95       p  -log2(p)
mu_     4.021     0.063       3.897       4.145 <0.0005       inf
sigma_  0.593     0.052       0.492       0.695 <0.0005       inf
lambda_ 1.018     0.037       0.945       1.091 <0.0005       inf
[[ 0.00400668 -0.00102151  0.0004146 ]
 [-0.00102151  0.00266154 -0.00035838]
 [ 0.0004146  -0.00035838  0.00138457]]





"""
gg = GeneralizedGammaFitter()
gg.fit(T, show_progress=True)
# gg.fit(T, initial_point=np.array([0., 0.6, 1.]), show_progress=True)

gg.print_summary(3)
print(gg.variance_matrix_)
