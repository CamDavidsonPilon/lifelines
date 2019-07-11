# -*- coding: utf-8 -*-
from autograd import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter, ParametericUnivariateFitter
from lifelines.utils.gamma import gammainc, gammaincc
from lifelines import GeneralizedGammaFitter


from lifelines.datasets import load_waltons

df = load_waltons()

T = np.arange(1, 100)

gg = GeneralizedGammaFitter()
# gg.fit(T, show_progress=True)
gg.fit(T, initial_point=np.array([0.0, 0.6, 1.0]), show_progress=True)

gg.print_summary(3)
print(gg.variance_matrix_)
