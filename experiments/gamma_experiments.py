# -*- coding: utf-8 -*-
from autograd import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter, ParametericUnivariateFitter
from lifelines.utils.gamma import gammainc, gammaincc
from lifelines import *


from lifelines.datasets import load_waltons

df = load_waltons()

T = np.arange(1, 100)

gg = GeneralizedGammaFitter()
# gg.fit(T, show_progress=True)
gg.fit(T)
print(gg.percentile(0.5))
print(gg.percentile(0.4))
print(gg.percentile(0.3))
print(gg.percentile(0.2))
print(gg.survival_function_)

gg.print_summary(3)


lg = LogNormalFitter().fit(T)
lg.print_summary()


lg = WeibullFitter().fit(T)
lg.print_summary()
