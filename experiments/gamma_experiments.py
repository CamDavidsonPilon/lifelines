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
gg.fit(T, show_progress=True)  # initial_point=np.array([4.5, 0.1, 9]))

gg.print_summary(3)


lg = LogNormalFitter().fit(T)
lg.print_summary()


lg = WeibullFitter().fit(T)
lg.print_summary()
