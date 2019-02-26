# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import weibull_min
import pandas as pd
from lifelines import WeibullAFTFitter, CoxPHFitter

# This is an implementation of https://uwspace.uwaterloo.ca/bitstream/handle/10012/10265/Cook_Richard-10265.pdf

N = 50000
p = 0.5
bX = np.log(0.5)
bZ = np.log(4)

Z = np.random.binomial(1, p, size=N)
X = np.random.binomial(1, 0.5, size=N)
X_ = 20000 + 10 * np.random.randn(N)

W = weibull_min.rvs(1, scale=1, loc=0, size=N)

Y = bX * X + bZ * Z + np.log(W)
T = np.exp(Y)

#######################################

df = pd.DataFrame({"T": T, "x": X, "x_": X_})


wf = WeibullAFTFitter().fit(df, "T")
wf.print_summary(4)


cph = CoxPHFitter().fit(df, "T", show_progress=True, step_size=1.0)
cph.print_summary(4)
