# -*- coding: utf-8 -*-
# piecewise regression model

import numpy as np
import pandas as pd
from lifelines.fitters.piecewise_exponential_regression_fitter import PiecewiseExponentialRegressionFitter
from lifelines import *
from lifelines.datasets import load_regression_dataset
from lifelines.generate_datasets import piecewise_exponential_survival_data


N, d = 2000, 1

breakpoints = (1, 31, 34, 62, 65, 93, 96)  # initial purchase  # second bill  # third bill  # fourth bill

betas = np.array(
    [
        [-1.25, np.log(15)],
        [-2.25, np.log(333)],
        [-1.1, np.log(18)],
        [-2.1, np.log(500)],
        [-1.0, np.log(20)],
        [-1.8, np.log(500)],
        [-0.5, np.log(20)],
        [-1.5, np.log(250)],
    ]
)

X = 0.1 * np.random.randn(N, d)
X = np.c_[X, np.ones(N)]

T = np.empty(N)
for i in range(N):
    lambdas = np.exp(-betas.dot(X[i, :]))
    T[i] = piecewise_exponential_survival_data(1, breakpoints, lambdas)[0]

T_censor = np.minimum(0.9 * T.mean() * np.random.exponential(size=N), 110)

df = pd.DataFrame(X)
df["T"] = np.minimum(T, T_censor)
df["E"] = T <= T_censor


pew = PiecewiseExponentialRegressionFitter(breakpoints=breakpoints, penalizer=0.0, fit_intercept=False).fit(
    df, "T", "E"
)
pew.print_summary()

kmf = KaplanMeierFitter().fit(df["T"], df["E"])
