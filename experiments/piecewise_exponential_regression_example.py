# -*- coding: utf-8 -*-
# piecewise regression model

import numpy as np
import pandas as pd
from lifelines.fitters.piecewise_exponential_regression_fitter import PiecewiseExponentialRegressionFitter
from lifelines import *
from lifelines.datasets import load_regression_dataset
from lifelines.generate_datasets import piecewise_exponential_survival_data


N, d = 2500, 2

breakpoints = (1, 31, 34, 62, 65, 93, 96)

betas = np.array(
    [
        [-1.25, 5.0, np.log(15)],
        [-2.25, 6.0, np.log(333)],
        [-1.1, 7.0, np.log(18)],
        [-2.1, 8.0, np.log(500)],
        [-1.0, 9.0, np.log(20)],
        [-1.8, 10.0, np.log(500)],
        [-0.5, 11.0, np.log(20)],
        [-1.5, 12.0, np.log(250)],
    ]
)

X = 0.1 * np.random.exponential(size=(N, d))
X = np.c_[X, np.ones(N)]

T = np.empty(N)
for i in range(N):
    lambdas = np.exp(-betas.dot(X[i, :]))
    T[i] = piecewise_exponential_survival_data(1, breakpoints, lambdas)[0]

T_censor = np.minimum(T.mean() * np.random.exponential(size=N), 110)

df = pd.DataFrame(X)
df["T"] = np.minimum(T, T_censor)
df["E"] = T <= T_censor


pew = PiecewiseExponentialRegressionFitter(breakpoints=breakpoints, penalizer=0.0, fit_intercept=False).fit(
    df, "T", "E"
)
pew.print_summary()

"""
ps = 10.0**np.arange(-5, 5)
results0 = []
results1 = []
results2 = []
results3 = []
results4 = []
results5 = []
results6 = []


for p in ps:
    pew = PiecewiseExponentialRegressionFitter(breakpoints=breakpoints, penalizer=p, fit_intercept=False).fit(df, "T", "E")
    results0.append(pew.summary.loc[('lambda_0_', 0), ['coef', 'lower 0.95', 'upper 0.95']])
    results1.append(pew.summary.loc[('lambda_1_', 0), ['coef', 'lower 0.95', 'upper 0.95']])
    results2.append(pew.summary.loc[('lambda_2_', 0), ['coef', 'lower 0.95', 'upper 0.95']])
    results3.append(pew.summary.loc[('lambda_3_', 0), ['coef', 'lower 0.95', 'upper 0.95']])
    results4.append(pew.summary.loc[('lambda_4_', 0), ['coef', 'lower 0.95', 'upper 0.95']])
    results5.append(pew.summary.loc[('lambda_5_', 0), ['coef', 'lower 0.95', 'upper 0.95']])
    results6.append(pew.summary.loc[('lambda_6_', 0), ['coef', 'lower 0.95', 'upper 0.95']])



ax = pd.DataFrame(results0).assign(x=ps).plot('x', 'coef', logx=True)
pd.DataFrame(results1).assign(x=ps).plot('x', 'coef', logx=True, ax=ax)
pd.DataFrame(results2).assign(x=ps).plot('x', 'coef', logx=True, ax=ax)
pd.DataFrame(results3).assign(x=ps).plot('x', 'coef', logx=True, ax=ax)
pd.DataFrame(results4).assign(x=ps).plot('x', 'coef', logx=True, ax=ax)
pd.DataFrame(results5).assign(x=ps).plot('x', 'coef', logx=True, ax=ax)
pd.DataFrame(results6).assign(x=ps).plot('x', 'coef', logx=True, ax=ax)
"""
