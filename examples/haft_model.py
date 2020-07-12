# -*- coding: utf-8 -*-
"""
Reimplementation of "A Heteroscedastic Accelerated Failure Time Model
for Survival Analysis", Wang, You, Lysy, 2019
"""
import pandas as pd
from patsy import dmatrix
import autograd.numpy as np
from autograd.scipy.stats import norm
from lifelines.fitters import ParametricRegressionFitter
from lifelines import CoxPHFitter, LogNormalAFTFitter


# this is all that's need to implement the HAFT model.
# lifelines handles all estimation.
class HAFT(ParametricRegressionFitter):

    _fitted_parameter_names = ["mu_", "sigma_"]

    def _cumulative_hazard(self, params, T, Xs):

        sigma_2 = np.exp(np.dot(Xs["sigma_"], params["sigma_"]))
        sigma_ = np.sqrt(sigma_2)
        mu_ = np.dot(Xs["mu_"], params["mu_"])

        Z = (np.log(T) - mu_) / sigma_
        return -norm.logsf(Z)


df = pd.read_csv("colon.csv", index_col=0).dropna()
df = df[df["etype"] == 2]

# Cox model
cph_string = """{extent} +
    {rx} +
    {differ} +
    sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
    obstruct:perfor +
    age:{differ} +
    age:sex +
    {rx}:sex +
    {rx}:obstruct +
    adhere:nodes
""".format(
    rx="C(rx, Treatment('Obs'))", differ="C(differ, Treatment(1))", extent="C(extent, Treatment(1))"
)

cph = CoxPHFitter().fit(df, "time", "status", formula=cph_string)
cph.print_summary()

# Log Normal model
aft_string = """{extent} +
    {rx} +
    {differ} +
    sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
    obstruct:perfor +
    age:{differ} +
    age:sex +
    {rx}:sex +
    age:adhere +
    adhere:{differ}
""".format(
    rx="C(rx, Treatment('Obs'))", differ="C(differ, Treatment(1))", extent="C(extent, Treatment(1))"
)
lnf = LogNormalAFTFitter().fit(df, "time", "status", formula=aft_string)
lnf.print_summary()

# H-AFT log normal model
haft = HAFT()
covariates = {"mu_": aft_string, "sigma_": ["C(rx, Treatment('Obs'))"]}

haft.fit(df, "time", event_col="status", regressors=covariates)
haft.print_summary(4)
