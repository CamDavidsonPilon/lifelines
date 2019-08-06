# -*- coding: utf-8 -*-
"""
Reimplementation of "A Heteroscedastic Accelerated Failure Time Model
for Survival Analysis", Wang et al., 2019
"""
from lifelines.fitters import ParametricRegressionFitter
import pandas as pd
from lifelines import CoxPHFitter, LogNormalFitter
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.scipy.stats import norm


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


cph_string = """{extent} +
    {rx} +
    {differ} +
    sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
    obstruct:perfor +
    age:{differ} +
    age:sex +
    {rx}:sex +
    {rx}:obstruct +
    adhere:nodes +
    time + status
""".format(
    rx="C(rx, Treatment('Obs'))", differ="C(differ, Treatment(1))", extent="C(extent, Treatment(1))"
)
cph_df_ = dmatrix(cph_string, df, return_type="dataframe")

cph = CoxPHFitter().fit(cph_df_.drop("Intercept", axis=1), "time", "status")
cph.print_summary()


aft_string = """{extent} +
    {rx} +
    {differ} +
    sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
    obstruct:perfor +
    age:{differ} +
    age:sex +
    {rx}:sex +
    age:adhere +
    adhere:{differ} +
    time + status
""".format(
    rx="C(rx, Treatment('Obs'))", differ="C(differ, Treatment(1))", extent="C(extent, Treatment(1))"
)
aft_df_ = dmatrix(aft_string, df, return_type="dataframe")
lnf = LogNormalAFTFitter(fit_intercept=True).fit(aft_df_.drop("Intercept", axis=1), "time", "status")
lnf.print_summary()

haft = HAFT(penalizer=0.0)
covariates = {
    "mu_": aft_df_.columns,
    "sigma_": ["Intercept", "C(rx, Treatment('Obs'))[T.Lev]", "C(rx, Treatment('Obs'))[T.Lev+5FU]"],
}

haft.fit(aft_df_, "time", event_col="status", regressors=covariates)
haft.print_summary(4)
