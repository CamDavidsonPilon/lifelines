# -*- coding: utf-8 -*-
from lifelines.fitters import ParametricRegressionFitter
from autograd.scipy.special import expit
from autograd import numpy as np
from lifelines.utils.safe_exp import safe_exp

exp = safe_exp
dot = np.dot


class CopulaFrailtyWeilbullModel(ParametricRegressionFitter):
    """

    Assume that subjects have two competing risks that are _not_ independent.
    Assume that their is a frailty term that both risks depend on, and, conditional
    on this term, the risks are IID Weibull models. If the frailty term is
    stable with parameter alpha, then the cumulative hazard can be written down explicitly.



    Reference
    --------------
    Frees and Valdez, UNDERSTANDING RELATIONSHIPS USING COPULAS

    """

    _fitted_parameter_names = ["lambda1", "rho1", "lambda2", "rho2", "alpha"]

    def _cumulative_hazard(self, params, T, Xs):
        lambda1 = exp(dot(Xs["lambda1"], params["lambda1"]))
        lambda2 = exp(dot(Xs["lambda2"], params["lambda2"]))
        rho2 = exp(dot(Xs["rho2"], params["rho2"]))
        rho1 = exp(dot(Xs["rho1"], params["rho1"]))
        alpha = exp(dot(Xs["alpha"], params["alpha"]))

        return ((T / lambda1) ** rho1 + (T / lambda2) ** rho2) ** alpha


swf = CopulaFrailtyWeilbullModel(penalizer=0.1)

rossi = load_rossi()
rossi["intercept"] = 1.0
rossi["week"] = rossi["week"] / 54.0

covariates = {
    "lambda1": rossi.columns,
    "lambda2": rossi.columns,
    "rho1": ["intercept"],
    "rho2": ["intercept"],
    "alpha": ["intercept"],
}

swf.fit(rossi, "week", event_col="arrest", regressors=covariates, timeline=np.linspace(0, 2))
swf.print_summary(2)
