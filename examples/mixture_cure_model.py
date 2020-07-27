# -*- coding: utf-8 -*-
from lifelines.fitters import ParametricRegressionFitter
from autograd.scipy.special import expit
from autograd import numpy as np
from lifelines.utils.safe_exp import safe_exp
from lifelines.datasets import load_rossi

exp = safe_exp
dot = np.dot


class MixtureCureModel(ParametricRegressionFitter):
    """
    Models two "cure" possibilities: default, repay

    """

    _fitted_parameter_names = ["beta_repay", "c_repay", "c_default", "beta_default"]

    def _cumulative_hazard(self, params, T, Xs):
        p_default_ = exp(dot(Xs["c_default"], params["c_default"]))
        p_repay_ = exp(dot(Xs["c_repay"], params["c_repay"]))
        p_cure_ = 1.0

        p_default = p_default_ / (p_cure_ + p_default_ + p_repay_)
        p_repay = p_repay_ / (p_cure_ + p_default_ + p_repay_)
        p_cure = p_cure_ / (p_cure_ + p_default_ + p_repay_)

        # cox like hazards.
        sf_default = exp(-exp(dot(Xs["beta_default"], params["beta_default"])) * T ** 2)
        sf_repay = exp(-exp(dot(Xs["beta_repay"], params["beta_repay"])) * T)
        sf_cure = 1.0

        return -np.log((p_repay * sf_repay) + (p_default * sf_default) + (p_cure * sf_cure))


swf = MixtureCureModel(penalizer=0.001)

rossi = load_rossi()
rossi["week"] = rossi["week"] / 54.0


covariates = rossi.columns.difference(["week", "arrest"])
regressors = {"beta_default": covariates, "beta_repay": covariates, "c_default": covariates, "c_repay": covariates}

swf.fit(rossi, "week", event_col="arrest", regressors=regressors, timeline=np.linspace(0, 2))
swf.print_summary(2)
