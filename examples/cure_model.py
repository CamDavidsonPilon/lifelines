# -*- coding: utf-8 -*-
from lifelines.fitters import ParametricRegressionFitter
import autograd.numpy as np
from autograd.scipy.special import expit
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


class CureModel(ParametricRegressionFitter):

    _fitted_parameter_names = ["lambda_", "beta_", "rho_"]

    def _cumulative_hazard(self, params, T, Xs):
        c = expit(np.dot(Xs["beta_"], params["beta_"]))
        lambda_ = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))
        rho_ = np.exp(np.dot(Xs["rho_"], params["rho_"]))

        survival = np.exp(-(T / lambda_) ** rho_)
        return -np.log((1 - c) * 1.0 + c * survival)


from lifelines.datasets import load_rossi

swf = CureModel(penalizer=0.0)

rossi = load_rossi()
rossi["intercept"] = 1.0

covariates = {"lambda_": rossi.columns, "rho_": ["intercept"], "beta_": rossi.columns}

swf.fit(rossi, "week", event_col="arrest", regressors=covariates)  # TODO: name
swf.print_summary(4)
# swf.plot()
# plt.show()
