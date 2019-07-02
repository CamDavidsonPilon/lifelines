# -*- coding: utf-8 -*-
from lifelines.fitters import ParametricRegressionFitter
import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt


class Custom(ParametricRegressionFitter):

    _fitted_parameter_names = ["lambda_", "rho_", "delta_"]

    def _cumulative_hazard(self, params, T, Xs):

        lambda_ = anp.exp(anp.dot(Xs["lambda_"], params["lambda_"]))  # > 0
        rho_ = anp.exp(anp.dot(Xs["rho_"], params["rho_"]))  # > 0
        delta_ = anp.exp(anp.dot(Xs["delta_"], params["delta_"]))  # > 0

        return (T / lambda_) + (T / rho_) ** 0.5 + (T / delta_) ** 2

    # def _add_penalty(self, params, neg_ll):


from lifelines.datasets import load_rossi

swf = Custom(penalizer=1.0)
rossi = load_rossi()
rossi["intercept"] = 1.0

covariates = {
    "lambda_": ["intercept", "fin", "age", "race"],  # need a shortcut for all columns?
    "rho_": ["intercept"],
    "delta_": ["age"],
}

swf.fit(rossi, "week", event_col="arrest", regressors=covariates)  # TODO: name
swf.print_summary()
swf.plot()

plt.show()
