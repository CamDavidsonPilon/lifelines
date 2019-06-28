# -*- coding: utf-8 -*-
from lifelines.fitters import ParametericRegressionFitter
import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt


class Custom(ParametericRegressionFitter):

    # TODO: check if _fitted_parameter_names is equal to keys in regressors!!
    _fitted_parameter_names = ["lambda_", "rho_", "delta_"]

    def _cumulative_hazard(self, params, T, Xs):
        """
        One problem with this is that the dict Xs is going to be very memory consuming. Maybe I can create a new
        "dict" that has shared df and dynamically creates the df on indexing with [], that's cool

        """
        lambda_ = anp.exp(anp.dot(Xs["lambda_"], params["lambda_"]))  # > 0
        rho_ = anp.exp(anp.dot(Xs["rho_"], params["rho_"]))  # > 0
        delta_ = anp.exp(anp.dot(Xs["delta_"], params["delta_"]))  # > 0

        return (T / lambda_) + (T / rho_) ** 0.5 + (T / delta_) ** 2


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
