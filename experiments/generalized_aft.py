# -*- coding: utf-8 -*-
from lifelines.fitters import ParametericRegressionFitter
import numpy as np
import autograd.numpy as anp


class Weibull(ParametericRegressionFitter):

    # TODO: check if _fitted_parameter_names is equal to keys in regressors!!
    _fitted_parameter_names = ["lambda_", "rho_"]

    def _cumulative_hazard(self, params, T, Xs):
        """
        One problem with this is that the dict Xs is going to be very memory consuming. Maybe I can create a new
        "dict" that has shared df and dynamically creates the df on indexing with [], that's cool

        """
        lambda_ = anp.exp(anp.dot(Xs["lambda_"], params["lambda_"]))  # > 0
        rho_ = anp.exp(anp.dot(Xs["rho_"], params["rho_"]))  # > 0

        return ((T) / lambda_) ** rho_


from lifelines.datasets import load_rossi

swf = Weibull(penalizer=1.0)
rossi = load_rossi()
rossi["intercept"] = 1.0

covariates = {
    "lambda_": ["intercept", "fin", "age", "race", "wexp", "mar", "paro", "prio"],  # need a shortcut for all columns?
    "rho_": ["intercept"],
}

swf.fit(rossi, "week", event_col="arrest", regressors=covariates)  # TODO: name
swf.print_summary()
