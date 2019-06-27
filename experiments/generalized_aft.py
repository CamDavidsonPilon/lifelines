# -*- coding: utf-8 -*-
import collections
from functools import wraps
import sys
import warnings
from datetime import datetime

# pylint: disable=wrong-import-position
warnings.simplefilter(action="ignore", category=FutureWarning)

from textwrap import dedent

import numpy as np
import autograd.numpy as anp
from autograd import hessian, value_and_grad, elementwise_grad as egrad, grad
from autograd.differential_operators import make_jvp_reversemode
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from numpy.linalg import inv, pinv


from lifelines.plotting import _plot_estimate, set_kwargs_drawstyle, set_kwargs_ax
from lifelines.fitters import BaseFitter

from lifelines.utils import (
    qth_survival_times,
    _to_array,
    _to_list,
    safe_zip,
    dataframe_interpolate_at_times,
    ConvergenceError,
    inv_normal_cdf,
    string_justify,
    format_floats,
    format_p_value,
    format_exp_floats,
    coalesce,
    check_nans_or_infs,
    pass_for_numeric_dtypes_or_raise_array,
    check_for_numeric_dtypes_or_raise,
    check_complete_separation,
    check_low_var,
    check_positivity,
    StatisticalWarning,
    StatError,
    median_survival_times,
    normalize,
    concordance_index,
    CensoringType,
)


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
