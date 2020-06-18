# -*- coding: utf-8 -*-

from autograd import numpy as np
from lifelines.fitters import ParametricRegressionFitter
from lifelines.fitters.mixins import SplineFitterMixin
from lifelines.utils.safe_exp import safe_exp


class CRCSplineFitter(SplineFitterMixin, ParametricRegressionFitter):
    """
    Below is an implementation of Crowther, Royston, Clements AFT cubic spline models.

    Parameters
    -----------

    n_baseline_knots: int
        the number of knots in the cubic spline. If equal to 2, then the model is equal to the WeibullAFT model.


    Reference
    ----------
    Crowther MJ, Royston P, Clements M. A flexible parametric accelerated failure time model.
    """

    _scipy_fit_method = "SLSQP"

    def __init__(self, n_baseline_knots: int, *args, **kwargs):
        assert n_baseline_knots > 1, "must be greater than 1"
        self.n_baseline_knots = n_baseline_knots
        self._fitted_parameter_names = ["beta_"] + ["gamma%d_" % i for i in range(0, self.n_baseline_knots)]
        super(CRCSplineFitter, self).__init__(*args, **kwargs)

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        return [
            {
                **{"beta_": np.zeros(len(Xs.mappings["beta_"])), "gamma0_": np.array([0.0]), "gamma1_": np.array([0.1])},
                **{"gamma%d_" % i: np.array([0.0]) for i in range(2, self.n_baseline_knots)},
            }
        ]

    def set_knots(self, T, E):
        self.knots = np.percentile(np.log(T[E.astype(bool).values]), np.linspace(5, 95, self.n_baseline_knots))

    def _pre_fit_model(self, Ts, E, df):
        self.set_knots(Ts[0], E)

    def _cumulative_hazard(self, params, T, Xs):
        # a negative sign makes the interpretation the same as other AFT models
        Xbeta = -np.dot(Xs["beta_"], params["beta_"])
        logT = np.log(T)

        H = safe_exp(params["gamma0_"] + params["gamma1_"] * (logT + Xbeta))

        for i in range(2, self.n_baseline_knots):
            H *= safe_exp(
                params["gamma%d_" % i]
                * self.basis(logT + Xbeta, self.knots[i - 1], min_knot=self.knots[0], max_knot=self.knots[-1])
            )
        return H
