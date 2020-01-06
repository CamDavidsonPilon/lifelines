# -*- coding: utf-8 -*-
from lifelines.fitters import KnownModelParametricUnivariateFitter
import autograd.numpy as np
from lifelines.utils.safe_exp import safe_exp
from lifelines import utils


class SplineFitter:
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-10}

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def basis(self, x, knot, min_knot, max_knot):
        lambda_ = (max_knot - knot) / (max_knot - min_knot)
        return self.relu(x - knot) ** 3 - (
            lambda_ * self.relu(x - min_knot) ** 3 + (1 - lambda_) * self.relu(x - max_knot) ** 3
        )


class SplineFitter(SplineFitter, KnownModelParametricUnivariateFitter):
    r"""
    Model the cumulative hazard using cubic splines. This offers great flexibility and smoothness of the cumulative hazard.

    .. math::
        H(t) = \exp{\phi_0 + \phi_1\log{t} + \sum_{j=2}^N \phi_2 v_j(\log{t})

    where :math:`v_j` are our cubic basis functions at predetermined knots.

    Parameters
    -----------
    knot_locations: list, np.array
        The locations of the cubic breakpoints. Typically, the first knot is the minimum observed death, the last knot is the maximum observed death, and the knots in between
        are the centiles of observed data (ex: if one additional knot, choose the 50th percentile, the median. If two additional knots, choose the 33th and 66th percentiles).


    References
    ------------
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment effects. Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203 
    """

    def __init__(self, knot_locations: np.ndarray, *args, **kwargs):
        self.knot_locations = knot_locations
        self.n_knots = len(self.knot_locations)
        self._fitted_parameter_names = ["phi_%d_" % i for i in range(self.n_knots)]
        self._bounds = [(None, None)] * (self.n_knots)

        super(SplineFitter, self).__init__(*args, **kwargs)

    def _cumulative_hazard(self, params, t):
        phis = params
        lT = np.log(t)

        cum_haz = np.exp(phis[0] + phis[1] * lT)
        for i in range(2, self.n_knots):
            cum_haz = cum_haz * np.exp(
                phis[i]
                * self.basis(
                    lT,
                    np.log(self.knot_locations[i - 1]),
                    np.log(self.knot_locations[0]),
                    np.log(self.knot_locations[-1]),
                )
            )

        return cum_haz
