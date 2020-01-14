# -*- coding: utf-8 -*-
from lifelines.fitters import KnownModelParametricUnivariateFitter
import autograd.numpy as np
from lifelines.utils.safe_exp import safe_exp
from lifelines import utils


class SplineFitterMixin:
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


class SplineFitter(SplineFitterMixin, KnownModelParametricUnivariateFitter):
    r"""
    Model the cumulative hazard using cubic splines. This offers great flexibility and smoothness of the cumulative hazard.

    .. math::

        H(t) = \exp{\phi_0 + \phi_1\log{t} + \sum_{j=2}^N \phi_j v_j(\log{t})

    where :math:`v_j` are our cubic basis functions at predetermined knots. See references for exact definition.

    Parameters
    -----------
    knot_locations: list, np.array
        The locations of the cubic breakpoints. Typically, the first knot is the minimum observed death, the last knot is the maximum observed death, and the knots in between
        are the centiles of observed data (ex: if one additional knot, choose the 50th percentile, the median. If two additional knots, choose the 33th and 66th percentiles).

    References
    ------------
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment effects. Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203 


    Examples
    --------

    >>> from lifelines import SplineFitter
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> T, E = waltons['T'], waltons['E']
    >>> knots = np.percentile(T.loc[E.astype(bool)], [0, 50, 100])
    >>> sf = SplineFitter(knots)
    >>> sf.fit()
    >>> sf.plot()
    >>> print(sf.knots)

    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    cumumlative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    median_survival_time_: float
        The median time to event
    lambda_: float
        The fitted parameter in the model
    rho_: float
        The fitted parameter in the model
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    knot_locations:
        The locations of the cubic breakpoints.

    """

    def __init__(self, knot_locations: np.ndarray, *args, **kwargs):
        self.knot_locations = knot_locations
        self.n_knots = len(self.knot_locations)
        self._fitted_parameter_names = ["phi_%d_" % i for i in range(self.n_knots)]
        self._bounds = [(None, None)] * (self.n_knots)
        super(SplineFitter, self).__init__(*args, **kwargs)

    def _create_initial_point(self, Ts, E, entry, weights):
        return 0.1 * np.ones(self.n_knots)

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
