# -*- coding: utf-8 -*-
from typing import Optional
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.fitters.mixins import SplineFitterMixin
import numpy as np
import autograd.numpy as anp
from lifelines.utils.safe_exp import safe_exp


class SplineFitter(KnownModelParametricUnivariateFitter, SplineFitterMixin):
    r"""
    Model the cumulative hazard using :math:`N` cubic splines. This offers great flexibility and smoothness of the cumulative hazard.

    .. math:: H(t) = \exp{\left( \phi_0 + \phi_1\log{t} + \sum_{j=2}^N \phi_j v_j(\log{t})\right)}

    where :math:`v_j` are our cubic basis functions at predetermined knots. See references for exact definition.

    Parameters
    -----------
    knot_locations: list, np.array
        The locations of the cubic breakpoints. Must be length two or more. Typically, the first knot is the minimum observed death, the last knot is the maximum observed death, and the knots in between
        are the centiles of observed data (ex: if one additional knot, choose the 50th percentile, the median. If two additional knots, choose the 33rd and 66th percentiles).

    References
    ------------
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment effects. Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203


    Examples
    --------
    .. code:: python

        from lifelines import SplineFitter
        from lifelines.datasets import load_waltons
        waltons = load_waltons()

        T, E = waltons['T'], waltons['E']
        knots = np.percentile(T.loc[E.astype(bool)], [0, 50, 100])

        sf = SplineFitter(knots)
        sf.fit(T, E)
        sf.plot()
        print(sf.knots)

    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    cumulative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    density_: DataFrame
        The estimated density function (PDF) (with custom timeline if provided)
    variance_matrix_ : DataFrame
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
    knot_locations: array
        The locations of the breakpoints.
    n_knots: int
        Count of breakpoints

    """
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"maxiter": 1000}

    def __init__(self, knot_locations: np.ndarray, *args, **kwargs):
        self.knot_locations = knot_locations
        self.n_knots = len(self.knot_locations)
        assert self.n_knots > 1, "knot_locations must have two or more elements."
        self._fitted_parameter_names = ["phi_%d_" % i for i in range(self.n_knots)]
        self._bounds = [(None, None)] * (self.n_knots)
        self._compare_to_values = np.zeros(self.n_knots)
        super(SplineFitter, self).__init__(*args, **kwargs)

    def _create_initial_point(self, Ts, E, entry, weights):
        return 0.01 * np.ones(self.n_knots)

    def _cumulative_hazard(self, params, t):
        phis = params
        lT = anp.log(t)

        cum_haz = anp.exp(phis[0] + phis[1] * lT)
        for i in range(2, self.n_knots):
            cum_haz = cum_haz * anp.exp(
                phis[i]
                * self.basis(
                    lT, anp.log(self.knot_locations[i - 1]), anp.log(self.knot_locations[0]), anp.log(self.knot_locations[-1])
                )
            )

        return cum_haz
