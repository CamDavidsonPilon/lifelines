# -*- coding: utf-8 -*-

from typing import Optional
from autograd import numpy as np
from lifelines.fitters import ParametricRegressionFitter
from lifelines.fitters.mixins import SplineFitterMixin
from lifelines.utils.safe_exp import safe_exp


class CRCSplineFitter(SplineFitterMixin, ParametricRegressionFitter):
    """
    Below is an implementation of Crowther, Royston, Clements AFT cubic spline models. Internally, lifelines
    uses this for survival model probability calibration, but it can also be useful for a highly flexible AFT model.



    Parameters
    -----------

    n_baseline_knots: int
        the number of knots in the cubic spline.


    References
    ------------
    Crowther MJ, Royston P, Clements M. A flexible parametric accelerated failure time model.


    Examples
    ---------
    .. code:: python

        from lifelines import datasets, CRCSplineFitter
        rossi = datasets.load_rossi()

        regressors = {"beta_": "age + C(fin)", "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}
        crc = CRCSplineFitter(n_baseline_knots=3).fit(rossi, "week", "arrest", regressors=regressors)
        crc.print_summary()


    """

    _KNOWN_MODEL = True
    _FAST_MEDIAN_PREDICT = False

    fit_intercept = True
    _scipy_fit_method = "SLSQP"

    def __init__(self, n_baseline_knots: Optional[int] = None, knots: Optional[list] = None, *args, **kwargs):
        if n_baseline_knots is not None:
            assert n_baseline_knots > 1, "must be greater than 1"
            self.n_baseline_knots = n_baseline_knots
            self.knots = None
        elif knots is not None:
            assert len(knots) > 1
            self.knots = knots
            self.n_baseline_knots = len(self.knots)
        else:
            assert False, "Must supply n_baseline_knots or knots"

        self._fitted_parameter_names = ["beta_"] + ["gamma%d_" % i for i in range(0, self.n_baseline_knots)]
        super(CRCSplineFitter, self).__init__(*args, **kwargs)

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        return [
            {
                **{"beta_": np.zeros(len(Xs["beta_"].columns)), "gamma0_": np.array([0.0]), "gamma1_": np.array([2.0])},
                **{"gamma%d_" % i: np.array([0.0]) for i in range(2, self.n_baseline_knots)},
            }
        ]

    def set_knots(self, T, E):
        self.knots = np.percentile(np.log(T[E.astype(bool).values]), np.linspace(5, 95, self.n_baseline_knots))

    def _pre_fit_model(self, Ts, E, df):
        if self.knots is None:
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
