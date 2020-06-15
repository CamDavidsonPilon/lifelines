# -*- coding: utf-8 -*-
"""
Below is a re-implementation of Royston, Clements and Crowther spline models,

Crowther MJ, Royston P, Clements M. A flexible parametric accelerated failure time model.
"""
from autograd import numpy as np
from lifelines.fitters import ParametricRegressionFitter
from lifelines.fitters.mixins import SplineFitterMixin
from lifelines.utils.safe_exp import safe_exp


class CRCSplineFitter(SplineFitterMixin, ParametricRegressionFitter):

    _scipy_fit_method = "SLSQP"

    def __init__(self, n_baseline_knots, *args, **kwargs):
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


# section 4.1
def generate_data(n=20000):
    X = np.random.binomial(1, 0.5, size=n)
    Z = np.random.normal(0, 4, size=n)
    T_actual = np.random.exponential(1 / np.exp(-5 + 1 * X + 1 * Z))
    C = 10 * np.random.random(size=n)

    T_observed = np.minimum(T_actual, C)
    E = T_actual < C
    return pd.DataFrame({"X": X, "E": E, "T": T_observed, "Z": Z, "constant": 1})


df = generate_data()


regressors = {"beta_": ["X", "Z"], "gamma0_": ["constant"], "gamma1_": ["constant"], "gamma2_": ["constant"]}

cf = CRCSplineFitter(3).fit(df, "T", "E", regressors=regressors)
cf.print_summary()


# section 4.2
def generate_data(n=1000):
    from scipy.optimize import root_scalar

    p = 0.8
    gamma2 = 1.6
    lambda2 = 0.1
    gamma1 = 3
    lambda1 = 0.1
    beta = 0.5
    X = np.random.binomial(1, 0.5, size=n)

    # confirm that S(5).mean() == 0.106 ✅
    # h(t) looks like graph ✅
    def S(t, x):
        return p * np.exp(-lambda1 * (t ** gamma1) * np.exp(-x * beta * gamma1)) + (1 - p) * np.exp(
            -lambda2 * (t ** gamma2) * np.exp(-x * beta * gamma2)
        )

    T_actual = np.empty(n)

    for i in range(n):
        u = np.random.random()
        x = X[i]
        sol = root_scalar(lambda t: S(t, x) - u, x0=1, x1=3)
        assert sol.converged
        T_actual[i] = sol.root

    MAX_TIME = 5
    T_observed = np.minimum(MAX_TIME, T_actual)
    E = T_actual < MAX_TIME
    return pd.DataFrame({"E": E, "T": T_observed, "X": X, "constant": 1})


df = generate_data()

regressors = {"beta_": ["X"], "gamma0_": ["constant"], "gamma1_": ["constant"], "gamma2_": ["constant"], "gamma3_": ["constant"]}

WeibullAFTFitter().fit(df.drop("constant", axis=1), "T", "E").print_summary()
cf = CRCSplineFitter(4).fit(df, "T", "E", regressors=regressors)
cf.print_summary()
cf.predict_hazard(df)[[0, 1, 2, 3]].plot()


cph = CoxPHFitter(baseline_estimation_method="spline").fit(df.drop("constant", axis=1), "T", "E")
