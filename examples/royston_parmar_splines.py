# -*- coding: utf-8 -*-
"""
Below is a re-implementation of Royston and Parmar spline models,

Reference: Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment e􏰎ects
"""
from autograd import numpy as np
from lifelines.datasets import load_lymph_node
from lifelines.fitters import ParametricRegressionFitter
from autograd.scipy.special import expit


class SplineFitter:
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-12}

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def basis(self, x, knot, min_knot, max_knot):
        lambda_ = (max_knot - knot) / (max_knot - min_knot)
        return self.relu(x - knot) ** 3 - (
            lambda_ * self.relu(x - min_knot) ** 3 + (1 - lambda_) * self.relu(x - max_knot) ** 3
        )


class PHSplineFitter(SplineFitter, ParametricRegressionFitter):
    """
    Proportional Hazard model

    References
    ------------
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment effects. Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203 
    """

    _fitted_parameter_names = ["beta_", "phi0_", "phi1_", "phi2_"]
    _KNOWN_MODEL = True

    KNOTS = [0.1972, 1.769, 6.728]

    def _cumulative_hazard(self, params, T, Xs):
        exp_Xbeta = np.exp(np.dot(Xs["beta_"], params["beta_"]))
        lT = np.log(T)
        return exp_Xbeta * np.exp(
            params["phi0_"]
            + params["phi1_"] * lT
            + params["phi2_"] * self.basis(lT, np.log(self.KNOTS[1]), np.log(self.KNOTS[0]), np.log(self.KNOTS[-1]))
        )


class POSplineFitter(SplineFitter, ParametricRegressionFitter):
    """
    Proportional Odds model

    References
    ------------
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment effects. Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203 
    """

    _fitted_parameter_names = ["beta_", "phi0_", "phi1_", "phi2_"]
    _KNOWN_MODEL = True

    KNOTS = [0.1972, 1.769, 6.728]

    def _cumulative_hazard(self, params, T, Xs):
        Xbeta = np.dot(Xs["beta_"], params["beta_"])
        lT = np.log(T)

        return np.log1p(
            np.exp(
                Xbeta
                + (
                    params["phi0_"]
                    + params["phi1_"] * lT
                    + params["phi2_"]
                    * self.basis(lT, np.log(self.KNOTS[1]), np.log(self.KNOTS[0]), np.log(self.KNOTS[-1]))
                )
            )
        )


class WeibullFitter(ParametricRegressionFitter):
    """
    Alternative parameterization of Weibull Model


    """

    _fitted_parameter_names = ["beta_", "phi0_", "phi1_"]
    _scipy_fit_method = "SLSQP"
    _KNOWN_MODEL = True

    def _cumulative_hazard(self, params, T, Xs):
        exp_Xbeta = np.exp(np.dot(Xs["beta_"], params["beta_"]))
        lT = np.log(T)
        return exp_Xbeta * np.exp(params["phi0_"] + params["phi1_"] * lT)


df = load_lymph_node()

df["T"] = df["rectime"] / 365.0
df["E"] = df["censrec"]

df["constant"] = 1.0

# see paper for where these come from
df["linear_predictor"] = (
    1.79 * (df["age"] / 50) ** (-2)
    - 8.02 * (df["age"] / 50) ** (-0.5)
    + 0.5 * (df["grade"] >= 2).astype(int)
    - 1.98 * np.exp(-0.12 * df["nodes"])
    - 0.058 * (df["prog_recp"] + 1) ** 0.5
    - 0.394 * df["hormone"]
)
df["binned_lp"] = pd.qcut(df["linear_predictor"], np.linspace(0, 1, 4))
df = pd.get_dummies(df, columns=["binned_lp"], drop_first=True)

columns_needed_for_fitting = ["binned_lp_(-8.257, -7.608]", "binned_lp_(-7.608, -3.793]"] + ["constant"] + ["T", "E"]


# these values look right. Differences could be due to handling ties vs Stata
cph = CoxPHFitter().fit(df[columns_needed_for_fitting].drop("constant", axis=1), "T", "E").print_summary()


# check PH(1) Weibull model (different parameterization from lifelines)
regressors = {
    "beta_": ["binned_lp_(-8.257, -7.608]", "binned_lp_(-7.608, -3.793]"],
    "phi0_": ["constant"],
    "phi1_": ["constant"],
}
waf = WeibullFitter().fit(df[columns_needed_for_fitting], "T", "E", regressors=regressors).print_summary()


# Check PH(2) model
regressors = {
    "beta_": ["binned_lp_(-8.257, -7.608]", "binned_lp_(-7.608, -3.793]"],
    "phi0_": ["constant"],
    "phi1_": ["constant"],
    "phi2_": ["constant"],
}
phf = PHSplineFitter()
phf.fit(df[columns_needed_for_fitting], "T", "E", regressors=regressors).print_summary()


# Check PO(2) mode
regressors = {
    "beta_": ["binned_lp_(-8.257, -7.608]", "binned_lp_(-7.608, -3.793]"],
    "phi0_": ["constant"],
    "phi1_": ["constant"],
    "phi2_": ["constant"],
}
pof = POSplineFitter()
pof.fit(df[columns_needed_for_fitting], "T", "E", regressors=regressors).print_summary(5)

# looks like figure 2 from paper.
pof.predict_hazard(
    pd.DataFrame(
        {"binned_lp_(-8.257, -7.608]": [0, 0, 1], "binned_lp_(-7.608, -3.793]": [0, 1, 0], "constant": [1.0, 1, 1]}
    )
).plot()
plt.show()
