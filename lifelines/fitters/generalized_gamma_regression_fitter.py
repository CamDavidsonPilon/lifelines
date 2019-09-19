# -*- coding: utf-8 -*-
import autograd.numpy as np
import warnings
from autograd.numpy import abs, log
from scipy.special import gammaincinv
from autograd_gamma import gammaincc, gammainc, gammaln, gammainccln, gammaincln
from lifelines.fitters import ParametricRegressionFitter
from lifelines.utils import CensoringType
from lifelines.utils.safe_exp import safe_exp
from lifelines import utils


class GeneralizedGammaRegressionFitter(ParametricRegressionFitter):
    r"""

    This class implements a Generalized Gamma model for regression data. The model has parameterized
    form:

    The survival function is:

    .. math::
        S(t \;|\; x)=\left\{ \begin{array}{}
           1-{{\Gamma}_{RL}}\left( \tfrac{1}{{{\lambda }^{2}}};\tfrac{{{e}^{\lambda \left( \tfrac{\text{ln}(t)-\mu }{\sigma } \right)}}}{{{\lambda }^{2}}} \right)\text{ if }\lambda >0  \\
           {{\Gamma}_{RL}}\left( \tfrac{1}{{{\lambda }^{2}}};\tfrac{{{e}^{\lambda \left( \tfrac{\text{ln}(t)-\mu }{\sigma } \right)}}}{{{\lambda }^{2}}} \right)\text{       if }\lambda < 0  \\
        \end{array} \right.\,\!

    where :math:`\Gamma_{RL}` is the regularized lower incomplete Gamma function, and :math:`\sigma = \exp(\alpha x^T), \lambda = \beta x^T, \mu = \gamma x^T`.

    This model has the Exponential, Weibull, Gamma and Log-Normal as sub-models, and thus can be used as a way to test which
    model to use:

    1. When :math:`\lambda = 1` and :math:`\sigma = 1`, then the data is Exponential.
    2. When :math:`\lambda = 1` then the data is Weibull.
    3. When :math:`\sigma = \lambda` then the data is Gamma.
    4. When :math:`\lambda = 0` then the data is  Log-Normal.
    5. When :math:`\lambda = -1` then the data is Inverse-Weibull.
    6. When :math:`-\sigma = \lambda` then the data is Inverse-Gamma.


    After calling the `.fit` method, you have access to properties like: ``cumulative_hazard_``, ``survival_function_``,
    A summary of the fit is available with the method ``print_summary()``.


    Important
    -------------
    The parameterization implemented has :math:`\log\sigma`, thus there is a `ln_sigma_` in the output. Exponentiate this parameter
    to recover :math:`\sigma`.


    Important
    -------------
    This model is experimental. It's API may change in the future. Also, it's convergence is not very stable.


    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.


    Examples
    --------

    >>> from lifelines import GeneralizedGammaFitter
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> ggf = GeneralizedGammaFitter()
    >>> ggf.fit(waltons['T'], waltons['E'])
    >>> ggf.plot()
    >>> ggf.summary

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
    median_: float
        The median time to event
    lambda_: float
        The fitted parameter in the model
    rho_: float
        The fitted parameter in the model
    alpha_: float
        The fitted parameter in the model
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    """
    _fitted_parameter_names = ["sigma_", "mu_", "lambda_"]
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-6, "maxiter": 200}

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        # detect constant columns
        constant_col = (Xs.df.var(0) < 1e-8).idxmax()

        import lifelines

        uni_model = lifelines.GeneralizedGammaFitter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if utils.CensoringType.is_right_censoring(self):
                uni_model.fit_right_censoring(Ts[0], event_observed=E, entry=entries, weights=weights)
            elif utils.CensoringType.is_interval_censoring(self):
                uni_model.fit_interval_censoring(Ts[0], Ts[1], event_observed=E, entry=entries, weights=weights)
            elif utils.CensoringType.is_left_censoring(self):
                uni_model.fit_left_censoring(Ts[1], event_observed=E, entry=entries, weights=weights)

            # we may use this later in print_summary
            self._ll_null_ = uni_model.log_likelihood_

            d = {}

            d["mu_"] = np.array([0.0] * (len(Xs.mappings["mu_"])))
            if constant_col in Xs.mappings["mu_"]:
                d["mu_"][Xs.mappings["mu_"].index(constant_col)] = uni_model.mu_

            d["sigma_"] = np.array([0.0] * (len(Xs.mappings["sigma_"])))
            if constant_col in Xs.mappings["mu_"]:
                d["sigma_"][Xs.mappings["sigma_"].index(constant_col)] = uni_model.ln_sigma_

            # this needs to be non-zero because we divide by it
            d["lambda_"] = np.array([0.01] * (len(Xs.mappings["lambda_"])))
            if constant_col in Xs.mappings["lambda_"]:
                d["lambda_"][Xs.mappings["lambda_"].index(constant_col)] = uni_model.lambda_

            return d

    def _survival_function(self, params, T, Xs):
        lambda_ = np.clip(Xs["lambda_"] @ params["lambda_"], 1e-25, 1e10)
        sigma_ = safe_exp(Xs["sigma_"] @ params["sigma_"])
        mu_ = Xs["mu_"] @ params["mu_"]

        Z = (log(T) - mu_) / sigma_
        ilambda_2 = 1 / lambda_ ** 2
        exp_term = safe_exp(lambda_ * Z - 2 * log(np.abs(lambda_)))

        return np.where(lambda_ > 0, gammaincc(ilambda_2, exp_term), gammainc(ilambda_2, exp_term))

    def _cumulative_hazard(self, params, T, Xs):
        lambda_ = Xs["lambda_"] @ params["lambda_"]
        sigma_ = safe_exp(Xs["sigma_"] @ params["sigma_"])
        mu_ = Xs["mu_"] @ params["mu_"]

        ilambda_2 = 1 / lambda_ ** 2
        Z = (log(T) - mu_) / np.clip(sigma_, 1e-10, 1e20)
        exp_term = np.clip(safe_exp(lambda_ * Z - 2 * log(np.abs(lambda_))), 1e-300, np.inf)

        return -np.where(lambda_ > 0, gammainccln(ilambda_2, exp_term), gammaincln(ilambda_2, exp_term))

    def _log_hazard(self, params, T, Xs):
        lambda_ = Xs["lambda_"] @ params["lambda_"]
        ln_sigma_ = Xs["sigma_"] @ params["sigma_"]
        mu_ = Xs["mu_"] @ params["mu_"]

        ilambda_2 = 1 / lambda_ ** 2
        Z = (log(T) - mu_) / np.clip(safe_exp(ln_sigma_), 1e-10, 1e20)
        exp_term = np.clip(safe_exp(lambda_ * Z - 2 * log(np.abs(lambda_))), 1e-300, np.inf)

        return (
            log(np.abs(lambda_))
            - log(T)
            - ln_sigma_
            - gammaln(ilambda_2)
            + (lambda_ * Z - 2 * log(np.abs(lambda_))) * ilambda_2
            - exp_term
            - np.where(lambda_ > 0, gammainccln(ilambda_2, exp_term), gammaincln(ilambda_2, exp_term))
        )
