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
from lifelines import GeneralizedGammaFitter


class GeneralizedGammaRegressionFitter(ParametricRegressionFitter):
    r"""

    This class implements a Generalized Gamma model for regression data. The model has parameterized
    form:

    The survival function is:

    .. math::

        S(t; x)=\left\{  \begin{array}{}
           1-\Gamma_{RL}\left( \frac{1}{{{\lambda }^{2}}};\frac{{e}^{\lambda \left( \frac{\log(t)-\mu }{\sigma} \right)}}{\lambda ^{2}} \right)  \textit{ if } \lambda> 0 \\
              \Gamma_{RL}\left( \frac{1}{{{\lambda }^{2}}};\frac{{e}^{\lambda \left( \frac{\log(t)-\mu }{\sigma} \right)}}{\lambda ^{2}} \right)  \textit{ if } \lambda \le 0 \\
        \end{array} \right.\,\!

    where :math:`\Gamma_{RL}` is the regularized lower incomplete Gamma function, and :math:`\sigma = \sigma(x) = \exp(\alpha x^T), \lambda = \lambda(x) = \beta x^T, \mu = \mu(x) = \gamma x^T`.

    This model has the Exponential, Weibull, Gamma and Log-Normal as sub-models, and thus can be used as a way to test which
    model to use:

    1. When :math:`\lambda = 1` and :math:`\sigma = 1`, then the data is Exponential.
    2. When :math:`\lambda = 1` then the data is Weibull.
    3. When :math:`\sigma = \lambda` then the data is Gamma.
    4. When :math:`\lambda = 0` then the data is  Log-Normal.
    5. When :math:`\lambda = -1` then the data is Inverse-Weibull.
    6. When :math:`-\sigma = \lambda` then the data is Inverse-Gamma.


    After calling the ``.fit`` method, you have access to properties like: ``cumulative_hazard_``, ``survival_function_``,
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
    penalizer: float or array, optional (default=0.0)
        the penalizer coefficient to the size of the coefficients. See `l1_ratio`. Must be equal to or greater than 0.
        Alternatively, penalizer is an array equal in size to the number of parameters, with penalty coefficients for specific variables. For
        example, `penalizer=0.01 * np.ones(p)` is the same as `penalizer=0.01`

    Examples
    --------
    .. code:: python

        from lifelines import GeneralizedGammaFitter
        from lifelines.datasets import load_waltons
        waltons = load_waltons()

        ggf = GeneralizedGammaFitter()
        ggf.fit(waltons['T'], waltons['E'])
        ggf.plot()
        ggf.summary

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

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        # detect constant columns
        constant_col = (Xs.var(0) < 1e-8).idxmax()

        uni_model = GeneralizedGammaFitter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if utils.CensoringType.is_right_censoring(self):
                uni_model.fit_right_censoring(Ts[0], event_observed=E, entry=entries, weights=weights)
            elif utils.CensoringType.is_interval_censoring(self):
                uni_model.fit_interval_censoring(Ts[0], Ts[1], entry=entries, weights=weights)
            elif utils.CensoringType.is_left_censoring(self):
                uni_model.fit_left_censoring(Ts[1], event_observed=E, entry=entries, weights=weights)

            # we may use these later in log_likelihood_test()
            self._ll_null_ = uni_model.log_likelihood_
            assert self._ll_null_dof == 3

            default_point = super(GeneralizedGammaRegressionFitter, self)._create_initial_point(Ts, E, entries, weights, Xs)
            nested_point = {}

            nested_point["mu_"] = np.array([0.0] * (len(Xs["mu_"].columns)))
            if constant_col in Xs["mu_"].columns:
                nested_point["mu_"][Xs["mu_"].columns.index(constant_col)] = uni_model.mu_

            nested_point["sigma_"] = np.array([0.0] * (len(Xs["sigma_"].columns)))
            if constant_col in Xs["sigma_"].columns:
                nested_point["sigma_"][Xs["sigma_"].columns.index(constant_col)] = uni_model.ln_sigma_

            # this needs to be non-zero because we divide by it
            nested_point["lambda_"] = np.array([0.01] * (len(Xs["lambda_"].columns)))
            if constant_col in Xs["lambda_"].columns:
                nested_point["lambda_"][Xs["lambda_"].columns.index(constant_col)] = uni_model.lambda_

            return [nested_point, default_point]

    def _survival_function(self, params, T, Xs):
        lambda_ = Xs["lambda_"] @ params["lambda_"]
        sigma_ = safe_exp(Xs["sigma_"] @ params["sigma_"])
        mu_ = Xs["mu_"] @ params["mu_"]

        Z = (log(T) - mu_) / sigma_
        ilambda_2 = 1 / lambda_ ** 2
        exp_term = np.clip(safe_exp(lambda_ * Z) * ilambda_2, 1e-300, 1e25)

        return np.where(lambda_ > 0, gammaincc(ilambda_2, exp_term), gammainc(ilambda_2, exp_term))

    def _cumulative_hazard(self, params, T, Xs):
        lambda_ = Xs["lambda_"] @ params["lambda_"]
        sigma_ = safe_exp(Xs["sigma_"] @ params["sigma_"])
        mu_ = Xs["mu_"] @ params["mu_"]

        ilambda_2 = 1 / lambda_ ** 2
        Z = (log(T) - mu_) / np.clip(sigma_, 0, 1e20)
        exp_term = np.clip(safe_exp(lambda_ * Z) * ilambda_2, 1e-300, 1e25)

        return -np.where(lambda_ > 0, gammainccln(ilambda_2, exp_term), gammaincln(ilambda_2, exp_term))

    def _log_hazard(self, params, T, Xs):
        lambda_ = Xs["lambda_"] @ params["lambda_"]
        ln_sigma_ = Xs["sigma_"] @ params["sigma_"]
        mu_ = Xs["mu_"] @ params["mu_"]

        ilambda_2 = 1 / lambda_ ** 2
        Z = (log(T) - mu_) / np.clip(safe_exp(ln_sigma_), 0, 1e20)
        exp_term = np.clip(safe_exp(lambda_ * Z) * ilambda_2, 1e-300, 1e25)

        return (
            log(np.abs(lambda_))
            - log(T)
            - ln_sigma_
            - gammaln(ilambda_2)
            + (lambda_ * Z - 2 * log(np.abs(lambda_))) * ilambda_2
            - exp_term
            - np.where(lambda_ > 0, gammainccln(ilambda_2, exp_term), gammaincln(ilambda_2, exp_term))
        )
