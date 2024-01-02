# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd.numpy import exp, log
from scipy.special import gammainccinv, gammaincinv
from autograd_gamma import gammaincc, gammainc, gammaln, gammainccln, gammaincln
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils import CensoringType
from lifelines.utils.safe_exp import safe_exp
from autograd.scipy.stats import norm


class GeneralizedGammaFitter(KnownModelParametricUnivariateFitter):
    r"""

    This class implements a Generalized Gamma model for univariate data. The model has parameterized
    form:

    The survival function is:

    .. math::

        S(t)=\left\{  \begin{array}{}
           1-\Gamma_{RL}\left( \frac{1}{{{\lambda }^{2}}};\frac{{e}^{\lambda \left( \frac{\log(t)-\mu }{\sigma} \right)}}{\lambda ^{2}} \right)  \textit{ if } \lambda> 0 \\
              \Gamma_{RL}\left( \frac{1}{{{\lambda }^{2}}};\frac{{e}^{\lambda \left( \frac{\log(t)-\mu }{\sigma} \right)}}{\lambda ^{2}} \right)  \textit{ if } \lambda \le 0 \\
        \end{array} \right.\,\!

    where :math:`\Gamma_{RL}` is the regularized lower incomplete Gamma function.

    This model has the Exponential, Weibull, Gamma and Log-Normal as sub-models, and thus can be used as a way to test which
    model to use:

    1. When :math:`\lambda = 1` and :math:`\sigma = 1`, then the data is Exponential.
    2. When :math:`\lambda = 1` then the data is Weibull.
    3. When :math:`\sigma = \lambda` then the data is Gamma.
    4. When :math:`\lambda = 0` then the data is Log-Normal.
    5. When :math:`\lambda = -1` then the data is Inverse-Weibull.
    6. When :math:`\sigma = -\lambda` then the data is Inverse-Gamma.


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
    median_survival_time_: float
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

    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"maxiter": 10_000, "maxfev": 10_000}
    _fitted_parameter_names = ["mu_", "ln_sigma_", "lambda_"]
    _bounds = [(None, None), (None, None), (None, None)]
    _compare_to_values = np.array([0.0, 0.0, 1.0])

    def _create_initial_point(self, Ts, E, *args):
        if CensoringType.is_right_censoring(self):
            log_data = log(Ts[0])
        elif CensoringType.is_left_censoring(self):
            log_data = log(Ts[1])
        elif CensoringType.is_interval_censoring(self):
            # this fails if Ts[1] == Ts[0], so we add a some fudge factors.
            log_data = log(Ts[1] - Ts[0] + 0.1)
        return np.array([log_data.mean() * 1.5, log(log_data.std() + 0.1), 1.0])

    def _cumulative_hazard(self, params, times):
        mu_, ln_sigma_, lambda_ = params

        sigma_ = safe_exp(ln_sigma_)
        Z = (log(times) - mu_) / sigma_
        ilambda_2 = 1 / lambda_**2
        clipped_exp = np.clip(safe_exp(lambda_ * Z) * ilambda_2, 1e-300, 1e20)

        if lambda_ > 0:
            v = -gammainccln(ilambda_2, clipped_exp)
        elif lambda_ < 0:
            v = -gammaincln(ilambda_2, clipped_exp)
        else:
            v = -norm.logsf(Z)
        return v

    def _log_hazard(self, params, times):
        mu_, ln_sigma_, lambda_ = params
        ilambda_2 = 1 / lambda_**2
        Z = (log(times) - mu_) / safe_exp(ln_sigma_)
        clipped_exp = np.clip(safe_exp(lambda_ * Z) * ilambda_2, 1e-300, 1e20)
        if lambda_ > 0:
            v = (
                log(lambda_)
                - log(times)
                - ln_sigma_
                - gammaln(ilambda_2)
                + -2 * log(lambda_) * ilambda_2
                - clipped_exp
                + Z / lambda_
                - gammainccln(ilambda_2, clipped_exp)
            )
        elif lambda_ < 0:
            v = (
                log(-lambda_)
                - log(times)
                - ln_sigma_
                - gammaln(ilambda_2)
                - 2 * log(-lambda_) * ilambda_2
                - clipped_exp
                + Z / lambda_
                - gammaincln(ilambda_2, clipped_exp)
            )
        else:
            v = norm.logpdf(Z, loc=0, scale=1) - ln_sigma_ - log(times) - norm.logsf(Z)
        return v

    def percentile(self, p):
        lambda_ = self.lambda_
        sigma_ = exp(self.ln_sigma_)

        if lambda_ > 0:
            return exp(sigma_ * log(gammainccinv(1 / lambda_**2, p) * lambda_**2) / lambda_) * exp(self.mu_)
        return exp(sigma_ * log(gammaincinv(1 / lambda_**2, p) * lambda_**2) / lambda_) * exp(self.mu_)
