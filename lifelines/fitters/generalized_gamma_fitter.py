# -*- coding: utf-8 -*-
import autograd.numpy as np
from scipy.special import gammainccinv
from lifelines.fitters import KnownModelParametericUnivariateFitter
from lifelines.utils.gamma import gammaincc, gammainc, gamma, gammaln, gammainccln, gammaincln


class GeneralizedGammaFitter(KnownModelParametericUnivariateFitter):
    r"""

    This class implements a Generalized Gamma model for univariate data. The model has parameterized
    form:

    The survival function is:

    .. math:: S(t) = 1-\text{RLG}(\frac{\alpha}{\rho}, \left(\frac{t}{\lambda}\right)^{\rho})

    Where RLG is the regularized lower incomplete gamma function. The cumulative hazard rate is

    .. math:: H(t) = -\log(1-\text{RLG}(\frac{\alpha}{\rho}, \left(\frac{t}{\lambda}\right)^{\rho}))

    This model has the Exponential, Weibull, Gamma and LogNormal as sub-models, and thus can be used as a way to test which
    model to use.

    1. When :math:`\alpha \approx 1` and :math:`\rho \approx 1`, then the data is likely Exponential.
    2. When :math:`\alpha \approx \rho` then the data is likely Weibull.
    3. When :math:`\rho \approx 1` then the data is likely Gamma.
    4. When :math:`\alpha >> 0, \lambda \approx 0, \rho > 0` then the data is likely LogNormal.


    After calling the `.fit` method, you have access to properties like: ``cumulative_hazard_``, ``survival_function_``, ``alpha_``, ``lambda_`` and ``rho_``.
    A summary of the fit is available with the method ``print_summary()``.

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

    _fitted_parameter_names = ["mu_", "ln_sigma_", "lambda_"]
    _bounds = [(None, None), (None, None), (None, None)]
    _initial_values = np.array([0.0, 0.0, 1.0])

    def _survival_function(self, params, times):
        mu_, ln_sigma_, lambda_ = params
        sigma_ = np.exp(ln_sigma_)
        Z = (np.log(times) - mu_) / sigma_
        if lambda_ > 0:
            return gammaincc(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
        else:
            return gammainc(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)

    def _cumulative_hazard(self, params, times):
        mu_, ln_sigma_, lambda_ = params
        sigma_ = np.exp(ln_sigma_)
        Z = (np.log(times) - mu_) / sigma_

        if lambda_ > 0:
            v = -gammainccln(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
        else:
            v = -gammaincln(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
        return v

    def _log_1m_sf(self, params, times):
        mu_, ln_sigma_, lambda_ = params
        sigma_ = np.exp(ln_sigma_)

        Z = (np.log(times) - mu_) / sigma_
        if lambda_ > 0:
            v = gammaincln(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
        else:
            v = gammainccln(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
        return v

    def _log_hazard(self, params, times):
        log = np.log
        mu_, ln_sigma_, lambda_ = params
        sigma_ = np.exp(ln_sigma_)

        Z = (log(times) - mu_) / sigma_
        if lambda_ > 0:
            v = (
                log(np.abs(lambda_))
                - log(times)
                - log(sigma_)
                - gammaln(1 / lambda_ ** 2)
                + (lambda_ * Z - np.exp(lambda_ * Z) - 2 * log(lambda_)) / lambda_ ** 2
                - gammainccln(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
            )
        else:
            v = (
                log(np.abs(lambda_))
                - log(times)
                - log(sigma_)
                - gammaln(1 / lambda_ ** 2)
                + (lambda_ * Z - np.exp(lambda_ * Z) - 2 * log(np.abs(lambda_))) / lambda_ ** 2
                - gammaincln(1 / lambda_ ** 2, np.exp(lambda_ * Z) / lambda_ ** 2)
            )
        return v

    def percentile(self, p):
        # TODO
        return 0.5
