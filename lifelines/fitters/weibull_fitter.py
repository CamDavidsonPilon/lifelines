# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import pandas as pd

from numpy.linalg import solve, norm, inv
from lifelines.fitters import UnivariateFitter
from lifelines.utils import inv_normal_cdf


def _negative_log_likelihood(lambda_rho, T, E):
    if np.any(lambda_rho < 0):
        return np.inf
    lambda_, rho = lambda_rho
    return - np.log(rho * lambda_) * E.sum() - (rho - 1) * (E * np.log(lambda_ * T)).sum() + ((lambda_ * T) ** rho).sum()


def _lambda_gradient(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return - rho * (E / lambda_ - (lambda_ * T) ** rho / lambda_).sum()


def _rho_gradient(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return - E.sum() / rho - (np.log(lambda_ * T) * E).sum() + (np.log(lambda_ * T) * (lambda_ * T) ** rho).sum()
    # - D/p - D Log[m t] + (m t)^p Log[m t]


def _d_rho_d_rho(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (1. / rho ** 2 * E + (np.log(lambda_ * T) ** 2 * (lambda_ * T) ** rho)).sum()
    # (D/p^2) + (m t)^p Log[m t]^2


def _d_lambda_d_lambda_(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (rho / lambda_ ** 2) * (E + (rho - 1) * (lambda_ * T) ** rho).sum()


def _d_rho_d_lambda_(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (-1. / lambda_) * (E - (lambda_ * T) ** rho - rho * (lambda_ * T) ** rho * np.log(lambda_ * T)).sum()


class WeibullFitter(UnivariateFitter):

    """
    This class implements a Weibull model for univariate data. The model has parameterized
    form:

      S(t) = exp(-(lambda*t)**rho),   lambda >0, rho > 0,

    which implies the cumulative hazard rate is

      H(t) = (lambda*t)**rho,

    and the hazard rate is:

      h(t) = rho*lambda(lambda*t)**(rho-1)

    After calling the `.fit` method, you have access to properties like:
    `cumulative_hazard_', 'survival_function_', 'lambda_' and 'rho_'.

    """

    def fit(self, durations, event_observed=None, timeline=None, entry=None,
            label='Weibull_estimate', alpha=None, ci_labels=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the estimate at the values in timeline (postively increasing)
          event_observed: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>

        Returns:
          self, with new properties like `cumulative_hazard_', 'survival_function_', 'lambda_' and 'rho_'.

        """
        self.durations = np.asarray(durations, dtype=float)
        # check for negative or 0 durations - these are not allowed in a weibull model.
        if np.any(self.durations <= 0):
            raise ValueError('This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements.')

        self.event_observed = np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        self.timeline = np.sort(np.asarray(timeline)) if timeline is not None else np.arange(int(self.durations.min()), int(self.durations.max()) + 1)
        self._label = label
        alpha = alpha if alpha is not None else self.alpha

        # estimation
        self.lambda_, self.rho_ = self._newton_rhaphson(self.durations, self.event_observed)
        self.survival_function_ = pd.DataFrame(self.survival_function_at_times(self.timeline), columns=[self._label], index=self.timeline)
        self.hazard_ = pd.DataFrame(self.hazard_at_times(self.timeline), columns=[self._label], index=self.timeline)
        self.cumulative_hazard_ = pd.DataFrame(self.cumulative_hazard_at_times(self.timeline), columns=[self._label], index=self.timeline)
        self.confidence_interval_ = self._bounds(alpha, ci_labels)
        self.median_ = 1. / self.lambda_ * (np.log(2)) ** (1. / self.rho_)

        # estimation functions - Cumulative hazard takes priority.
        self.predict = self._predict("cumulative_hazard_", self._label)
        self.subtract = self._subtract("cumulative_hazard_")
        self.divide = self._divide("cumulative_hazard_")

        # plotting - Cumulative hazard takes priority.
        self.plot = self._plot_estimate("cumulative_hazard_")
        self.plot_cumulative_hazard = self.plot

        return self

    def hazard_at_times(self, times):
        return self.lambda_ * self.rho_ * (self.lambda_ * times) ** (self.rho_ - 1)

    def survival_function_at_times(self, times):
        return np.exp(-self.cumulative_hazard_at_times(times))

    def cumulative_hazard_at_times(self, times):
        return (self.lambda_ * times) ** self.rho_

    def _newton_rhaphson(self, T, E, precision=1e-5):
        from lifelines.utils import _smart_search

        def jacobian_function(parameters, T, E):
            return np.array([
                [_d_lambda_d_lambda_(parameters, T, E), _d_rho_d_lambda_(parameters, T, E)],
                [_d_rho_d_lambda_(parameters, T, E), _d_rho_d_rho(parameters, T, E)]
            ])

        def gradient_function(parameters, T, E):
            return np.array([_lambda_gradient(parameters, T, E), _rho_gradient(parameters, T, E)])

        # initialize the parameters. This shows dramatic improvements.
        parameters = _smart_search(_negative_log_likelihood, 2, T, E)

        iter = 1
        step_size = 1.
        converging = True

        while converging and iter < 50:
            # Do not override hessian and gradient in case of garbage
            j, g = jacobian_function(parameters, T, E), gradient_function(parameters, T, E)

            delta = solve(j, - step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ValueError("delta contains nan value(s). Convergence halted.")

            parameters += delta

            # Save these as pending result
            jacobian = j

            if norm(delta) < precision:
                converging = False
            iter += 1

        self._jacobian = jacobian
        return parameters

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1. + alpha) / 2.)
        df = pd.DataFrame(index=self.timeline)
        var_lambda_, var_rho_ = inv(self._jacobian).diagonal()

        def _dH_d_lambda(lambda_, rho, T):
            return rho / lambda_ * (lambda_ * T) ** rho

        def _dH_d_rho(lambda_, rho, T):
            return np.log(lambda_ * T) * (lambda_ * T) ** rho

        def sensitivity_analysis(lambda_, rho, var_lambda_, var_rho_, T):
            return var_lambda_ * _dH_d_lambda(lambda_, rho, T) ** 2 + var_rho_ * _dH_d_rho(lambda_, rho, T) ** 2

        std_cumulative_hazard = np.sqrt(sensitivity_analysis(self.lambda_, self.rho_, var_lambda_, var_rho_, self.timeline))

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = self.cumulative_hazard_at_times(self.timeline) + alpha2 * std_cumulative_hazard
        df[ci_labels[1]] = self.cumulative_hazard_at_times(self.timeline) - alpha2 * std_cumulative_hazard
        return df

    def _compute_standard_errors(self):
        var_lambda_, var_rho_ = inv(self._jacobian).diagonal()
        return pd.DataFrame([[np.sqrt(var_lambda_), np.sqrt(var_rho_)]],
                            index=['se'], columns=['lambda_', 'rho_'])

    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().ix['se']
        alpha2 = inv_normal_cdf((1. + self.alpha) / 2.)
        return pd.DataFrame([
            np.array([self.lambda_, self.rho_]) + alpha2 * se,
            np.array([self.lambda_, self.rho_]) - alpha2 * se,
        ], columns=['lambda_', 'rho_'], index=['upper-bound', 'lower-bound'])

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper"""
        lower_upper_bounds = self._compute_confidence_bounds_of_parameters()
        df = pd.DataFrame(index=['lambda_', 'rho_'])
        df['coef'] = [self.lambda_, self.rho_]
        df['se(coef)'] = self._compute_standard_errors().ix['se']
        df['lower %.2f' % self.alpha] = lower_upper_bounds.ix['lower-bound']
        df['upper %.2f' % self.alpha] = lower_upper_bounds.ix['upper-bound']
        return df

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        df = self.summary

        # Print information about data first
        print('n={}, number of events={}'.format(self.durations.shape[0],
                                                 np.where(self.event_observed)[0].shape[0]),
              end='\n\n')
        print(df.to_string(float_format=lambda f: '{:.3e}'.format(f)))
        return
