# -*- coding: utf-8 -*-
from __future__ import print_function
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from numpy import dot, exp
from numpy.linalg import solve, norm, inv
from lifelines.fitters import BaseFitter

from lifelines.utils import inv_normal_cdf, \
    significance_code, concordance_index, qth_survival_times,\
    pass_for_numeric_dtypes_or_raise, check_low_var, coalesce,\
    check_for_overlapping_intervals


def at_(df, t):
    return df.iloc[(df.index.get_level_values(0).get_loc(t))]


class CoxTimeVaryingFitter(BaseFitter):

    """
    This class implements fitting Cox's proportional hazard model:

    h(t|x) = h_0(t)*exp(x'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      penalizer:
    """

    def __init__(self, alpha=0.95, penalizer=0.0):
        self.alpha = alpha
        self.penalizer = penalizer

    def fit(self, df, id_col, event_col, start_col='start', stop_col='stop', show_progress=True, step_size=0.5):

        df = df.copy()
        df = df.rename(columns={id_col: 'id', event_col: 'event', start_col: 'start', stop_col: 'stop'})
        df['event'] = df['event'].astype(bool)

        df['interval'] = df.apply(lambda r: pd.Interval(r['start'], r['stop']), axis=1)
        df = df.drop(['start'], axis=1)\
               .set_index(['interval', 'id'])

        self._check_values(df.drop(["event", "stop"], axis=1))
        hazards_ = self._newton_rhaphson(df, show_progress=show_progress,
                                         step_size=step_size)

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns, index=['coef'])
        self.confidence_intervals_ = self._compute_confidence_intervals()

    @staticmethod
    def _check_values(df):
        check_for_overlapping_intervals(df)
        check_low_var(df)
        pass_for_numeric_dtypes_or_raise(df)

    def _compute_standard_errors(self):
        se = np.sqrt(inv(-self._hessian_).diagonal())
        return pd.DataFrame(se[None, :],
                            index=['se'], columns=self.hazards_.columns)

    def _compute_z_values(self):
        return (self.hazards_.loc['coef'] /
                self._compute_standard_errors().loc['se'])

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1. + self.alpha) / 2.)
        se = self._compute_standard_errors()
        hazards = self.hazards_.values
        return pd.DataFrame(np.r_[hazards - alpha2 * se,
                                  hazards + alpha2 * se],
                            index=['lower-bound', 'upper-bound'],
                            columns=self.hazards_.columns)

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper"""

        df = pd.DataFrame(index=self.hazards_.columns)
        df['coef'] = self.hazards_.loc['coef'].values
        df['exp(coef)'] = exp(self.hazards_.loc['coef'].values)
        df['se(coef)'] = self._compute_standard_errors().loc['se'].values
        df['z'] = self._compute_z_values()
        df['p'] = self._compute_p_values()
        df['lower %.2f' % self.alpha] = self.confidence_intervals_.loc['lower-bound'].values
        df['upper %.2f' % self.alpha] = self.confidence_intervals_.loc['upper-bound'].values
        return df

    def _newton_rhaphson(self, df, show_progress=True, step_size=0.5, precision=10e-6,):
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            df: (n,d+2) Pandas DataFrame of observations with specific Interval multiindex.
            initial_beta: (1,d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0.001 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.

        Returns:
            beta: (1,d) numpy array.
        """
        assert precision <= 1., "precision must be less than or equal to 1."

        stop_times = df.pop("stop")
        events = df.pop("event")

        _, d = df.shape

        # make sure betas are correct size.
        beta = np.zeros((d, 1))

        i = 0
        converging = True
        ll = 0
        previous_ll = 0

        while converging:
            i += 1
            h, g, ll = self._get_gradients(df, stop_times, events, beta)

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta.T
                h.flat[::d + 1] -= self.penalizer

            delta = solve(-h, step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ValueError("delta contains nan value(s). Convergence halted.")

            # Save these as pending result
            hessian, gradient = h, g

            if show_progress:
                print("Iteration %d: norm_delta = %.5f, step_size = %.5f, ll = %.5f" % (i, norm(delta), step_size, ll))

            # convergence criteria
            if norm(delta) < precision:
                converging = False
            elif abs(ll - previous_ll) < precision:
                converging = False
            elif i >= 50:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is ~10 steps
                warnings.warn("Newton-Rhapson failed to converge sufficiently in 50 steps.", RuntimeWarning)
                converging = False
            elif step_size <= 0.0001:
                converging = False

            # Only allow small steps
            if norm(delta) > 10:
                step_size *= 0.5

            # anneal the step size down.
            step_size *= 0.99

            beta += delta
            previous_ll = ll

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll
        if show_progress:
            print("Convergence completed after %d iterations." % (i))
        return beta

    def _get_gradients(self, df, stop_times, events, beta):
        """
        return the gradient, hessian, and log-like
        """
        """
        Calculates the first and second order vector differentials,
        with respect to beta.

        Returns:
            hessian: (d, d) numpy array,
            gradient: (1, d) numpy array
            log_likelihood: double
        """

        _, d = df.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros((1, d))
        log_lik = 0

        unique_death_times = np.sort(stop_times.loc[events])

        for t in unique_death_times:
            x_tie_sum = np.zeros((1, d))
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            tie_phi_x_x = np.zeros((d, d))
            risk_phi = 0
            risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))
            risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

            xi = at_(df, t)
            phi_i = exp(dot(xi, beta))
            phi_x_i = phi_i * xi
            phi_x_x_i = dot(xi.T, phi_i * xi)

            # Calculate sums of Risk set
            risk_phi += phi_i.sum()
            risk_phi_x += phi_x_i.sum(0).values.reshape((1, d))
            risk_phi_x_x += phi_x_x_i

            # Calculate the sums of Tie set
            deaths = (at_(stop_times, t) == t) & (at_(events, t))
            death_counts = deaths.sum()
            xi_deaths = xi.loc[deaths]
            x_tie_sum += xi_deaths.sum(0).values.reshape((1, d))

            tie_phi += phi_i[deaths.values].sum()
            tie_phi_x += phi_x_i.loc[deaths].sum(0).values.reshape((1, d))
            tie_phi_x_x += dot(xi_deaths.T, phi_i[deaths.values] * xi_deaths)

            partial_gradient = np.zeros((1, d))

            for l in range(death_counts):
                c = l / death_counts

                denom = (risk_phi - c * tie_phi)
                z = (risk_phi_x - c * tie_phi_x)

                # Gradient
                partial_gradient += z / denom
                # Hessian
                a1 = (risk_phi_x_x - c * tie_phi_x_x) / denom
                # In case z and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = dot(z.T / denom, z / denom)

                hessian -= (a1 - a2)

                log_lik -= np.log(denom).ravel()[0]

            # Values outside tie sum
            gradient += x_tie_sum - partial_gradient
            log_lik += dot(x_tie_sum, beta).ravel()[0]

        return hessian, gradient, log_lik

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        df = self.summary
        # Significance codes last
        df[''] = [significance_code(p) for p in df['p']]

        # Print information about data first
        print('n={}, number of events={}'.format(self._n_examples,
                                                 np.where(self.event_observed)[0].shape[0]),
              end='\n\n')
        print(df.to_string(float_format=lambda f: '{:4.4f}'.format(f)))
        # Significance code explanation
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ",
              end='\n\n')
        print("Concordance = {:.3f}".format(self.score_))
        return
