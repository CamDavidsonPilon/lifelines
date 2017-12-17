# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from numpy import dot, exp
from numpy.linalg import solve, norm, inv
from lifelines.fitters import BaseFitter

from lifelines.utils import inv_normal_cdf, \
    significance_code, normalize,\
    pass_for_numeric_dtypes_or_raise, check_low_var,\
    check_for_overlapping_intervals, check_complete_separation


class CoxTimeVaryingFitter(BaseFitter):

    """
    This class implements fitting Cox's proportional hazard model:

    h(t|x(t)) = h_0(t)*exp(x(t)'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      penalizer: the coefficient of an l2 penalizer in the regression
    """

    def __init__(self, alpha=0.95, penalizer=0.0):
        self.alpha = alpha
        self.penalizer = penalizer

    def fit(self, df, id_col, event_col, start_col='start', stop_col='stop', show_progress=True, step_size=0.95):

        df = df.copy()
        if not (id_col in df and event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the dataframe provided.")

        df = df.rename(columns={id_col: 'id', event_col: 'event', start_col: 'start', stop_col: 'stop'})
        df['event'] = df['event'].astype(bool)

        df['interval'] = df.apply(lambda r: pd.Interval(r['start'], r['stop']), axis=1)
        df = df.drop(['start'], axis=1)\
               .set_index(['interval', 'id'])

        self._check_values(df.drop(["event", "stop"], axis=1), df['event'])

        stop_times_events = df[["event", "stop"]]
        df = df.drop(["event", "stop"], axis=1)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)

        hazards_ = self._newton_rhaphson(normalize(df, self._norm_mean, self._norm_std), stop_times_events, show_progress=show_progress,
                                         step_size=step_size)

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns, index=['coef']) / self._norm_std
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self._n_examples = df.shape[0]

        return self

    @staticmethod
    def _check_values(df, E):
        check_for_overlapping_intervals(df)
        check_low_var(df)
        check_complete_separation(df, E)
        pass_for_numeric_dtypes_or_raise(df)

    def _compute_standard_errors(self):
        se = np.sqrt(inv(-self._hessian_).diagonal()) / self._norm_std
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

    def _newton_rhaphson(self, df, stop_times_events, show_progress=False, step_size=0.5, precision=10e-6,):
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            df: (n,d+2) Pandas DataFrame of observations with specific Interval multiindex.
            initial_beta: (1,d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.

        Returns:
            beta: (1,d) numpy array.
        """
        assert precision <= 1., "precision must be less than or equal to 1."

        _, d = df.shape

        # make sure betas are correct size.
        beta = np.zeros((d, 1))

        i = 0
        converging = True
        ll = 0
        previous_ll = 0

        while converging:
            i += 1
            h, g, ll = self._get_gradients(df, stop_times_events, beta)

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
        self.event_observed = stop_times_events['event']
        if show_progress:
            print("Convergence completed after %d iterations." % (i))
        return beta

    def _get_gradients(self, df, stops_events, beta):
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
        # the below INDEX_ is a faster way to use the at_ function. See https://stackoverflow.com/questions/47621886
        # def at_(df, t):
        #     # this adds about a 100%+ runtime increase =(
        #     return df.iloc[(df.index.get_level_values(0).get_loc(t))]

        INDEX_df = df.index.get_level_values(0)
        INDEX_stops_events = stops_events.index.get_level_values(0)

        _, d = df.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros((1, d))
        log_lik = 0

        unique_death_times = np.sort(stops_events['stop'].loc[stops_events['event']].unique())

        for t in unique_death_times:
            x_death_sum = np.zeros((1, d))
            tie_phi, risk_phi = 0, 0
            risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))
            risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

            df_at_t = df.iloc[INDEX_df.get_loc(t)]
            stops_events_at_t = stops_events.iloc[INDEX_stops_events.get_loc(t)]

            phi_i = exp(dot(df_at_t, beta))
            phi_x_i = phi_i * df_at_t
            phi_x_x_i = dot(df_at_t.T, phi_i * df_at_t)

            # Calculate sums of Risk set
            risk_phi += phi_i.sum()
            risk_phi_x += phi_x_i.sum(0).values.reshape((1, d))
            risk_phi_x_x += phi_x_x_i

            # Calculate the sums of Tie set
            deaths = stops_events_at_t['event'] & (stops_events_at_t['stop'] == t)
            death_counts = deaths.sum()  # should always be atleast 1.

            xi_deaths = df_at_t.loc[deaths]
            x_death_sum += xi_deaths.sum(0).values.reshape((1, d))

            if death_counts > 1:
                # it's faster if we can skip computing these when we don't need to.
                tie_phi += phi_i[deaths.values].sum()
                tie_phi_x += phi_x_i.loc[deaths].sum(0).values.reshape((1, d))
                tie_phi_x_x += dot(xi_deaths.T, phi_i[deaths.values] * xi_deaths)

            partial_gradient = np.zeros((1, d))

            for l in range(death_counts):

                if death_counts > 1:
                    c = l / death_counts
                    denom = (risk_phi - c * tie_phi)
                    z = (risk_phi_x - c * tie_phi_x)
                    # Hessian
                    a1 = (risk_phi_x_x - c * tie_phi_x_x) / denom
                else:
                    denom = risk_phi
                    z = risk_phi_x
                    # Hessian
                    a1 = risk_phi_x_x / denom

                # Gradient
                partial_gradient += z / denom
                # In case z and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = dot(z.T / denom, z / denom)

                hessian -= (a1 - a2)

                log_lik -= np.log(denom).ravel()[0]

            # Values outside tie sum
            gradient += x_death_sum - partial_gradient
            log_lik += dot(x_death_sum, beta).ravel()[0]
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
        return

    def plot(self, standardized=False, **kwargs):
        """
        standardized: standardize each estimated coefficient and confidence interval endpoints by the standard error of the estimate.

        """
        from matplotlib import pyplot as plt

        ax = kwargs.get('ax', None) or plt.figure().add_subplot(111)
        yaxis_locations = range(len(self.hazards_.columns))

        summary = self.summary
        lower_bound = self.confidence_intervals_.loc['lower-bound'].copy()
        upper_bound = self.confidence_intervals_.loc['upper-bound'].copy()
        hazards = self.hazards_.values[0].copy()

        if standardized:
            se = summary['se(coef)']
            lower_bound /= se
            upper_bound /= se
            hazards /= se

        order = np.argsort(hazards)
        ax.scatter(upper_bound.values[order], yaxis_locations, marker='|', c='k')
        ax.scatter(lower_bound.values[order], yaxis_locations, marker='|', c='k')
        ax.scatter(hazards[order], yaxis_locations, marker='o', c='k')
        ax.hlines(yaxis_locations, lower_bound.values[order], upper_bound.values[order], color='k', lw=1)

        tick_labels = [c + significance_code(p).strip() for (c, p) in summary['p'][order].iteritems()]
        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("standardized coef" if standardized else "coef")
        return ax
