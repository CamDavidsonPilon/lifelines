# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import warnings
import time

import numpy as np
import pandas as pd
from scipy import stats

from numpy import dot, exp
from numpy.linalg import solve, norm, inv
from lifelines.fitters import BaseFitter

from lifelines.utils import inv_normal_cdf, \
    significance_code, normalize,\
    pass_for_numeric_dtypes_or_raise, check_low_var,\
    check_for_overlapping_intervals, check_complete_separation_low_variance,\
    ConvergenceWarning, StepSizer, _get_index


class CoxTimeVaryingFitter(BaseFitter):

    """
    This class implements fitting Cox's time-varying proportional hazard model:

    h(t|x(t)) = h_0(t)*exp(x(t)'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      penalizer: the coefficient of an l2 penalizer in the regression
    """

    def __init__(self, alpha=0.95, penalizer=0.0):
        self.alpha = alpha
        self.penalizer = penalizer

    def fit(self, df, id_col, event_col, start_col='start', stop_col='stop', show_progress=False, step_size=None):
        """
        Fit the Cox Propertional Hazard model to a time varying dataset. Tied survival times
        are handled using Efron's tie-method.

        Parameters:
          df: a Pandas dataframe with necessary columns `duration_col` and
             `event_col`, plus other covariates. `duration_col` refers to
             the lifetimes of the subjects. `event_col` refers to whether
             the 'death' events was observed: 1 if observed, 0 else (censored).
          id_col:  A subject could have multiple rows in the dataframe. This column contains
             the unique identifer per subject.
          event_col: the column in dataframe that contains the subjects' death
             observation. If left as None, assume all individuals are non-censored.
          start_col: the column that contains the start of a subject's time period.
          end_col: the column that contains the end of a subject's time period.
          show_progress: since the fitter is iterative, show convergence
             diagnostics.
          step_size: set an initial step size for the fitting algorithm.

        Returns:
            self, with additional properties: hazards_

        """

        df = df.copy()
        if not (id_col in df and event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the dataframe provided.")

        df = df.rename(columns={id_col: 'id', event_col: 'event', start_col: 'start', stop_col: 'stop'})
        df['event'] = df['event'].astype(bool)

        df = df.set_index(['id'])

        self._check_values(df.drop(["event", "stop", "start"], axis=1), df['event'])

        stop_times_events = df[["event", "stop", "start"]]
        df = df.drop(["event", "stop", "start"], axis=1)
        df = df.astype(float)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)

        hazards_ = self._newton_rhaphson(normalize(df, self._norm_mean, self._norm_std), stop_times_events, show_progress=show_progress,
                                         step_size=step_size)

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns, index=['coef']) / self._norm_std
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self._n_examples = df.shape[0]
        self._n_unique = df.index.unique().shape[0]

        return self

    @staticmethod
    def _check_values(df, E):
        # check_for_overlapping_intervals(df) # this is currenty too slow for production.
        check_low_var(df)
        check_complete_separation_low_variance(df, E)
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
        """
        Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns:
            df:.DataFrame, Contains columns coef, exp(coef), se(coef), z, p, lower, upper
        """

        df = pd.DataFrame(index=self.hazards_.columns)
        df['coef'] = self.hazards_.loc['coef'].values
        df['exp(coef)'] = exp(self.hazards_.loc['coef'].values)
        df['se(coef)'] = self._compute_standard_errors().loc['se'].values
        df['z'] = self._compute_z_values()
        df['p'] = self._compute_p_values()
        df['lower %.2f' % self.alpha] = self.confidence_intervals_.loc['lower-bound'].values
        df['upper %.2f' % self.alpha] = self.confidence_intervals_.loc['upper-bound'].values
        return df

    def _newton_rhaphson(self, df, stop_times_events, show_progress=False, step_size=None, precision=10e-6):
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            df: (n, d+2) Pandas DataFrame of observations with specific Interval multiindex.
            initial_beta: (1, d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.

        Returns:
            beta: (1,d) numpy array.
        """
        assert precision <= 1., "precision must be less than or equal to 1."

        n, d = df.shape

        # make sure betas are correct size.
        beta = np.zeros((d, 1))

        i = 0
        converging = True
        ll, previous_ll = 0, 0
        start = time.time()

        step_sizer = StepSizer(step_size)
        step_size = step_sizer.next()

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
            norm_delta = norm(delta)

            if show_progress:
                print("Iteration %d: norm_delta = %.6f, step_size = %.3f, ll = %.6f, seconds_since_start = %.1f" % (i, norm_delta, step_size, ll, time.time() - start))

            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif i >= 50:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is ~10 steps
                converging, completed = False, True
            elif step_size <= 0.0001:
                converging, completed = False, False
            elif abs(ll - previous_ll) < precision:
                converging, completed = False, True
            elif abs(ll) < 0.0001 and norm_delta > 1.0:
                warnings.warn("The log-likelihood is getting suspciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
See https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/ ", ConvergenceWarning)
                converging, completed = False, False

            step_size = step_sizer.update(norm_delta).next()

            beta += delta

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll
        self.event_observed = stop_times_events['event']

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        if not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    def _get_gradients(self, df, stops_events, beta):
        """
        Calculates the first and second order vector differentials, with respect to beta.

        Returns:
            hessian: (d, d) numpy array,
            gradient: (1, d) numpy array
            log_likelihood: double
        """

        _, d = df.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros(d)
        log_lik = 0

        unique_death_times = np.unique(stops_events['stop'].loc[stops_events['event']])

        for t in unique_death_times:

            ix = (stops_events['start'] < t) & (t <= stops_events['stop'])
            df_at_t = df.loc[ix]
            stops_events_at_t = stops_events.loc[ix]

            phi_i = exp(dot(df_at_t, beta))
            phi_x_i = phi_i * df_at_t
            phi_x_x_i = dot(df_at_t.T, phi_x_i) # dot(df_at_t.T, phi_i * df_at_t)

            # Calculate sums of Risk set
            risk_phi = phi_i.sum()
            risk_phi_x = phi_x_i.sum(0).values
            risk_phi_x_x = phi_x_x_i

            # Calculate the sums of Tie set
            deaths = stops_events_at_t['event'] & (stops_events_at_t['stop'] == t)
            death_counts = deaths.sum()  # should always be atleast 1.

            xi_deaths = df_at_t.loc[deaths]

            x_death_sum = xi_deaths.sum(0).values

            if death_counts > 1:
                # it's faster if we can skip computing these when we don't need to.
                tie_phi = phi_i[deaths.values].sum()
                tie_phi_x = phi_x_i.loc[deaths].sum(0).values
                tie_phi_x_x = dot(xi_deaths.T, phi_i[deaths.values] * xi_deaths)

            partial_gradient = np.zeros(d)

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
                a2 = np.outer(z / denom, z / denom)

                hessian -= (a1 - a2)

                log_lik -= np.log(denom)

            # Values outside tie sum
            gradient += x_death_sum - partial_gradient
            log_lik += dot(x_death_sum, beta)[0]

        return hessian, gradient.reshape(1, d), log_lik

    def predict_log_partial_hazard(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to \beta (X - \bar{X})

        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        if isinstance(X, pd.DataFrame):
            order = self.hazards_.columns
            X = X[order]

        index = _get_index(X)
        X = normalize(X, self._norm_mean.values, 1)
        return pd.DataFrame(np.dot(X, self.hazards_.T), index=index)


    def predict_partial_hazard(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.


        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to \exp{\beta X}
        """
        return exp(self.predict_log_partial_hazard(X))


    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        df = self.summary
        # Significance codes last
        df[''] = [significance_code(p) for p in df['p']]

        # Print information about data first
        print('periods={}, uniques={}, number of events={}'.format(self._n_examples, self._n_unique,
                                                                   self.event_observed.sum()),
              end='\n\n')
        print(df.to_string(float_format=lambda f: '{:4.4f}'.format(f)))
        # Significance code explanation
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ",
              end='\n\n')
        return

    def plot(self, standardized=False, columns=None, **kwargs):
        """
        Produces a visual representation of the fitted coefficients, including their standard errors and magnitudes.

        Parameters:
            standardized: standardize each estimated coefficient and confidence interval
                          endpoints by the standard error of the estimate.
            columns : list-like, default None
        Returns:
            ax: the matplotlib axis that be edited.

        """
        from matplotlib import pyplot as plt

        ax = kwargs.get('ax', None) or plt.figure().add_subplot(111)
        yaxis_locations = range(len(self.hazards_.columns))

        if columns is not None:
            yaxis_locations = range(len(columns))
            summary = self.summary.loc[columns]
            lower_bound = self.confidence_intervals_[columns].loc['lower-bound'].copy()
            upper_bound = self.confidence_intervals_[columns].loc['upper-bound'].copy()
            hazards = self.hazards_[columns].values[0].copy()
        else:
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

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifelines.%s: fitted with %d periods, %d uniques, %d events>""" % (
                classname, self._n_examples, self._n_unique, self.event_observed.sum())
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s
