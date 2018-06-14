# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import time
import warnings
import numpy as np
import pandas as pd

from numpy import dot, exp
from numpy.linalg import solve, norm, inv
from scipy.integrate import trapz
import scipy.stats as stats

from lifelines.fitters import BaseFitter
from lifelines.utils import survival_table_from_events, inv_normal_cdf, normalize,\
    significance_code, concordance_index, _get_index, qth_survival_times,\
    pass_for_numeric_dtypes_or_raise, check_low_var, coalesce,\
    check_complete_separation, check_nans, StatError, ConvergenceWarning,\
    StepSizer


class CoxPHFitter(BaseFitter):

    """
    This class implements fitting Cox's proportional hazard model:

    h(t|x) = h_0(t)*exp(x'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      tie_method: specify how the fitter should deal with ties. Currently only
        'Efron' is available.
      penalizer: Attach a L2 penalizer to the size of the coeffcients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the absolute value of beta_i. Recommended, even if a small value.
        The penalty is 1/2 * penalizer * ||beta||^2.
      strata: specify a list of columns to use in stratification. This is useful if a
         catagorical covariate does not obey the proportional hazard assumption. This
         is used similar to the `strata` expression in R.
         See http://courses.washington.edu/b515/l17.pdf.
    """

    def __init__(self, alpha=0.95, tie_method='Efron', penalizer=0.0, strata=None):
        if not (0 < alpha <= 1.):
            raise ValueError('alpha parameter must be between 0 and 1.')
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")
        if tie_method != 'Efron':
            raise NotImplementedError("Only Efron is available atm.")

        self.alpha = alpha
        self.tie_method = tie_method
        self.penalizer = penalizer
        self.strata = strata

    def fit(self, df, duration_col, event_col=None,
            show_progress=False, initial_beta=None,
            strata=None, step_size=None, weights_col=None):
        """
        Fit the Cox Propertional Hazard model to a dataset. Tied survival times
        are handled using Efron's tie-method.

        Parameters:
          df: a Pandas dataframe with necessary columns `duration_col` and
             `event_col`, plus other covariates. `duration_col` refers to
             the lifetimes of the subjects. `event_col` refers to whether
             the 'death' events was observed: 1 if observed, 0 else (censored).
          duration_col: the column in dataframe that contains the subjects'
             lifetimes.
          event_col: the column in dataframe that contains the subjects' death
             observation. If left as None, assume all individuals are non-censored.
          weights_col: an optional column in the dataframe that denotes the weight per subject.
             This column is expelled and not used as a covariate, but as a weight in the
             final regression. Default weight is 1.
          show_progress: since the fitter is iterative, show convergence
             diagnostics.
          initial_beta: initialize the starting point of the iterative
             algorithm. Default is the zero vector.
          strata: specify a list of columns to use in stratification. This is useful if a
             catagorical covariate does not obey the proportional hazard assumption. This
             is used similar to the `strata` expression in R.
             See http://courses.washington.edu/b515/l17.pdf.
          step_size: set an initial step size for the fitting algorithm.

        Returns:
            self, with additional properties: hazards_

        """

        df = df.copy()

        # Sort on time
        df = df.sort_values(by=duration_col)

        self._n_examples = df.shape[0]
        self.strata = coalesce(strata, self.strata)
        if self.strata is not None:
            original_index = df.index.copy()
            df = df.set_index(self.strata)

        # Extract time and event
        T = df[duration_col]
        del df[duration_col]
        if event_col is None:
            E = pd.Series(np.ones(df.shape[0]), index=df.index)
        else:
            E = df[event_col]
            del df[event_col]

        if weights_col:
            weights = df.pop(weights_col)
            if (weights.astype(int) != weights).any():
                warnings.warn("""It looks like your weights are not integers, possibly propensity scores then?
It's important to know that the naive variance estimates of the coefficients are biased. Instead use Monte Carlo to
estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
                    """, RuntimeWarning)

        else:
            weights = pd.DataFrame(np.ones((self._n_examples, 1)), index=df.index)

        self._check_values(df, T, E)
        df = df.astype(float)

        # save fitting data for later
        self.durations = T.copy()
        self.event_observed = E.copy()
        if self.strata is not None:
            self.durations.index = original_index
            self.event_observed.index = original_index
        self.event_observed = self.event_observed.astype(bool)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)

        E = E.astype(bool)

        hazards_ = self._newton_rhaphson(normalize(df, self._norm_mean, self._norm_std), T, E,
                                         weights=weights,
                                         initial_beta=initial_beta,
                                         show_progress=show_progress,
                                         step_size=step_size)

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns, index=['coef']) / self._norm_std
        self.confidence_intervals_ = self._compute_confidence_intervals()

        self.baseline_hazard_ = self._compute_baseline_hazards(df, T, E)
        self.baseline_cumulative_hazard_ = self._compute_baseline_cumulative_hazard()
        self.baseline_survival_ = self._compute_baseline_survival()
        self.score_ = concordance_index(self.durations,
                                        -self.predict_partial_hazard(df).values.ravel(),
                                        self.event_observed)
        self._train_log_partial_hazard = self.predict_log_partial_hazard(self._norm_mean.to_frame().T)
        return self

    def _newton_rhaphson(self, X, T, E, weights=None, initial_beta=None, step_size=None,
                         precision=10e-6, show_progress=True, max_steps=50):
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            X: (n,d) Pandas DataFrame of observations.
            T: (n) Pandas Series representing observed durations.
            E: (n) Pandas Series representing death events.
            weights: (n) an iterable representing weights per observation.
            initial_beta: (1,d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0.001 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.
            show_progress: since the fitter is iterative, show convergence
                     diagnostics.
            max_steps: the maximum number of interations of the Newton-Rhaphson algorithm.

        Returns:
            beta: (1,d) numpy array.
        """
        self.path = []
        assert precision <= 1., "precision must be less than or equal to 1."
        n, d = X.shape

        # make sure betas are correct size.
        if initial_beta is not None:
            assert initial_beta.shape == (d, 1)
            beta = initial_beta
        else:
            beta = np.zeros((d, 1))

        step_sizer = StepSizer(step_size)
        step_size = step_sizer.next()

        # Method of choice is just efron right now
        if self.tie_method == 'Efron':
            get_gradients = self._get_efron_values
        else:
            raise NotImplementedError("Only Efron is available.")

        i = 0
        converging = True
        ll, previous_ll = 0, 0
        start = time.time()

        while converging:
            self.path.append(beta.copy())
            i += 1
            if self.strata is None:
                h, g, ll = get_gradients(X.values, beta, T.values, E.values, weights.values)
            else:
                g = np.zeros_like(beta).T
                h = np.zeros((beta.shape[0], beta.shape[0]))
                ll = 0
                for strata in np.unique(X.index):
                    stratified_X, stratified_T, stratified_E, stratified_W = X.loc[[strata]], T.loc[[strata]], E.loc[[strata]], weights.loc[[strata]]
                    _h, _g, _ll = get_gradients(stratified_X.values, beta, stratified_T.values, stratified_E.values, stratified_W.values)
                    g += _g
                    h += _h
                    ll += _ll

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta.T
                h.flat[::d + 1] -= self.penalizer

            delta = solve(-h, step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ValueError("""delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
""")

            # Save these as pending result
            hessian, gradient = h, g
            norm_delta = norm(delta)

            if show_progress:
                print("Iteration %d: norm_delta = %.5f, step_size = %.5f, ll = %.5f, seconds_since_start = %.1f" % (i, norm_delta, step_size, ll, time.time() - start))
            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif abs(ll - previous_ll) < precision:
                converging, completed = False, True
            elif i >= max_steps:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is ~10 steps
                converging, completed = False, False
            elif step_size <= 0.00001:
                converging, completed = False, False
            elif abs(ll) < 0.0001 and norm_delta > 1.0:
                warnings.warn("The log-likelihood is getting suspciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
See https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/ ", ConvergenceWarning)
                converging, completed = False, False

            step_size = step_sizer.update(norm_delta).next()

            beta += delta
            previous_ll = ll

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        if not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    def _get_efron_values(self, X, beta, T, E, weights):
        """
        Calculates the first and second order vector differentials,
        with respect to beta.
        Note that X, T, E are assumed to be sorted on T!
        Parameters:
            X: (n,d) numpy array of observations.
            beta: (1, d) numpy array of coefficients.
            T: (n) numpy array representing observed durations.
            E: (n) numpy array representing death events.
            weights: (n) an array representing weights per observation.
        Returns:
            hessian: (d, d) numpy array,
            gradient: (1, d) numpy array
            log_likelihood: double
        """

        n, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros((1, d))
        log_lik = 0

        # Init risk and tie sums to zero
        x_tie_sum = np.zeros((1, d))
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))
        risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

        # Init number of ties
        tie_count = 0

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]
            xi = X[i:i + 1]
            w = weights[i]

            # Calculate phi values
            phi_i = w * exp(dot(xi, beta))
            phi_x_i = phi_i * xi
            phi_x_x_i = dot(xi.T, phi_x_i)

            # Calculate sums of Risk set
            risk_phi += phi_i
            risk_phi_x += phi_x_i
            risk_phi_x_x += phi_x_x_i

            # Calculate sums of Ties, if this is an event
            if ei:
                x_tie_sum += w * xi
                tie_phi += phi_i
                tie_phi_x += phi_x_i
                tie_phi_x_x += phi_x_x_i

                # Keep track of count
                tie_count += int(w)

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                # Only censored with current time, move on
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            partial_gradient = np.zeros((1, d))

            for l in range(tie_count):
                c = l / tie_count

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

                log_lik -= np.log(denom)[0][0]

            # Values outside tie sum
            gradient += x_tie_sum - partial_gradient
            log_lik += dot(x_tie_sum, beta)[0][0]

            # reset tie values
            tie_count = 0
            x_tie_sum = np.zeros((1, d))
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            tie_phi_x_x = np.zeros((d, d))
        return hessian, gradient, log_lik

    def _compute_baseline_cumulative_hazard(self):
        return self.baseline_hazard_.cumsum()

    @staticmethod
    def _check_values(df, T, E):
        pass_for_numeric_dtypes_or_raise(df)
        check_nans(T)
        check_nans(E)
        check_low_var(df)
        check_complete_separation(df, E, T)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1. + self.alpha) / 2.)
        se = self._compute_standard_errors()
        hazards = self.hazards_.values
        return pd.DataFrame(np.r_[hazards - alpha2 * se,
                                  hazards + alpha2 * se],
                            index=['lower-bound', 'upper-bound'],
                            columns=self.hazards_.columns)

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

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        df = self.summary
        # Significance codes last
        df[''] = [significance_code(p) for p in df['p']]

        # Print information about data first
        print('n={}, number of events={}'.format(self._n_examples,
                                                 self.event_observed.sum()),
              end='\n\n')
        print(df.to_string(float_format=lambda f: '{:4.4f}'.format(f)))
        # Significance code explanation
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ",
              end='\n\n')
        print("Concordance = {:.3f}".format(self.score_))
        return

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

    def predict_log_hazard_relative_to_mean(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the log hazard relative to the hazard of the mean covariates. This is the behaviour
        of R's predict.coxph. Equal to \beta X - \beta \bar{X_{train}}
        """

        return self.predict_log_partial_hazard(X) - self._train_log_partial_hazard.squeeze()

    def predict_cumulative_hazard(self, X, times=None):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        times: an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved). Uses a linear interpolation if
            points in time are not in the index.

        Returns the cumulative hazard of individuals.
        """

        if self.strata:
            cumulative_hazard_ = pd.DataFrame()
            for stratum, stratified_X in X.groupby(self.strata):
                try:
                    c_0 = self.baseline_cumulative_hazard_[[stratum]]
                except KeyError:
                    raise StatError("""The stratum %s was not found in the original training data. For example, try
the following on the original dataset, df: `df.groupby(%s).size()`. Expected is that %s is not present in the output.
""" % (stratum, self.strata, stratum))
                col = _get_index(stratified_X)
                v = self.predict_partial_hazard(stratified_X)
                cumulative_hazard_ = cumulative_hazard_.merge(pd.DataFrame(np.dot(c_0, v.T), index=c_0.index, columns=col), how='outer', right_index=True, left_index=True)
        else:
            c_0 = self.baseline_cumulative_hazard_
            col = _get_index(X)
            v = self.predict_partial_hazard(X)
            cumulative_hazard_ = pd.DataFrame(np.dot(c_0, v.T), columns=col, index=c_0.index)

        if times is not None:
            # non-linear interpolations can push the survival curves above 1 and below 0.
            return cumulative_hazard_.reindex(cumulative_hazard_.index.union(times)).interpolate("index").loc[times]
        else:
            return cumulative_hazard_

    def predict_survival_function(self, X, times=None):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        times: an iterable of increasing times to predict the survival function at. Default
            is the set of all durations (observed and unobserved)

        Returns the estimated survival functions for the individuals
        """
        return exp(-self.predict_cumulative_hazard(X, times=times))

    def predict_percentile(self, X, p=0.5):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        By default, returns the median lifetimes for the individuals.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions
        """
        subjects = _get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X)[subjects]).T

    def predict_median(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the median lifetimes for the individuals
        """
        return self.predict_percentile(X, 0.5)

    def predict_expectation(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Compute the expected lifetime, E[T], using covarites X.
        """
        subjects = _get_index(X)
        v = self.predict_survival_function(X)[subjects]
        return pd.DataFrame(trapz(v.values.T, v.index), index=subjects)

    def _compute_baseline_hazard(self, data, durations, event_observed, name):
        # http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
        ind_hazards = self.predict_partial_hazard(data)
        ind_hazards['event_at'] = durations.values
        ind_hazards_summed_over_durations = ind_hazards.groupby('event_at')[0].sum().sort_index(ascending=False).cumsum()
        ind_hazards_summed_over_durations.name = 'hazards'

        event_table = survival_table_from_events(durations, event_observed)
        event_table = event_table.join(ind_hazards_summed_over_durations)
        baseline_hazard = pd.DataFrame(event_table['observed'] / event_table['hazards'], columns=[name]).fillna(0)
        return baseline_hazard


    def _compute_baseline_hazards(self, df, T, E):
        if self.strata:
            index = self.durations.unique()
            baseline_hazards_ = pd.DataFrame(index=index)
            for stratum in df.index.unique():
                baseline_hazards_ = baseline_hazards_.merge(
                    self._compute_baseline_hazard(data=df.loc[[stratum]], durations=T.loc[[stratum]], event_observed=E.loc[[stratum]], name=stratum),
                    left_index=True,
                    right_index=True,
                    how='left')
            return baseline_hazards_.fillna(0)

        else:
            return self._compute_baseline_hazard(data=df, durations=T, event_observed=E, name='baseline hazard')

    def _compute_baseline_survival(self):
        survival_df = exp(-self.baseline_cumulative_hazard_)
        if self.strata is None:
            survival_df.columns = ['baseline survival']
        return survival_df

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

    def plot_covariate_groups(self, covariate, groups, **kwargs):
        """
        Produces a visual representation comparing the baseline survival curve of the model versus
        what happens when a covariate is varied over values in a group. This is useful to compare
        subjects' survival as we vary a single covariate, all else being held equal. The baseline survival
        curve is equal to the predicted survival curve at all average values in the original dataset.

        Parameters:
            covariate: a string of the covariate in the original dataset that we wish to vary.
            groups: an iterable of the values we wish the covariate to take on.

        Returns:
            ax: the matplotlib axis that be edited.
        """
        from matplotlib import pyplot as plt

        if covariate not in self.summary.index:
            raise KeyError('covariate `%s` is not present in the original dataset' % covariate)

        ax = kwargs.get('ax', None) or plt.figure().add_subplot(111)
        x_bar = self._norm_mean.to_frame().T
        X = pd.concat([x_bar] * len(groups))
        X.index = ['%s=%s' % (covariate, g) for g in groups]
        X[covariate] = groups

        self.predict_survival_function(X).plot(ax=ax)
        self.baseline_survival_.plot(ax=ax, ls='--')
        return ax
