# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd

from numpy import dot, exp
from numpy.linalg import solve, norm, inv
from scipy.integrate import trapz
import scipy.stats as stats

from lifelines.fitters import BaseFitter
from lifelines.utils import survival_table_from_events, inv_normal_cdf, normalize,\
    significance_code, concordance_index, _get_index, qth_survival_times


class CoxPHFitter(BaseFitter):

    """
    This class implements fitting Cox's proportional hazard model:

    h(t|x) = h_0(t)*exp(x'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      tie_method: specify how the fitter should deal with ties. Currently only
        'Efron' is available.
      normalize: substract the mean and divide by standard deviation of each covariate
        in the input data before performing any fitting.
      penalizer: Attach a L2 penalizer to the size of the coeffcients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the absolute value of beta_i. Recommended, even if a small value.
        The penalty is 1/2 * penalizer * ||beta||^2.
    """

    def __init__(self, alpha=0.95, tie_method='Efron', normalize=True, penalizer=0.0):
        if not (0 < alpha <= 1.):
            raise ValueError('alpha parameter must be between 0 and 1.')
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")
        if tie_method != 'Efron':
            raise NotImplementedError("Only Efron is available atm.")

        self.alpha = alpha
        self.normalize = normalize
        self.tie_method = tie_method
        self.penalizer = penalizer
        self.strata = None

    def _get_efron_values(self, X, beta, T, E, include_likelihood=False):
        """
        Calculates the first and second order vector differentials,
        with respect to beta. If 'include_likelihood' is True, then
        the log likelihood is also calculated. This is omitted by default
        to speed up the fit.

        Note that X, T, E are assumed to be sorted on T!

        Parameters:
            X: (n,d) numpy array of observations.
            beta: (1, d) numpy array of coefficients.
            T: (n) numpy array representing observed durations.
            E: (n) numpy array representing death events.

        Returns:
            hessian: (d, d) numpy array,
            gradient: (1, d) numpy array
            log_likelihood: double, if include_likelihood=True
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
        for i, (ti, ei) in reversed(list(enumerate(zip(T, E)))):
            # Doing it like this to preserve shape
            xi = X[i:i + 1]
            # Calculate phi values
            phi_i = exp(dot(xi, beta))
            phi_x_i = dot(phi_i, xi)
            phi_x_x_i = dot(xi.T, xi) * phi_i

            # Calculate sums of Risk set
            risk_phi += phi_i
            risk_phi_x += phi_x_i
            risk_phi_x_x += phi_x_x_i
            # Calculate sums of Ties, if this is an event
            if ei:
                x_tie_sum += xi
                tie_phi += phi_i
                tie_phi_x += phi_x_i
                tie_phi_x_x += phi_x_x_i

                # Keep track of count
                tie_count += 1

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

                if denom == 0:
                    # Can't divide by zero
                    raise ValueError("Denominator was zero")

                # Gradient
                partial_gradient += z / denom
                # Hessian
                a1 = (risk_phi_x_x - c * tie_phi_x_x) / denom
                # In case z and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = dot(z.T / denom, z / denom)

                hessian -= (a1 - a2)

                if include_likelihood:
                    log_lik -= np.log(denom).ravel()[0]

            # Values outside tie sum
            gradient += x_tie_sum - partial_gradient
            if include_likelihood:
                log_lik += dot(x_tie_sum, beta).ravel()[0]

            # reset tie values
            tie_count = 0
            x_tie_sum = np.zeros((1, d))
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            tie_phi_x_x = np.zeros((d, d))

        if include_likelihood:
            return hessian, gradient, log_lik
        else:
            return hessian, gradient

    def _newton_rhaphson(self, X, T, E, initial_beta=None, step_size=1.,
                         precision=10e-5, show_progress=True, include_likelihood=False):
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            X: (n,d) Pandas DataFrame of observations.
            T: (n) Pandas Series representing observed durations.
            E: (n) Pandas Series representing death events.
            initial_beta: (1,d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0.001 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.
            include_likelihood: saves the final log-likelihood to the CoxPHFitter under _log_likelihood.

        Returns:
            beta: (1,d) numpy array.
        """
        assert precision <= 1., "precision must be less than or equal to 1."
        n, d = X.shape

        # Want as bools
        E = E.astype(bool)

        # make sure betas are correct size.
        if initial_beta is not None:
            assert initial_beta.shape == (d, 1)
            beta = initial_beta
        else:
            beta = np.zeros((d, 1))

        # Method of choice is just efron right now
        if self.tie_method == 'Efron':
            get_gradients = self._get_efron_values
        else:
            raise NotImplementedError("Only Efron is available.")

        i = 1
        converging = True
        # 50 iterations steps with N-R is a lot.
        # Expected convergence is ~10 steps
        while converging and i < 50 and step_size > 0.001:

            if self.strata is None:
                output = get_gradients(X.values, beta, T.values, E.values, include_likelihood=include_likelihood)
                h, g = output[:2]
            else:
                g = np.zeros_like(beta).T
                h = np.zeros((beta.shape[0], beta.shape[0]))
                ll = 0
                for strata in np.unique(X.index):
                    stratified_X, stratified_T, stratified_E = X.loc[[strata]], T.loc[[strata]], E.loc[[strata]]
                    output = get_gradients(stratified_X.values, beta, stratified_T.values, stratified_E.values, include_likelihood=include_likelihood)
                    _h, _g = output[:2]
                    g += _g
                    h += _h
                    ll += output[2] if include_likelihood else 0

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta.T
                h.flat[::d + 1] -= self.penalizer

            delta = solve(-h, step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ValueError("delta contains nan value(s). Convergence halted.")

            # Save these as pending result
            hessian, gradient = h, g

            if norm(delta) < precision:
                converging = False

            # Only allow small steps
            if norm(delta) > 10:
                step_size *= 0.5
                continue

            beta += delta

            if ((i % 10) == 0) and show_progress:
                print("Iteration %d: delta = %.5f" % (i, norm(delta)))
            i += 1

        self._hessian_ = hessian
        self._score_ = gradient
        if include_likelihood:
            self._log_likelihood = output[-1] if self.strata is None else ll
        if show_progress:
            print("Convergence completed after %d iterations." % (i))
        return beta

    def fit(self, df, duration_col, event_col=None,
            show_progress=False, initial_beta=None, include_likelihood=False,
            strata=None):
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
          show_progress: since the fitter is iterative, show convergence
             diagnostics.
          initial_beta: initialize the starting point of the iterative
             algorithm. Default is the zero vector.
          include_likelihood: saves the final log-likelihood to the CoxPHFitter under
             the property _log_likelihood.
          strata: specify a list of columns to use in stratification. This is useful if a
             catagorical covariate does not obey the proportional hazard assumption. This
             is used similar to the `strata` expression in R.
             See http://courses.washington.edu/b515/l17.pdf.

        Returns:
            self, with additional properties: hazards_

        """
        df = df.copy()
        # Sort on time
        df.sort_values(by=duration_col, inplace=True)

        # remove strata coefs
        self.strata = strata
        if strata is not None:
            df = df.set_index(strata)

        # Extract time and event
        T = df[duration_col]
        del df[duration_col]
        if event_col is None:
            E = pd.Series(np.ones(df.shape[0]), index=df.index)
        else:
            E = df[event_col]
            del df[event_col]

        # Store original non-normalized data
        self.data = df if self.strata is None else df.reset_index()

        if self.normalize:
            # Need to normalize future inputs as well
            self._norm_mean = df.mean(0)
            self._norm_std = df.std(0)
            df = normalize(df)

        E = E.astype(bool)
        self._check_values(df)

        hazards_ = self._newton_rhaphson(df, T, E, initial_beta=initial_beta,
                                         show_progress=show_progress,
                                         include_likelihood=include_likelihood)

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns,
                                     index=['coef'])
        self.confidence_intervals_ = self._compute_confidence_intervals()

        self.durations = T
        self.event_observed = E

        self.baseline_hazard_ = self._compute_baseline_hazard()
        self.baseline_cumulative_hazard_ = self.baseline_hazard_.cumsum()
        self.baseline_survival_ = exp(-self.baseline_cumulative_hazard_)
        return self

    def _check_values(self, X):
        low_var = (X.var(0) < 10e-5)
        if low_var.any():
            cols = str(list(X.columns[low_var]))
            print("Warning: column(s) %s have very low variance.\
 This may harm convergence." % cols)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1. + self.alpha) / 2.)
        se = self._compute_standard_errors()
        hazards = self.hazards_.values
        return pd.DataFrame(np.r_[hazards - alpha2 * se,
                                  hazards + alpha2 * se],
                            index=['lower-bound', 'upper-bound'],
                            columns=self.hazards_.columns)

    def _compute_standard_errors(self):
        se = np.sqrt(inv(-self._hessian_).diagonal())
        return pd.DataFrame(se[None, :],
                            index=['se'], columns=self.hazards_.columns)

    def _compute_z_values(self):
        return (self.hazards_.ix['coef'] /
                self._compute_standard_errors().ix['se'])

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
        df['coef'] = self.hazards_.ix['coef'].values
        df['exp(coef)'] = exp(self.hazards_.ix['coef'].values)
        df['se(coef)'] = self._compute_standard_errors().ix['se'].values
        df['z'] = self._compute_z_values()
        df['p'] = self._compute_p_values()
        df['lower %.2f' % self.alpha] = self.confidence_intervals_.ix['lower-bound'].values
        df['upper %.2f' % self.alpha] = self.confidence_intervals_.ix['upper-bound'].values
        return df

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        df = self.summary
        # Significance codes last
        df[''] = [significance_code(p) for p in df['p']]

        # Print information about data first
        print('n={}, number of events={}'.format(self.data.shape[0],
                                                 np.where(self.event_observed)[0].shape[0]),
              end='\n\n')
        print(df.to_string(float_format=lambda f: '{:.3e}'.format(f)))
        # Significance code explanation
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ",
              end='\n\n')
        print("Concordance = {:.3f}"
              .format(concordance_index(self.durations,
                                        -self.predict_partial_hazard(self.data).values.ravel(),
                                        self.event_observed)))
        return

    def predict_partial_hazard(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        If covariates were normalized during fitting, they are normalized
        in the same way here.

        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to \exp{\beta X}
        """
        index = _get_index(X)

        if isinstance(X, pd.DataFrame):
            order = self.hazards_.columns
            X = X[order]

        if self.normalize:
            # Assuming correct ordering and number of columns
            X = normalize(X, self._norm_mean.values, self._norm_std.values)

        return pd.DataFrame(exp(np.dot(X, self.hazards_.T)), index=index)

    def predict_log_hazard_relative_to_mean(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the log hazard relative to the hazard of the mean covariates. This is the behaviour
        of R's predict.coxph.
        """
        mean_covariates = self.data.mean(0).to_frame().T
        return np.log(self.predict_partial_hazard(X) / self.predict_partial_hazard(mean_covariates).squeeze())

    def predict_cumulative_hazard(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the cumulative hazard for the individuals.
        """
        v = self.predict_partial_hazard(X)
        s_0 = self.baseline_survival_
        col = _get_index(X)
        return pd.DataFrame(-np.dot(np.log(s_0), v.T), index=self.baseline_survival_.index, columns=col)

    def predict_survival_function(self, X):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the estimated survival functions for the individuals
        """
        return exp(-self.predict_cumulative_hazard(X))

    def predict_percentile(self, X, p=0.5):
        """
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        By default, returns the median lifetimes for the individuals.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions
        """
        index = _get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X)[index])

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
        index = _get_index(X)
        v = self.predict_survival_function(X)[index]
        return pd.DataFrame(trapz(v.values.T, v.index), index=index)

    def _compute_baseline_hazard(self):
        # http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
        ind_hazards = self.predict_partial_hazard(self.data).values

        event_table = survival_table_from_events(self.durations.values,
                                                 self.event_observed.values)

        baseline_hazard_ = pd.DataFrame(np.zeros((event_table.shape[0], 1)),
                                        index=event_table.index,
                                        columns=['baseline hazard'])

        for t, s in event_table.iterrows():
            less = np.array(self.durations >= t)
            if ind_hazards[less].sum() == 0:
                v = 0
            else:
                v = (s['observed'] / ind_hazards[less].sum())
            baseline_hazard_.ix[t] = v

        return baseline_hazard_
