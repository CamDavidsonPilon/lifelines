# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import time
from datetime import datetime
import warnings
import numpy as np
import pandas as pd

from numpy import dot, exp
from numpy.linalg import norm, inv
from scipy.linalg import solve as spsolve
from scipy.integrate import trapz
from scipy import stats

from lifelines.fitters import BaseFitter
from lifelines.statistics import chisq_test
from lifelines.utils import (
    survival_table_from_events,
    inv_normal_cdf,
    normalize,
    significance_code,
    significance_codes_as_text,
    concordance_index,
    _get_index,
    qth_survival_times,
    pass_for_numeric_dtypes_or_raise,
    check_low_var,
    coalesce,
    check_complete_separation,
    check_nans_or_infs,
    StatError,
    ConvergenceWarning,
    StepSizer,
    ConvergenceError,
    string_justify,
    _to_list,
)


class CoxPHFitter(BaseFitter):

    r"""
    This class implements fitting Cox's proportional hazard model:

    .. math::  h(t|x) = h_0(t) \exp(x \beta)

    Parameters
    ----------

      alpha: float, optional (default=0.95)
        the level in the confidence intervals.

      tie_method: string, optional
        specify how the fitter should deal with ties. Currently only
        'Efron' is available.

      penalizer: float, optional (default=0.0)
        Attach a L2 penalizer to the size of the coeffcients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the absolute value of :math:`\beta_i`. Recommended, even if a small value.
        The penalty is :math:`1/2  \text{penalizer}  ||beta||^2`.

      strata: list, optional
        specify a list of columns to use in stratification. This is useful if a
         catagorical covariate does not obey the proportional hazard assumption. This
         is used similar to the `strata` expression in R.
         See http://courses.washington.edu/b515/l17.pdf.

    Examples
    --------
    >>> from lifelines.datasets import load_rossi
    >>> from lifelines import CoxPHFitter
    >>> rossi = load_rossi()
    >>> cph = CoxPHFitter()
    >>> cph.fit(rossi, 'week', 'arrest')
    >>> cph.print_summary()
    """

    def __init__(self, alpha=0.95, tie_method="Efron", penalizer=0.0, strata=None):
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha parameter must be between 0 and 1.")
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")
        if tie_method != "Efron":
            raise NotImplementedError("Only Efron is available atm.")

        self.alpha = alpha
        self.tie_method = tie_method
        self.penalizer = penalizer
        self.strata = strata

    def fit(
        self,
        df,
        duration_col,
        event_col=None,
        show_progress=False,
        initial_beta=None,
        strata=None,
        step_size=None,
        weights_col=None,
        cluster_col=None,
        robust=False,
    ):
        """
        Fit the Cox Propertional Hazard model to a dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas dataframe with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in dataframe that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of thecolumn in dataframe that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        weights_col: string, optional
            an optional column in the dataframe, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use `robust=True` to get more accurate standard errors.

        show_progress: boolean, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        initial_beta: numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            catagorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.

        step_size: float, optional
            set an initial step size for the fitting algorithm.

        robust: boolean, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078

        cluster_col: string, optional
            specifies what column has unique identifers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.

        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: print_summary, hazards_, confidence_intervals_, baseline_survival_, etc.


        Note
        ----
        Tied survival times are handled using Efron's tie-method.


        Examples
        --------
        >>> from lifelines import CoxPHFitter
        >>>
        >>> df = pd.DataFrame({
        >>>     'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        >>>     'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        >>>     'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>> })
        >>>
        >>> cph = CoxPHFitter()
        >>> cph.fit(df, 'T', 'E')
        >>> cph.print_summary()
        >>> cph.predict_median(df)


        >>> from lifelines import CoxPHFitter
        >>>
        >>> df = pd.DataFrame({
        >>>     'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        >>>     'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        >>>     'weights': [1.1, 0.5, 2.0, 1.6, 1.2, 4.3, 1.4, 4.5, 3.0, 3.2, 0.4, 6.2],
        >>>     'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>> })
        >>>
        >>> cph = CoxPHFitter()
        >>> cph.fit(df, 'T', 'E', strata=['month', 'age'], robust=True, weights_col='weights')
        >>> cph.print_summary()
        >>> cph.predict_median(df)

        """
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self.robust = robust
        self.cluster_col = cluster_col
        self.weights_col = weights_col
        self._n_examples = df.shape[0]
        self.strata = coalesce(strata, self.strata)

        X, T, E, weights, original_index, self._clusters = self._preprocess_dataframe(df)

        self.durations = T.copy()
        self.event_observed = E.copy()
        self.weights = weights.copy()

        if self.strata is not None:
            self.durations.index = original_index
            self.event_observed.index = original_index
            self.weights.index = original_index

        self._norm_mean = X.mean(0)
        self._norm_std = X.std(0)

        hazards_ = self._newton_rhaphson(
            normalize(X, self._norm_mean, self._norm_std),
            T,
            E,
            weights=weights,
            initial_beta=initial_beta,
            show_progress=show_progress,
            step_size=step_size,
        )

        self.hazards_ = pd.DataFrame(hazards_.T, columns=X.columns, index=["coef"]) / self._norm_std

        self.variance_matrix_ = -inv(self._hessian_) / np.outer(self._norm_std, self._norm_std)
        self.standard_errors_ = self._compute_standard_errors(
            normalize(X, self._norm_mean, self._norm_std), T, E, weights
        )
        self.confidence_intervals_ = self._compute_confidence_intervals()

        self.baseline_hazard_ = self._compute_baseline_hazards(X, T, E, weights)
        self.baseline_cumulative_hazard_ = self._compute_baseline_cumulative_hazard()
        self.baseline_survival_ = self._compute_baseline_survival()
        self._predicted_partial_hazards_ = self.predict_partial_hazard(X).values

        return self

    def _preprocess_dataframe(self, df):
        # this should be a pure function

        df = df.copy()

        if self.strata is not None:
            df = df.sort_values(by=[self.duration_col] + _to_list(self.strata))
            original_index = df.index.copy()
            df = df.set_index(self.strata)
        else:
            df = df.sort_values(by=self.duration_col)
            original_index = df.index.copy()

        # Extract time and event
        T = df.pop(self.duration_col)
        E = (
            df.pop(self.event_col)
            if (self.event_col is not None)
            else pd.Series(np.ones(self._n_examples), index=df.index, name="weights")
        )
        W = (
            df.pop(self.weights_col)
            if (self.weights_col is not None)
            else pd.Series(np.ones((self._n_examples,)), index=df.index, name="E")
        )

        # check to make sure their weights are okay
        if self.weights_col:
            if (W.astype(int) != W).any() and not self.robust:
                warnings.warn(
                    """It appears your weights are not integers, possibly propensity or sampling scores then?
It's important to know that the naive variance estimates of the coefficients are biased. Instead a) set `robust=True` in the call to `fit`, or b) use Monte Carlo to
estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
""",
                    RuntimeWarning,
                )
            if (W <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

        _clusters = df.pop(self.cluster_col).values if self.cluster_col else None

        self._check_values(df, T, E)

        X = df.astype(float)
        T = T.astype(float)
        E = E.astype(bool)

        return X, T, E, W, original_index, _clusters

    @staticmethod
    def _check_values(X, T, E):
        pass_for_numeric_dtypes_or_raise(X)
        check_nans_or_infs(T)
        check_nans_or_infs(E)
        check_nans_or_infs(X)
        check_low_var(X)
        check_complete_separation(X, E, T)

    def _newton_rhaphson(
        self,
        X,
        T,
        E,
        weights=None,
        initial_beta=None,
        step_size=None,
        precision=10e-6,
        show_progress=True,
        max_steps=50,
    ):  # pylint: disable=too-many-statements,too-many-branches
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note
        ----
        The data is assumed to be sorted on T!

        Parameters
        ----------
        X: (n,d) Pandas DataFrame of observations.
        T: (n) Pandas Series representing observed durations.
        E: (n) Pandas Series representing death events.
        weights: (n) an iterable representing weights per observation.
        initial_beta: (1,d) numpy array of initial starting point for
                      NR algorithm. Default 0.
        step_size: float, optional
            > 0.001 to determine a starting step size in NR algorithm.
        precision: float, optional
            the convergence halts if the norm of delta between
            successive positions is less than epsilon.
        show_progress: boolean, optional
            since the fitter is iterative, show convergence
                 diagnostics.
        max_steps: int, optional
            the maximum number of interations of the Newton-Rhaphson algorithm.

        Returns
        -------
        beta: (1,d) numpy array.
        """
        self.path = []
        assert precision <= 1.0, "precision must be less than or equal to 1."
        _, d = X.shape

        # make sure betas are correct size.
        if initial_beta is not None:
            assert initial_beta.shape == (d, 1)
            beta = initial_beta
        else:
            beta = np.zeros((d, 1))

        step_sizer = StepSizer(step_size)
        step_size = step_sizer.next()

        # Method of choice is just efron right now
        if self.tie_method == "Efron":
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
                h, g, ll = get_gradients(X.values, T.values, E.values, weights.values, beta)
            else:
                g = np.zeros_like(beta).T
                h = np.zeros((beta.shape[0], beta.shape[0]))
                ll = 0
                for _h, _g, _ll in self._partition_by_strata_and_apply(X, T, E, weights, get_gradients, beta):
                    g += _g
                    h += _h
                    ll += _ll

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta.T
                h.flat[:: d + 1] -= self.penalizer

            # reusing a piece to make g * inv(h) * g.T faster later
            try:
                inv_h_dot_g_T = spsolve(-h, g.T, sym_pos=True)
            except ValueError as e:
                if "infs or NaNs" in str(e):
                    raise ConvergenceError(
                        """hessian or gradient contains nan or inf value(s). Convergence halted. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
"""
                    )
                else:
                    # something else?
                    raise e

            delta = step_size * inv_h_dot_g_T

            if np.any(np.isnan(delta)):
                raise ConvergenceError(
                    """delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
"""
                )

            # Save these as pending result
            hessian, gradient = h, g
            norm_delta = norm(delta)

            # reusing an above piece to make g * inv(h) * g.T faster.
            newton_decrement = g.dot(inv_h_dot_g_T) / 2

            if show_progress:
                print(
                    "Iteration %d: norm_delta = %.5f, step_size = %.5f, ll = %.5f, newton_decrement = %.5f, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll, newton_decrement, time.time() - start)
                )

            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif previous_ll != 0 and abs(ll - previous_ll) / (-previous_ll) < 1e-09:
                # this is what R uses by default
                converging, completed = False, True
            elif newton_decrement < precision:
                converging, completed = False, True
            elif i >= max_steps:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is ~10 steps
                converging, completed = False, False
            elif step_size <= 0.00001:
                converging, completed = False, False
            elif abs(ll) < 0.0001 and norm_delta > 1.0:
                warnings.warn(
                    "The log-likelihood is getting suspciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
See https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/ ",
                    ConvergenceWarning,
                )
                converging, completed = False, False

            step_size = step_sizer.update(norm_delta).next()

            beta += delta
            previous_ll = ll

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        elif show_progress and not completed:
            print("Convergence failed. See warning messages.")
        if not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    def _get_efron_values(self, X, T, E, weights, beta):
        """
        Calculates the first and second order vector differentials, with respect to beta.
        Note that X, T, E are assumed to be sorted on T!

        A good explaination for Efron. Consider three of five subjects who fail at the time.
        As it is not known a priori that who is the first to fail, so one-third of
        (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
        of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

        From https://cran.r-project.org/web/packages/survival/survival.pdf:

        "Setting all weights to 2 for instance will give the same coefficient estimate but halve the variance. When
        the Efron approximation for ties (default) is employed replication of the data will not give exactly the same coefficients as the
        weights option, and in this case the weighted fit is arguably the correct one."

        Parameters
        ----------
        X: array
            (n,d) numpy array of observations.
        T: array
            (n) numpy array representing observed durations.
        E: array
            (n) numpy array representing death events.
        weights: array
            (n) an array representing weights per observation.
        beta: array
            (1, d) numpy array of coefficients.

        Returns
        -------
        hessian:
            (d, d) numpy array,
        gradient:
            (1, d) numpy array
        log_likelihood: float
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

        # Init number of ties and weights
        weight_count = 0.0
        tie_count = 0
        scores = weights[:, None] * exp(dot(X, beta))

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]
            xi = X[i : i + 1]
            score = scores[i : i + 1]
            w = weights[i]

            # Calculate phi values
            phi_i = score
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
                tie_count += 1
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                # Only censored with current time, move on
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            partial_gradient = np.zeros((1, d))
            weighted_average = weight_count / tie_count

            for l in range(tie_count):

                # A good explaination for Efron. Consider three of five subjects who fail at the time.
                # As it is not known a priori that who is the first to fail, so one-third of
                # (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
                # of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

                numer = risk_phi_x - l * tie_phi_x / tie_count
                denom = risk_phi - l * tie_phi / tie_count

                # Gradient
                partial_gradient += weighted_average * numer / denom
                # Hessian
                a1 = (risk_phi_x_x - l * tie_phi_x_x / tie_count) / denom

                # In case numer and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = dot(numer.T / denom, numer / denom)

                hessian -= weighted_average * (a1 - a2)

                log_lik -= weighted_average * np.log(denom[0][0])

            # Values outside tie sum
            gradient += x_tie_sum - partial_gradient
            log_lik += dot(x_tie_sum, beta)[0][0]

            # reset tie values
            tie_count = 0
            weight_count = 0.0
            x_tie_sum = np.zeros((1, d))
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            tie_phi_x_x = np.zeros((d, d))
        return hessian, gradient, log_lik

    def _partition_by_strata(self, X, T, E, weights, as_dataframes=False):
        for stratum, stratified_X in X.groupby(self.strata):
            stratified_E, stratified_T, stratified_W = (E.loc[[stratum]], T.loc[[stratum]], weights.loc[[stratum]])
            if not as_dataframes:
                yield (stratified_X.values, stratified_T.values, stratified_E.values, stratified_W.values), stratum
            else:
                yield (stratified_X, stratified_T, stratified_E, stratified_W), stratum

    def _partition_by_strata_and_apply(self, X, T, E, weights, function, *args):
        for (stratified_X, stratified_T, stratified_E, stratified_W), _ in self._partition_by_strata(X, T, E, weights):
            yield function(stratified_X, stratified_T, stratified_E, stratified_W, *args)

    def _compute_martingale(self, X, T, E, _weights, index=None):
        # TODO: decide if I want to attach T and E to the final dataframe...
        # TODO: _weights unused
        partial_hazard = self.predict_partial_hazard(X)[0].values

        if not self.strata:
            baseline_at_T = self.baseline_cumulative_hazard_.loc[T, "baseline hazard"].values
        else:
            baseline_at_T = np.empty(0)
            for name, T_ in T.groupby(level=0):
                baseline_at_T = np.append(baseline_at_T, self.baseline_cumulative_hazard_.loc[T_, name])

        martingale = E - (partial_hazard * baseline_at_T)
        martingale.index = index  # overrides the strata index, if necessary
        return pd.DataFrame({self.duration_col: T, self.event_col: E, "martingale": martingale})

    def _compute_deviance(self, X, T, E, weights, index=None):
        rmart = self._compute_martingale(X, T, E, weights, index)["martingale"]
        log_term = np.where((E.values - rmart.values) <= 0, 0, E.values * np.log(E.values - rmart.values))
        deviance = np.sign(rmart) * np.sqrt(-2 * (rmart + log_term))
        return pd.DataFrame({self.duration_col: T, self.event_col: E, "deviance": deviance})

    def _compute_scaled_schoenfeld(self, X, T, E, weights, index=None):
        r"""
        Let s_k be the kth schoenfeld residuals. Then E[s_k] = 0. 
        For tests of proportionality, we want to test if \beta_i(t) is \beta_i (constant) or not.

        Let V_k be the contribution to the information matrix at time t_k. A main result from Grambsch and Therneau is that 

        \beta(t) = E[s_k*V_k^{-1} + \hat{beta}]

        so define s_k^* = s_k*V_k^{-1} + \hat{beta} as the scaled schoenfeld residuals. 

        We can approximate V_k with Hessian/d, so the inverse of Hessian/d is (d * variance_matrix_)


        """

        n_deaths = sum(self.event_observed)
        scaled_schoenfeld_resids = n_deaths * self._compute_schoenfeld(X, T, E, weights, index).dot(
            self.variance_matrix_
        )
        scaled_schoenfeld_resids.columns = self.hazards_.columns
        return scaled_schoenfeld_resids

    def _compute_schoenfeld(self, X, T, E, weights, index=None):
        # Assumes sorted on T and on strata
        # index will be set later
        # cluster does nothing to this, as expected.

        _, d = X.shape

        if self.strata is not None:
            schoenfeld_residuals = np.empty((0, d))

            for schoenfeld_residuals_in_strata in self._partition_by_strata_and_apply(
                X, T, E, weights, self._compute_schoenfeld_within_strata
            ):
                schoenfeld_residuals = np.append(schoenfeld_residuals, schoenfeld_residuals_in_strata, axis=0)

        else:
            schoenfeld_residuals = self._compute_schoenfeld_within_strata(X.values, T, E.values, weights.values)

        # schoenfeld residuals are only defined for subjects with a non-zero event.
        return pd.DataFrame(schoenfeld_residuals[E, :], columns=self.hazards_.columns, index=index[E])

    def _compute_schoenfeld_within_strata(self, X, T, E, weights):
        # TODO: the diff_against is gross
        # This uses Efron ties.

        n, d = X.shape

        # Init risk and tie sums to zero
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))

        # Init number of ties and weights
        weight_count = 0.0
        tie_count = 0
        scores = weights[:, None] * exp(dot(X, self.hazards_.T))

        diff_against = []

        schoenfeld_residuals = np.empty((0, d))

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]
            xi = X[i : i + 1]
            score = scores[i : i + 1]
            w = weights[i]

            # Calculate phi values
            phi_i = score
            phi_x_i = phi_i * xi

            # Calculate sums of Risk set
            risk_phi += phi_i
            risk_phi_x += phi_x_i

            # Calculate sums of Ties, if this is an event
            diff_against.append((xi, ei))
            if ei:

                tie_phi += phi_i
                tie_phi_x += phi_x_i

                # Keep track of count
                tie_count += 1  # aka death counts
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                # Only censored with current time, move on
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            weighted_mean = np.zeros((1, d))

            for l in range(tie_count):

                numer = risk_phi_x - l * tie_phi_x / tie_count
                denom = risk_phi - l * tie_phi / tie_count

                weighted_mean += numer / (denom * tie_count)

            for xi, ei in diff_against:
                schoenfeld_residuals = np.append(schoenfeld_residuals, ei * (xi - weighted_mean), axis=0)

            # reset tie values
            tie_count = 0
            weight_count = 0.0
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            diff_against = []

        return schoenfeld_residuals[::-1]

    def _compute_delta_beta(self, X, T, E, weights, index=None):
        """
        approximate change in betas as a result of excluding ith row. Good for finding outliers
        """
        score_residuals = self._compute_score(X, T, E, weights, index=index)
        delta_betas = score_residuals.dot(self.variance_matrix_) * self._norm_std.values
        delta_betas.columns = self.hazards_.columns
        return delta_betas

    def _compute_score(self, X, T, E, weights, index=None):

        _, d = X.shape

        if self.strata is not None:
            score_residuals = np.empty((0, d))

            for score_residuals_in_strata in self._partition_by_strata_and_apply(
                X, T, E, weights, self._compute_score_within_strata
            ):
                score_residuals = np.append(score_residuals, score_residuals_in_strata, axis=0)

        else:
            score_residuals = self._compute_score_within_strata(X.values, T, E.values, weights.values)

        return pd.DataFrame(score_residuals, columns=self.hazards_.columns, index=index)

    def _compute_score_within_strata(self, X, _T, E, weights):
        # https://www.stat.tamu.edu/~carroll/ftp/gk001.pdf
        # lin1989
        # https://www.ics.uci.edu/~dgillen/STAT255/Handouts/lecture10.pdf
        # Assumes X already sorted by T
        # TODO: doesn't handle ties.
        # TODO: _T unused

        n, d = X.shape

        # we already unnormalized the betas in `fit`, so we need normalize them again since X is
        # normalized.
        beta = self.hazards_.values[0] * self._norm_std

        E = E.astype(int)
        score_residuals = np.zeros((n, d))

        phi_s = exp(dot(X, beta))

        # need to store these histories, as we access them often
        # this is a reverse cumulative sum. See original code in https://github.com/CamDavidsonPilon/lifelines/pull/496/files#diff-81ee0759dbae0770e1a02cf17f4cfbb1R431
        risk_phi_x_history = (X * (weights * phi_s)[:, None])[::-1].cumsum(0)[::-1]
        risk_phi_history = (weights * phi_s)[::-1].cumsum()[::-1][:, None]

        # Iterate forwards
        for i in range(0, n):

            xi = X[i : i + 1]
            phi_i = phi_s[i]

            score = -phi_i * (
                (
                    E[: i + 1] * weights[: i + 1] / risk_phi_history[: i + 1].T
                ).T  # this is constant-ish, and could be cached
                * (xi - risk_phi_x_history[: i + 1] / risk_phi_history[: i + 1])
            ).sum(0)

            if E[i]:
                score = score + (xi - risk_phi_x_history[i] / risk_phi_history[i])

            score_residuals[i, :] = score

        return score_residuals * weights[:, None]

    def compute_residuals(self, df, kind):
        """

        Parameters
        ----------
        df : the same training data given in `fit`
        kind : string
            {'schoenfeld', 'score', 'delta_beta', 'deviance', 'martingale'}
        TODO: can I check the same training data is inputted? checksum?

        """
        ALLOWED_RESIDUALS = {"schoenfeld", "score", "delta_beta", "deviance", "martingale", "scaled_schoenfeld"}
        assert kind in ALLOWED_RESIDUALS, "kind must be in %s" % ALLOWED_RESIDUALS

        X, T, E, weights, shuffled_original_index, _ = self._preprocess_dataframe(df)

        resids = getattr(self, "_compute_%s" % kind)(X, T, E, weights, index=shuffled_original_index)
        return resids

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        se = self.standard_errors_
        hazards = self.hazards_.values
        return pd.DataFrame(
            np.r_[hazards - alpha2 * se, hazards + alpha2 * se],
            index=["lower-bound", "upper-bound"],
            columns=self.hazards_.columns,
        )

    def _compute_standard_errors(self, X, T, E, weights):
        if self.robust or self.cluster_col:
            se = np.sqrt(self._compute_sandwich_estimator(X, T, E, weights).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.DataFrame(se[None, :], index=["se"], columns=self.hazards_.columns)

    def _compute_sandwich_estimator(self, X, T, E, weights):
        delta_betas = self._compute_delta_beta(X, T, E, weights)

        if self.cluster_col:
            delta_betas = delta_betas.groupby(self._clusters).sum()

        sandwich_estimator = delta_betas.T.dot(delta_betas)

        return sandwich_estimator.values

    def _compute_z_values(self):
        return self.hazards_.loc["coef"] / self.standard_errors_.loc["se"]

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper"""

        df = pd.DataFrame(index=self.hazards_.columns)
        df["coef"] = self.hazards_.loc["coef"].values
        df["exp(coef)"] = exp(self.hazards_.loc["coef"].values)
        df["se(coef)"] = self.standard_errors_.loc["se"].values
        df["z"] = self._compute_z_values()
        df["p"] = self._compute_p_values()
        df["lower %.2f" % self.alpha] = self.confidence_intervals_.loc["lower-bound"].values
        df["upper %.2f" % self.alpha] = self.confidence_intervals_.loc["upper-bound"].values
        return df

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        # pylint: disable=unnecessary-lambda

        # Print information about data first
        justify = string_justify(18)
        print(self)
        print("{} = {}".format(justify("duration col"), self.duration_col))
        print("{} = {}".format(justify("event col"), self.event_col))
        if self.weights_col:
            print("{} = {}".format(justify("weights col"), self.weights_col))

        if self.cluster_col:
            print("{} = {}".format(justify("cluster col"), self.cluster_col))

        if self.robust or self.cluster_col:
            print("{} = {}".format(justify("robust variance"), True))

        if self.strata:
            print("{} = {}".format(justify("strata"), self.strata))

        print("{} = {}".format(justify("number of subjects"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.3f}".format(justify("log-likelihood"), self._log_likelihood))
        print("{} = {}".format(justify("time fit was run"), self._time_fit_was_called), end="\n\n")
        print("---")

        df = self.summary
        # Significance codes last
        df[""] = [significance_code(p) for p in df["p"]]
        print(df.to_string(float_format=lambda f: "{:4.4f}".format(f)))
        # Significance code explanation
        print("---")
        print(significance_codes_as_text(), end="\n\n")
        print("Concordance = {:.3f}".format(self.score_))
        print("Likelihood ratio test = {:.3f} on {} df, p={:.5f}".format(*self._compute_likelihood_ratio_test()))

    def _compute_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Cox model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.

        Conveniently, we can actually use the class itself to do most of the work.

        """
        trivial_dataset = pd.DataFrame({"E": self.event_observed, "T": self.durations, "W": self.weights})

        cp_null = CoxPHFitter()
        cp_null.fit(trivial_dataset, "T", "E", weights_col="W", show_progress=False)

        ll_null = cp_null._log_likelihood
        ll_alt = self._log_likelihood

        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.hazards_.shape[1]
        _, p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom, alpha=0.0)
        return test_stat, degrees_freedom, p_value

    def predict_partial_hazard(self, X):
        r"""
        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        partial_hazard: DataFrame
            Returns the partial hazard for the individuals, partial since the
            baseline hazard is not included. Equal to :math:`\exp{\beta (X - mean(X_{train}))}`

        Notes
        -----
        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        return exp(self.predict_log_partial_hazard(X))

    def predict_log_partial_hazard(self, X):
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`\beta (X - mean(X_{train}))`


        Parameters
        ----------
        X:  numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        log_partial_hazard: DataFrame


        Notes
        -----
        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """

        hazard_names = self.hazards_.columns
        if isinstance(X, pd.DataFrame):
            order = hazard_names
            X = X[order]
            pass_for_numeric_dtypes_or_raise(X)
        elif isinstance(X, pd.Series) and ((X.shape[0] == len(hazard_names) + 2) or (X.shape[0] == len(hazard_names))):
            X = X.to_frame().T
            order = hazard_names
            X = X[order]
            pass_for_numeric_dtypes_or_raise(X)
        elif isinstance(X, pd.Series):
            assert len(hazard_names) == 1, "Series not the correct arugment"
            X = pd.DataFrame(X)
            pass_for_numeric_dtypes_or_raise(X)

        X = X.astype(float)
        index = _get_index(X)

        X = normalize(X, self._norm_mean.values, 1)
        return pd.DataFrame(np.dot(X, self.hazards_.T), index=index)

    def predict_cumulative_hazard(self, X, times=None):
        """
        Parameters
        ----------

        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved). Uses a linear interpolation if
            points in time are not in the index.

        Returns
        -------
        cumulative_hazard_ : DataFrame
            the cumulative hazard of individuals over the timeline
        """

        if self.strata:
            cumulative_hazard_ = pd.DataFrame()
            for stratum, stratified_X in X.groupby(self.strata):
                try:
                    c_0 = self.baseline_cumulative_hazard_[[stratum]]
                except KeyError:
                    raise StatError(
                        """The stratum %s was not found in the original training data. For example, try
the following on the original dataset, df: `df.groupby(%s).size()`. Expected is that %s is not present in the output.
"""
                        % (stratum, self.strata, stratum)
                    )
                col = _get_index(stratified_X)
                v = self.predict_partial_hazard(stratified_X)
                cumulative_hazard_ = cumulative_hazard_.merge(
                    pd.DataFrame(np.dot(c_0, v.T), index=c_0.index, columns=col),
                    how="outer",
                    right_index=True,
                    left_index=True,
                )
        else:
            c_0 = self.baseline_cumulative_hazard_
            v = self.predict_partial_hazard(X)
            col = _get_index(v)
            cumulative_hazard_ = pd.DataFrame(np.dot(c_0, v.T), columns=col, index=c_0.index)

        if times is not None:
            # non-linear interpolations can push the survival curves above 1 and below 0.
            return cumulative_hazard_.reindex(cumulative_hazard_.index.union(times)).interpolate("index").loc[times]
        return cumulative_hazard_

    def predict_survival_function(self, X, times=None):
        """
        Predict the survival function for individuals, given their covariates. This assumes that the individual
        just entered the study (that is, we do not condition on how long they have already lived for.)

        Parameters
        ----------

        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved). Uses a linear interpolation if
            points in time are not in the index.


        Returns
        -------
        survival_function : DataFrame
            the survival probabilities of individuals over the timeline
        """
        return exp(-self.predict_cumulative_hazard(X, times=times))

    def predict_percentile(self, X, p=0.5):
        """
        Returns the median lifetimes for the individuals, by default. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions

        Parameters
        ----------
        X:  numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        p: float, optional (default=0.5)
            the percentile, must be between 0 and 1.

        Returns
        -------
        percentiles: DataFrame

        See Also
        --------
        predict_median

        """
        subjects = _get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X)[subjects]).T

    def predict_median(self, X):
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        percentiles: DataFrame
            the median lifetimes for the individuals. If the survival curve of an
            individual does not cross 0.5, then the result is infinity.


        See Also
        --------
        predict_percentile

        """
        return self.predict_percentile(X, 0.5)

    def predict_expectation(self, X):
        r"""
        Compute the expected lifetime, :math:`E[T]`, using covarites X. This algorithm to compute the expection is
        to use the fact that :math:`E[T] = \int_0^\inf P(T > t) dt = \int_0^\inf S(t) dt`. To compute the integal, we use the trapizoidal rule to approximate the integral.

        Caution
        --------
        However, if the survival function doesn't converge to 0, the the expectation is really infinity and the returned
        values are meaningless/too large. In that case, using ``predict_median`` or ``predict_percentile`` would be better.

        Parameters
        ----------

        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        expectations : DataFrame

        Notes
        -----
        If X is a dataframe, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        See Also
        --------
        predict_median
        predict_percentile

        """
        subjects = _get_index(X)
        v = self.predict_survival_function(X)[subjects]
        return pd.DataFrame(trapz(v.values.T, v.index), index=subjects)

    def _compute_baseline_hazard(self, X, durations, event_observed, weights, name):
        # https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
        ind_hazards = self.predict_partial_hazard(X) * weights[:, None]
        ind_hazards["event_at"] = durations.values
        ind_hazards_summed_over_durations = (
            ind_hazards.groupby("event_at")[0].sum().sort_index(ascending=False).cumsum()
        )
        ind_hazards_summed_over_durations.name = "hazards"

        event_table = survival_table_from_events(durations, event_observed, weights=weights)
        event_table = event_table.join(ind_hazards_summed_over_durations)
        baseline_hazard = pd.DataFrame(event_table["observed"] / event_table["hazards"], columns=[name]).fillna(0)

        return baseline_hazard

    def _compute_baseline_hazards(self, X, T, E, weights):
        if self.strata:
            index = self.durations.unique()
            baseline_hazards_ = pd.DataFrame(index=index)

            for (X_, T_, E_, W_), name in self._partition_by_strata(X, T, E, weights, as_dataframes=True):
                baseline_hazards_ = baseline_hazards_.merge(
                    self._compute_baseline_hazard(X_, T_, E_, W_, name), left_index=True, right_index=True, how="left"
                )
            return baseline_hazards_.fillna(0)

        return self._compute_baseline_hazard(X, T, E, weights, name="baseline hazard")

    def _compute_baseline_cumulative_hazard(self):
        return self.baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        """
        Importantly, this agrees with what the KaplanMeierFitter produces. Ex:
        from lifelines.datasets import load_rossi
        from lifelines import CoxPHFitter, KaplanMeierFitter
        rossi = load_rossi()

        kmf = KaplanMeierFitter()
        kmf.fit(rossi['week'], rossi['arrest'])

        rossi2 = rossi[['week', 'arrest']].copy()
        rossi2['var1'] = np.random.randn(432)

        cph = CoxPHFitter()
        cph.fit(rossi2, 'week', 'arrest')

        ax = cph.baseline_survival_.plot()
        kmf.plot(ax=ax)
        """
        survival_df = exp(-self.baseline_cumulative_hazard_)
        if self.strata is None:
            survival_df.columns = ["baseline survival"]
        return survival_df

    def plot(self, columns=None, display_significance_code=True, **errorbar_kwargs):
        """
        Produces a visual representation of the coefficients, including their standard errors and magnitudes.

        Parameters
        ----------
        columns : list, optional
            specifiy a subset of the columns to plot
        display_significance_code: bool, optional (default: True)
            display asteriks beside statistically significant variables
        errorbar_kwargs:
            pass in additional plotting commands to matplotlib errorbar command

        Returns
        -------
        ax: matplotlib axis
            the matplotlib axis that be edited.

        """
        from matplotlib import pyplot as plt

        ax = errorbar_kwargs.get("ax", None) or plt.figure().add_subplot(111)

        errorbar_kwargs.setdefault("c", "k")
        errorbar_kwargs.setdefault("fmt", "s")
        errorbar_kwargs.setdefault("markerfacecolor", "white")
        errorbar_kwargs.setdefault("markeredgewidth", 1.25)
        errorbar_kwargs.setdefault("elinewidth", 1.25)
        errorbar_kwargs.setdefault("capsize", 3)

        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)

        if columns is None:
            columns = self.hazards_.columns

        yaxis_locations = list(range(len(columns)))
        summary = self.summary.loc[columns]
        symmetric_errors = alpha2 * self.standard_errors_[columns].squeeze().values.copy()
        hazards = self.hazards_[columns].values[0].copy()

        order = np.argsort(hazards)

        ax.errorbar(hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
        ax.set_ylim(best_ylim)

        if display_significance_code:
            tick_labels = [c + significance_code(p).strip() for (c, p) in summary["p"][order].iteritems()]
        else:
            tick_labels = columns[order]

        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("log(HR) (%g%% CI)" % (self.alpha * 100))

        return ax

    def plot_covariate_groups(self, covariate, groups, **kwargs):
        """
        Produces a visual representation comparing the baseline survival curve of the model versus
        what happens when a covariate is varied over values in a group. This is useful to compare
        subjects' survival as we vary a single covariate, all else being held equal. The baseline survival
        curve is equal to the predicted survival curve at all average values in the original dataset.

        Parameters
        ----------
        covariate: string
            a string of the covariate in the original dataset that we wish to vary.
        groups: iterable
            an iterable of the values we wish the covariate to take on.
        kwargs:
            pass in additional plotting commands

        Returns
        -------
        ax: matplotlib axis
            the matplotlib axis that be edited.
        """
        from matplotlib import pyplot as plt

        if covariate not in self.hazards_.columns:
            raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        ax = kwargs.get("ax", None) or plt.figure().add_subplot(111)
        x_bar = self._norm_mean.to_frame().T
        X = pd.concat([x_bar] * len(groups))
        X.index = ["%s=%s" % (covariate, g) for g in groups]
        X[covariate] = groups

        self.predict_survival_function(X).plot(ax=ax)
        self.baseline_survival_.plot(ax=ax, ls="--")
        return ax

    def check_assumptions(self, df, help=True):
        """section 5 in https://socialsciences.mcmaster.ca/jfox/Books/Companion/appendices/Appendix-Cox-Regression.pdf
        http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf
        http://eprints.lse.ac.uk/84988/1/06_ParkHendry2015-ReassessingSchoenfeldTests_Final.pdf
        """
        pass

    @property
    def score_(self):
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the AUC
        to survival data, including censorships.

        For this purpose, the ``score_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset. It's analgous to the R^2 in linear models.

        """
        # pylint: disable=access-member-before-definition
        if hasattr(self, "_concordance_score_"):
            return self._concordance_score_
        self._concordance_score_ = concordance_index(
            self.durations, -self._predicted_partial_hazards_, self.event_observed
        )
        del self._predicted_partial_hazards_
        return self._concordance_score_
