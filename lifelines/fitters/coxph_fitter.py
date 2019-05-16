# -*- coding: utf-8 -*-
import time
from datetime import datetime
import warnings
from textwrap import dedent, fill
import numpy as np
import pandas as pd

from numpy.linalg import norm, inv
from scipy.linalg import solve as spsolve, LinAlgError
from scipy.integrate import trapz
from scipy import stats
from bottleneck import nansum as array_sum_to_scalar

from lifelines.fitters import BaseFitter
from lifelines.plotting import set_kwargs_ax, set_kwargs_drawstyle
from lifelines.statistics import chisq_test, proportional_hazard_test, TimeTransformers, StatisticalResult
from lifelines.utils.lowess import lowess
from lifelines.utils.concordance import _concordance_summary_statistics, _concordance_ratio
from lifelines.utils import (
    _get_index,
    _to_list,
    _to_tuple,
    _to_array,
    inv_normal_cdf,
    normalize,
    qth_survival_times,
    coalesce,
    check_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation,
    check_nans_or_infs,
    StatError,
    ConvergenceWarning,
    StatisticalWarning,
    StepSizer,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
    format_exp_floats,
    dataframe_interpolate_at_times,
    CensoringType,
)

__all__ = ["CoxPHFitter"]

matrix_axis_0_sum_to_array = lambda m: np.sum(m, 0)


class BatchVsSingle:
    @staticmethod
    def decide(batch_mode, T):
        n_total = T.shape[0]
        n_unique = T.unique().shape[0]
        frac_dups = n_unique / n_total
        if batch_mode or (
            # https://github.com/CamDavidsonPilon/lifelines/issues/591 for original issue.
            # new values from from perf/batch_vs_single script.
            (batch_mode is None)
            and (0.712085 + -0.000025 * n_total + 0.579359 * frac_dups + 0.000044 * n_total * frac_dups < 1)
        ):
            return "batch"
        return "single"


class CoxPHFitter(BaseFitter):

    r"""
    This class implements fitting Cox's proportional hazard model:

    .. math::  h(t|x) = h_0(t) \exp((x - \overline{x})' \beta)

    Parameters
    ----------

      alpha: float, optional (default=0.05)
        the level in the confidence intervals.

      tie_method: string, optional
        specify how the fitter should deal with ties. Currently only
        'Efron' is available.

      penalizer: float, optional (default=0.0)
        Attach an L2 penalizer to the size of the coefficients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the absolute value of :math:`\beta_i`.
        The penalty is :math:`\frac{1}{2} \text{penalizer} ||\beta||^2`.

      strata: list, optional
        specify a list of columns to use in stratification. This is useful if a
         categorical covariate does not obey the proportional hazard assumption. This
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

    Attributes
    ----------
    hazards_ : Series
        The estimated hazards
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    durations: Series
        The durations provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    score_: float
        the concordance index of the model.
    baseline_hazard_: DataFrame
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """

    def __init__(self, alpha=0.05, tie_method="Efron", penalizer=0.0, strata=None):
        super(CoxPHFitter, self).__init__(alpha=alpha)
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")
        if tie_method != "Efron":
            raise NotImplementedError("Only Efron is available at the moment.")

        self.alpha = alpha
        self.tie_method = tie_method
        self.penalizer = penalizer
        self.strata = strata

    @CensoringType.right_censoring
    def fit(
        self,
        df,
        duration_col=None,
        event_col=None,
        show_progress=False,
        initial_point=None,
        strata=None,
        step_size=None,
        weights_col=None,
        cluster_col=None,
        robust=False,
        batch_mode=None,
    ):
        """
        Fit the Cox proportional hazard model to a dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of thecolumn in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        weights_col: string, optional
            an optional column in the DataFrame, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use `robust=True` to get more accurate standard errors.

        show_progress: boolean, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.

        step_size: float, optional
            set an initial step size for the fitting algorithm. Setting to 1.0 may improve performance, but could also hurt convergence.

        robust: boolean, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078

        cluster_col: string, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.

        batch_mode: bool, optional
            enabling batch_mode can be faster for datasets with a large number of ties. If left as None, lifelines will choose the best option.

        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: ``print_summary``, ``hazards_``, ``confidence_intervals_``, ``baseline_survival_``, etc.


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
        if duration_col is None:
            raise TypeError("duration_col cannot be None.")

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self.robust = robust
        self.cluster_col = cluster_col
        self.weights_col = weights_col
        self._n_examples = df.shape[0]
        self._batch_mode = batch_mode
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
        X_norm = normalize(X, self._norm_mean, self._norm_std)

        hazards_ = self._fit_model(
            X_norm, T, E, weights=weights, initial_point=initial_point, show_progress=show_progress, step_size=step_size
        )

        self.hazards_ = pd.Series(hazards_, index=X.columns, name="coef") / self._norm_std

        self.variance_matrix_ = -inv(self._hessian_) / np.outer(self._norm_std, self._norm_std)
        self.standard_errors_ = self._compute_standard_errors(X_norm, T, E, weights)
        self.confidence_intervals_ = self._compute_confidence_intervals()

        self._predicted_partial_hazards_ = (
            self.predict_partial_hazard(X)
            .rename(columns={0: "P"})
            .assign(T=self.durations.values, E=self.event_observed.values, W=self.weights.values)
            .set_index(X.index)
        )
        self.baseline_hazard_ = self._compute_baseline_hazards()
        self.baseline_cumulative_hazard_ = self._compute_baseline_cumulative_hazard()
        self.baseline_survival_ = self._compute_baseline_survival()

        if hasattr(self, "_concordance_score_"):
            # we have already fit the model.
            del self._concordance_score_

        return self

    def _preprocess_dataframe(self, df):
        # this should be a pure function

        df = df.copy()

        if self.strata is not None:
            df = df.sort_values(by=_to_list(self.strata) + [self.duration_col])
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
            else pd.Series(np.ones(self._n_examples), index=df.index, name="E")
        )
        W = (
            df.pop(self.weights_col)
            if (self.weights_col is not None)
            else pd.Series(np.ones((self._n_examples,)), index=df.index, name="weights")
        )

        _clusters = df.pop(self.cluster_col).values if self.cluster_col else None

        X = df.astype(float)
        T = T.astype(float)

        # we check nans here because converting to bools maps NaNs to True..
        check_nans_or_infs(E)
        E = E.astype(bool)

        self._check_values(X, T, E, W)

        return X, T, E, W, original_index, _clusters

    def _check_values(self, X, T, E, W):
        check_for_numeric_dtypes_or_raise(X)
        check_nans_or_infs(T)
        check_nans_or_infs(X)
        check_low_var(X)
        check_complete_separation(X, E, T, self.event_col)
        # check to make sure their weights are okay
        if self.weights_col:
            if (W.astype(int) != W).any() and not self.robust:
                warnings.warn(
                    """It appears your weights are not integers, possibly propensity or sampling scores then?
It's important to know that the naive variance estimates of the coefficients are biased. Instead a) set `robust=True` in the call to `fit`, or b) use Monte Carlo to
estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
""",
                    StatisticalWarning,
                )
            if (W <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

    def _fit_model(
        self,
        X,
        T,
        E,
        weights=None,
        initial_point=None,
        step_size=None,
        precision=1e-07,
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
        initial_point: (d,) numpy array of initial starting point for
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
            the maximum number of iterations of the Newton-Rhaphson algorithm.

        Returns
        -------
        beta: (1,d) numpy array.
        """
        self.path = []
        assert precision <= 1.0, "precision must be less than or equal to 1."
        _, d = X.shape

        # make sure betas are correct size.
        if initial_point is not None:
            assert initial_point.shape == (d,)
            beta = initial_point
        else:
            beta = np.zeros((d,))

        step_sizer = StepSizer(step_size)
        step_size = step_sizer.next()

        # Method of choice is just efron right now
        if self.tie_method == "Efron":
            decision = BatchVsSingle.decide(self._batch_mode, T)
            get_gradients = getattr(self, "_get_efron_values_%s" % decision)
            self._batch_mode = decision == "batch"
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
                g = np.zeros_like(beta)
                h = np.zeros((beta.shape[0], beta.shape[0]))
                ll = 0
                for _h, _g, _ll in self._partition_by_strata_and_apply(X, T, E, weights, get_gradients, beta):
                    g += _g
                    h += _h
                    ll += _ll

            if i == 1 and np.all(beta == 0):
                # this is a neat optimization, the null partial likelihood
                # is the same as the full partial but evaluated at zero.
                # if the user supplied a non-trivial initial point, we need to delay this.
                self._log_likelihood_null = ll

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta
                h.flat[:: d + 1] -= self.penalizer

            # reusing a piece to make g * inv(h) * g.T faster later
            try:
                inv_h_dot_g_T = spsolve(-h, g, assume_a="pos", check_finite=False)
            except ValueError as e:
                if "infs or NaNs" in str(e):
                    raise ConvergenceError(
                        """Hessian or gradient contains nan or inf value(s). Convergence halted. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
""",
                        e,
                    )
                else:
                    # something else?
                    raise e
            except LinAlgError as e:
                raise ConvergenceError(
                    """Convergence halted due to matrix inversion problems. Suspicion is high collinearity. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
""",
                    e,
                )

            delta = inv_h_dot_g_T

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
                    "\rIteration %d: norm_delta = %.5f, step_size = %.4f, ll = %.5f, newton_decrement = %.5f, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll, newton_decrement, time.time() - start),
                    end=""
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
                    "The log-likelihood is getting suspiciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression",
                    ConvergenceWarning,
                )
                converging, completed = False, False

            beta += step_size * delta

            previous_ll = ll
            step_size = step_sizer.update(norm_delta).next()

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        elif show_progress and not completed:
            print("Convergence failed. See any warning messages.")

        # report to the user problems that we detect.
        if completed and norm_delta > 0.1:
            warnings.warn(
                "Newton-Rhapson convergence completed but norm(delta) is still high, %.3f. This may imply non-unique solutions to the maximum likelihood. Perhaps there is collinearity or complete separation in the dataset?"
                % norm_delta,
                ConvergenceWarning,
            )
        elif not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    def _get_efron_values_single(self, X, T, E, weights, beta):
        """
        Calculates the first and second order vector differentials, with respect to beta.
        Note that X, T, E are assumed to be sorted on T!

        A good explanation for Efron. Consider three of five subjects who fail at the time.
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
        gradient = np.zeros((d,))
        log_lik = 0

        # Init risk and tie sums to zero
        x_death_sum = np.zeros((d,))
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((d,)), np.zeros((d,))
        risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

        # Init number of ties and weights
        weight_count = 0.0
        tied_death_counts = 0
        scores = weights * np.exp(np.dot(X, beta))

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]
            xi = X[i]
            score = scores[i]
            w = weights[i]

            # Calculate phi values
            phi_i = score
            phi_x_i = phi_i * xi
            phi_x_x_i = np.outer(xi, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = risk_phi + phi_i
            risk_phi_x = risk_phi_x + phi_x_i
            risk_phi_x_x = risk_phi_x_x + phi_x_x_i

            # Calculate sums of Ties, if this is an event
            if ei:
                x_death_sum = x_death_sum + w * xi
                tie_phi = tie_phi + phi_i
                tie_phi_x = tie_phi_x + phi_x_i
                tie_phi_x_x = tie_phi_x_x + phi_x_x_i

                # Keep track of count
                tied_death_counts += 1
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tied_death_counts == 0:
                # Only censored with current time, move on
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            #
            # This code is near identical to the _batch algorithm below. In fact, see _batch for comments.
            #
            weighted_average = weight_count / tied_death_counts

            if tied_death_counts > 1:
                increasing_proportion = np.arange(tied_death_counts) / tied_death_counts
                denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)
                numer = risk_phi_x - np.outer(increasing_proportion, tie_phi_x)
                a1 = np.einsum("ab,i->ab", risk_phi_x_x, denom) - np.einsum(
                    "ab,i->ab", tie_phi_x_x, increasing_proportion * denom
                )
            else:
                denom = 1.0 / np.array([risk_phi])
                numer = risk_phi_x
                a1 = risk_phi_x_x * denom

            summand = numer * denom[:, None]
            a2 = summand.T.dot(summand)

            gradient = gradient + x_death_sum - weighted_average * summand.sum(0)

            log_lik = log_lik + np.dot(x_death_sum, beta) + weighted_average * np.log(denom).sum()
            hessian = hessian + weighted_average * (a2 - a1)

            # reset tie values
            tied_death_counts = 0
            weight_count = 0.0
            x_death_sum = np.zeros((d,))
            tie_phi = 0
            tie_phi_x = np.zeros((d,))
            tie_phi_x_x = np.zeros((d, d))

        return hessian, gradient, log_lik

    @staticmethod
    def _trivial_log_likelihood_batch(T, E, weights):
        # used for log-likelihood test
        n = T.shape[0]
        log_lik = 0
        _, counts = np.unique(-T, return_counts=True)
        risk_phi = 0
        pos = n

        for count_of_removals in counts:

            slice_ = slice(pos - count_of_removals, pos)

            weights_at_t = weights[slice_]

            phi_i = weights_at_t

            # Calculate sums of Risk set
            risk_phi = risk_phi + array_sum_to_scalar(phi_i)

            # Calculate the sums of Tie set
            deaths = E[slice_]

            tied_death_counts = array_sum_to_scalar(deaths.astype(int))
            if tied_death_counts == 0:
                # no deaths, can continue
                pos -= count_of_removals
                continue

            weights_deaths = weights_at_t[deaths]
            weight_count = array_sum_to_scalar(weights_deaths)

            if tied_death_counts > 1:
                tie_phi = array_sum_to_scalar(phi_i[deaths])
                factor = np.log(risk_phi - np.arange(tied_death_counts) * tie_phi / tied_death_counts).sum()
            else:
                factor = np.log(risk_phi)

            log_lik = log_lik - weight_count / tied_death_counts * factor
            pos -= count_of_removals

        return log_lik

    @staticmethod
    def _trivial_log_likelihood_single(T, E, weights):
        # assumes sorted on T!

        log_lik = 0
        n = T.shape[0]

        # Init risk and tie sums to zero
        risk_phi, tie_phi = 0, 0
        # Init number of ties and weights
        weight_count = 0.0
        tied_death_counts = 0

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]

            # Calculate phi values
            phi_i = weights[i]
            w = weights[i]

            # Calculate sums of Risk set
            risk_phi = risk_phi + phi_i

            # Calculate sums of Ties, if this is an event
            if ei:
                tie_phi = tie_phi + phi_i

                # Keep track of count
                tied_death_counts += 1
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tied_death_counts == 0:
                # Only censored with current time, move on
                continue

            if tied_death_counts > 1:
                factor = np.log(risk_phi - np.arange(tied_death_counts) * tie_phi / tied_death_counts).sum()
            else:
                factor = np.log(risk_phi)
            log_lik = log_lik - weight_count / tied_death_counts * factor

            # reset tie values
            tied_death_counts = 0
            weight_count = 0.0
            tie_phi = 0
        return log_lik

    def _get_efron_values_batch(self, X, T, E, weights, beta):  # pylint: disable=too-many-locals
        """
        Assumes sorted on ascending on T
        Calculates the first and second order vector differentials, with respect to beta.

        A good explanation for how Efron handles ties. Consider three of five subjects who fail at the time.
        As it is not known a priori that who is the first to fail, so one-third of
        (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
        of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

        Returns
        -------
        hessian: (d, d) numpy array,
        gradient: (1, d) numpy array
        log_likelihood: float
        """

        n, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros((d,))
        log_lik = 0
        # weights = weights[:, None]

        # Init risk and tie sums to zero
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((d,)), np.zeros((d,))
        risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

        # counts are sorted by -T
        _, counts = np.unique(-T, return_counts=True)
        scores = weights * np.exp(np.dot(X, beta))
        pos = n

        for count_of_removals in counts:

            slice_ = slice(pos - count_of_removals, pos)

            X_at_t = X[slice_]
            weights_at_t = weights[slice_]

            phi_i = scores[slice_, None]
            phi_x_i = phi_i * X_at_t
            phi_x_x_i = np.dot(X_at_t.T, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = risk_phi + array_sum_to_scalar(phi_i)
            risk_phi_x = risk_phi_x + matrix_axis_0_sum_to_array(phi_x_i)
            risk_phi_x_x = risk_phi_x_x + phi_x_x_i

            # Calculate the sums of Tie set
            deaths = E[slice_]

            tied_death_counts = array_sum_to_scalar(deaths.astype(int))
            if tied_death_counts == 0:
                # no deaths, can continue
                pos -= count_of_removals
                continue

            xi_deaths = X_at_t[deaths]
            weights_deaths = weights_at_t[deaths]

            x_death_sum = matrix_axis_0_sum_to_array(weights_deaths[:, None] * xi_deaths)

            weight_count = array_sum_to_scalar(weights_deaths)
            weighted_average = weight_count / tied_death_counts

            if tied_death_counts > 1:

                # a lot of this is now in Einstein notation for performance, but see original "expanded" code here
                # https://github.com/CamDavidsonPilon/lifelines/blob/e7056e7817272eb5dff5983556954f56c33301b1/lifelines/fitters/coxph_fitter.py#L755-L789

                # it's faster if we can skip computing these when we don't need to.
                tie_phi = array_sum_to_scalar(phi_i[deaths])
                tie_phi_x = matrix_axis_0_sum_to_array(phi_x_i[deaths])
                tie_phi_x_x = np.dot(xi_deaths.T, phi_i[deaths] * xi_deaths)

                increasing_proportion = np.arange(tied_death_counts) / tied_death_counts
                denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)
                numer = risk_phi_x - np.outer(increasing_proportion, tie_phi_x)

                # computes outer products and sums them together.
                # Naive approach is to
                # 1) broadcast tie_phi_x_x and increasing_proportion into a (tied_death_counts, d, d) matrix
                # 2) broadcast risk_phi_x_x and denom into a (tied_death_counts, d, d) matrix
                # 3) subtract them, and then sum to (d, d)
                # Alternatively, we can sum earlier without having to explicitly create (_, d, d) matrices. This is used here.
                #
                a1 = np.einsum("ab,i->ab", risk_phi_x_x, denom) - np.einsum(
                    "ab,i->ab", tie_phi_x_x, increasing_proportion * denom
                )
            else:
                # no tensors here, but do some casting to make it easier in the converging step next.
                denom = 1.0 / np.array([risk_phi])
                numer = risk_phi_x
                a1 = risk_phi_x_x * denom

            summand = numer * denom[:, None]
            # This is a batch outer product.
            # given a matrix t, for each row, m, compute it's outer product: m.dot(m.T), and stack these new matrices together.
            # which would be: np.einsum("Bi, Bj->Bij", t, t)
            a2 = summand.T.dot(summand)

            gradient = gradient + x_death_sum - weighted_average * summand.sum(0)
            log_lik = log_lik + np.dot(x_death_sum, beta) + weighted_average * np.log(denom).sum()
            hessian = hessian + weighted_average * (a2 - a1)
            pos -= count_of_removals

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
        # TODO: _weights unused
        partial_hazard = self.predict_partial_hazard(X)[0].values

        if not self.strata:
            baseline_at_T = self.baseline_cumulative_hazard_.loc[T, "baseline hazard"].values
        else:
            baseline_at_T = np.empty(0)
            for name, T_ in T.groupby(by=self.strata):
                baseline_at_T = np.append(baseline_at_T, self.baseline_cumulative_hazard_[name].loc[T_])

        martingale = E - (partial_hazard * baseline_at_T)
        return pd.DataFrame(
            {self.duration_col: T.values, self.event_col: E.values, "martingale": martingale.values}, index=index
        )

    def _compute_deviance(self, X, T, E, weights, index=None):
        df = self._compute_martingale(X, T, E, weights, index)
        rmart = df.pop("martingale")

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore")
            log_term = np.where((E.values - rmart.values) <= 0, 0, E.values * np.log(E.values - rmart.values))

        deviance = np.sign(rmart) * np.sqrt(-2 * (rmart + log_term))
        df["deviance"] = deviance
        return df

    def _compute_scaled_schoenfeld(self, X, T, E, weights, index=None):
        r"""
        Let s_k be the kth schoenfeld residuals. Then E[s_k] = 0.
        For tests of proportionality, we want to test if \beta_i(t) is \beta_i (constant) or not.

        Let V_k be the contribution to the information matrix at time t_k. A main result from Grambsch and Therneau is that

        \beta(t) = E[s_k*V_k^{-1} + \hat{beta}]

        so define s_k^* = s_k*V_k^{-1} + \hat{beta} as the scaled schoenfeld residuals.

        We can approximate V_k with Hessian/d, so the inverse of Hessian/d is (d * variance_matrix_)

        Notes
        -------
        lifelines does not add the coefficients to the final results, but R does when you call residuals(c, "scaledsch")


        """

        n_deaths = self.event_observed.sum()
        scaled_schoenfeld_resids = n_deaths * self._compute_schoenfeld(X, T, E, weights, index).dot(
            self.variance_matrix_
        )

        scaled_schoenfeld_resids.columns = self.hazards_.index
        return scaled_schoenfeld_resids

    def _compute_schoenfeld(self, X, T, E, weights, index=None):
        # TODO: should the index by times, i.e. T[E]?

        # Assumes sorted on T and on strata
        # cluster does nothing to this, as expected.

        _, d = X.shape

        if self.strata is not None:
            schoenfeld_residuals = np.empty((0, d))

            for schoenfeld_residuals_in_strata in self._partition_by_strata_and_apply(
                X, T, E, weights, self._compute_schoenfeld_within_strata
            ):
                schoenfeld_residuals = np.append(schoenfeld_residuals, schoenfeld_residuals_in_strata, axis=0)

        else:
            schoenfeld_residuals = self._compute_schoenfeld_within_strata(X.values, T.values, E.values, weights.values)

        # schoenfeld residuals are only defined for subjects with a non-zero event.
        df = pd.DataFrame(schoenfeld_residuals[E, :], columns=self.hazards_.index, index=index[E])
        return df

    def _compute_schoenfeld_within_strata(self, X, T, E, weights):
        """
        A positive value of the residual shows an X value that is higher than expected at that death time.
        """
        # TODO: the diff_against is gross
        # This uses Efron ties.

        n, d = X.shape

        if not np.any(E):
            # sometimes strata have no deaths. This means nothing is returned
            # in the below code.
            return np.zeros((n, d))

        # Init risk and tie sums to zero
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))

        # Init number of ties and weights
        weight_count = 0.0
        tie_count = 0

        scores = weights * np.exp(np.dot(X, self.hazards_))

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
            risk_phi = risk_phi + phi_i
            risk_phi_x = risk_phi_x + phi_x_i

            # Calculate sums of Ties, if this is an event
            diff_against.append((xi, ei))
            if ei:

                tie_phi = tie_phi + phi_i
                tie_phi_x = tie_phi_x + phi_x_i

                # Keep track of count
                tie_count += 1  # aka death counts
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                for _ in diff_against:
                    schoenfeld_residuals = np.append(schoenfeld_residuals, np.zeros((1, d)), axis=0)
                diff_against = []
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
        approximate change in betas as a result of excluding ith row. Good for finding outliers / specific
        subjects that influence the model disproportionately. Good advice: don't drop these outliers, model them.
        """
        score_residuals = self._compute_score(X, T, E, weights, index=index)

        d = X.shape[1]
        scaled_variance_matrix = self.variance_matrix_ * np.tile(self._norm_std.values, (d, 1)).T

        delta_betas = score_residuals.dot(scaled_variance_matrix)
        delta_betas.columns = self.hazards_.index

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

        return pd.DataFrame(score_residuals, columns=self.hazards_.index, index=index)

    def _compute_score_within_strata(self, X, _T, E, weights):
        # https://www.stat.tamu.edu/~carroll/ftp/gk001.pdf
        # lin1989
        # https://www.ics.uci.edu/~dgillen/STAT255/Handouts/lecture10.pdf
        # Assumes X already sorted by T with strata
        # TODO: doesn't handle ties.
        # TODO: _T unused

        n, d = X.shape

        # we already unnormalized the betas in `fit`, so we need normalize them again since X is
        # normalized.
        beta = self.hazards_.values * self._norm_std

        E = E.astype(int)
        score_residuals = np.zeros((n, d))

        phi_s = np.exp(np.dot(X, beta))

        # need to store these histories, as we access them often
        # this is a reverse cumulative sum. See original code in https://github.com/CamDavidsonPilon/lifelines/pull/496/files#diff-81ee0759dbae0770e1a02cf17f4cfbb1R431
        risk_phi_x_history = (X * (weights * phi_s)[:, None])[::-1].cumsum(0)[::-1]
        risk_phi_history = (weights * phi_s)[::-1].cumsum()[::-1][:, None]

        # Iterate forwards
        for i in range(0, n):

            xi = X[i : i + 1]
            phi_i = phi_s[i]

            score = -phi_i * matrix_axis_0_sum_to_array(
                (
                    E[: i + 1] * weights[: i + 1] / risk_phi_history[: i + 1].T
                ).T  # this is constant-ish, and could be cached
                * (xi - risk_phi_x_history[: i + 1] / risk_phi_history[: i + 1])
            )

            if E[i]:
                score = score + (xi - risk_phi_x_history[i] / risk_phi_history[i])

            score_residuals[i, :] = score

        return score_residuals * weights[:, None]

    def compute_residuals(self, training_dataframe, kind):
        """

        Parameters
        ----------
        training_dataframe : pandas DataFrame
            the same training DataFrame given in `fit`
        kind : string
            {'schoenfeld', 'score', 'delta_beta', 'deviance', 'martingale', 'scaled_schoenfeld'}

        """
        ALLOWED_RESIDUALS = {"schoenfeld", "score", "delta_beta", "deviance", "martingale", "scaled_schoenfeld"}
        assert kind in ALLOWED_RESIDUALS, "kind must be in %s" % ALLOWED_RESIDUALS

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X, T, E, weights, shuffled_original_index, _ = self._preprocess_dataframe(training_dataframe)

        resids = getattr(self, "_compute_%s" % kind)(X, T, E, weights, index=shuffled_original_index)
        return resids

    def _compute_confidence_intervals(self):
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        hazards = self.hazards_.values
        return pd.DataFrame(
            np.c_[hazards - z * se, hazards + z * se], columns=["lower-bound", "upper-bound"], index=self.hazards_.index
        )

    def _compute_standard_errors(self, X, T, E, weights):
        if self.robust or self.cluster_col:
            se = np.sqrt(self._compute_sandwich_estimator(X, T, E, weights).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.Series(se, name="se", index=self.hazards_.index)

    def _compute_sandwich_estimator(self, X, T, E, weights):
        delta_betas = self._compute_delta_beta(X, T, E, weights)

        if self.cluster_col:
            delta_betas = delta_betas.groupby(self._clusters).sum()

        sandwich_estimator = delta_betas.T.dot(delta_betas)

        return sandwich_estimator.values

    def _compute_z_values(self):
        return self.hazards_ / self.standard_errors_

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
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
        ci = 1 - self.alpha
        with np.errstate(invalid="ignore", divide="ignore"):
            df = pd.DataFrame(index=self.hazards_.index)
            df["coef"] = self.hazards_
            df["exp(coef)"] = np.exp(self.hazards_)
            df["se(coef)"] = self.standard_errors_
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -np.log2(df["p"])
            df["lower %g" % ci] = self.confidence_intervals_["lower-bound"]
            df["upper %g" % ci] = self.confidence_intervals_["upper-bound"]
            return df

    def print_summary(self, decimals=2, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        # Print information about data first
        justify = string_justify(18)
        print(self)
        print("{} = '{}'".format(justify("duration col"), self.duration_col))

        if self.event_col:
            print("{} = '{}'".format(justify("event col"), self.event_col))
        if self.weights_col:
            print("{} = '{}'".format(justify("weights col"), self.weights_col))

        if self.cluster_col:
            print("{} = '{}'".format(justify("cluster col"), self.cluster_col))

        if self.robust or self.cluster_col:
            print("{} = {}".format(justify("robust variance"), True))

        if self.strata:
            print("{} = {}".format(justify("strata"), self.strata))

        if self.penalizer > 0:
            print("{} = {}".format(justify("penalizer"), self.penalizer))

        print("{} = {}".format(justify("number of subjects"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.{prec}f}".format(justify("partial log-likelihood"), self._log_likelihood, prec=decimals))
        print("{} = {}".format(justify("time fit was run"), self._time_fit_was_called))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        # Significance codes as last column
        print(
            df.to_string(
                float_format=format_floats(decimals),
                formatters={"p": format_p_value(decimals), "exp(coef)": format_exp_floats(decimals)},
            )
        )

        # Significance code explanation
        print("---")
        print("Concordance = {:.{prec}f}".format(self.score_, prec=decimals))
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = self.log_likelihood_ratio_test()
            print(
                "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                    sr.test_statistic, sr.degrees_freedom, -np.log2(sr.p_value), prec=decimals
                )
            )

    def log_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Cox model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.
        """
        if hasattr(self, "_log_likelihood_null"):
            ll_null = self._log_likelihood_null
        else:
            if self._batch_mode:
                ll_null = self._trivial_log_likelihood_batch(
                    self.durations.values, self.event_observed.values, self.weights.values
                )
            else:
                ll_null = self._trivial_log_likelihood_single(
                    self.durations.values, self.event_observed.values, self.weights.values
                )
        ll_alt = self._log_likelihood
        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.hazards_.shape[0]
        p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value,
            test_stat,
            name="log-likelihood ratio test",
            null_distribution="chi squared",
            degrees_freedom=degrees_freedom,
        )

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
            baseline hazard is not included. Equal to :math:`\exp{(x - mean(x_{train}))'\beta}`

        Notes
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        return np.exp(self.predict_log_partial_hazard(X))

    def predict_log_partial_hazard(self, X):
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - mean(x_{train}))'\beta `


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
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        hazard_names = self.hazards_.index

        if isinstance(X, pd.Series) and ((X.shape[0] == len(hazard_names) + 2) or (X.shape[0] == len(hazard_names))):
            X = X.to_frame().T
            return self.predict_log_partial_hazard(X)
        elif isinstance(X, pd.Series):
            assert len(hazard_names) == 1, "Series not the correct argument"
            X = X.to_frame().T
            return self.predict_log_partial_hazard(X)

        index = _get_index(X)

        if isinstance(X, pd.DataFrame):
            order = hazard_names
            X = X.reindex(order, axis="columns")
            check_for_numeric_dtypes_or_raise(X)
            X = X.values

        X = X.astype(float)

        X = normalize(X, self._norm_mean.values, 1)
        return pd.DataFrame(np.dot(X, self.hazards_), index=index)

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
            return dataframe_interpolate_at_times(cumulative_hazard_, times)
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
        return np.exp(-self.predict_cumulative_hazard(X, times=times))

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
        Compute the expected lifetime, :math:`E[T]`, using covariates X. This algorithm to compute the expectation is
        to use the fact that :math:`E[T] = \int_0^\inf P(T > t) dt = \int_0^\inf S(t) dt`. To compute the integral, we use the trapizoidal rule to approximate the integral.

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
        If X is a DataFrame, the order of the columns do not matter. But
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

    def _compute_baseline_hazard(self, partial_hazards, name):
        # https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
        ind_hazards = partial_hazards.copy()
        ind_hazards["P"] *= ind_hazards["W"]
        ind_hazards["E"] *= ind_hazards["W"]
        ind_hazards_summed_over_durations = ind_hazards.groupby("T")[["P", "E"]].sum()
        ind_hazards_summed_over_durations["P"] = ind_hazards_summed_over_durations["P"].loc[::-1].cumsum()
        baseline_hazard = pd.DataFrame(
            ind_hazards_summed_over_durations["E"] / ind_hazards_summed_over_durations["P"], columns=[name]
        )
        return baseline_hazard

    def _compute_baseline_hazards(self):
        if self.strata:

            index = self.durations.unique()
            baseline_hazards_ = pd.DataFrame(index=index).sort_index()

            for name, stratum_predicted_partial_hazards_ in self._predicted_partial_hazards_.groupby(self.strata):
                baseline_hazards_ = baseline_hazards_.merge(
                    self._compute_baseline_hazard(stratum_predicted_partial_hazards_, name),
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            return baseline_hazards_.fillna(0)

        return self._compute_baseline_hazard(self._predicted_partial_hazards_, name="baseline hazard")

    def _compute_baseline_cumulative_hazard(self):
        return self.baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        """
        Importantly, this agrees with what the KaplanMeierFitter produces. Ex:

        Example
        -------
        >>> from lifelines.datasets import load_rossi
        >>> from lifelines import CoxPHFitter, KaplanMeierFitter
        >>> rossi = load_rossi()
        >>> kmf = KaplanMeierFitter()
        >>> kmf.fit(rossi['week'], rossi['arrest'])
        >>> rossi2 = rossi[['week', 'arrest']].copy()
        >>> rossi2['var1'] = np.random.randn(432)
        >>> cph = CoxPHFitter()
        >>> cph.fit(rossi2, 'week', 'arrest')
        >>> ax = cph.baseline_survival_.plot()
        >>> kmf.plot(ax=ax)
        """
        survival_df = np.exp(-self.baseline_cumulative_hazard_)
        if self.strata is None:
            survival_df.columns = ["baseline survival"]
        return survival_df

    def plot(self, columns=None, **errorbar_kwargs):
        """
        Produces a visual representation of the coefficients, including their standard errors and magnitudes.

        Parameters
        ----------
        columns : list, optional
            specify a subset of the columns to plot
        errorbar_kwargs:
            pass in additional plotting commands to matplotlib errorbar command

        Returns
        -------
        ax: matplotlib axis
            the matplotlib axis that be edited.

        """
        from matplotlib import pyplot as plt

        set_kwargs_ax(errorbar_kwargs)
        ax = errorbar_kwargs.pop("ax")
        errorbar_kwargs.setdefault("c", "k")
        errorbar_kwargs.setdefault("fmt", "s")
        errorbar_kwargs.setdefault("markerfacecolor", "white")
        errorbar_kwargs.setdefault("markeredgewidth", 1.25)
        errorbar_kwargs.setdefault("elinewidth", 1.25)
        errorbar_kwargs.setdefault("capsize", 3)

        z = inv_normal_cdf(1 - self.alpha / 2)

        if columns is None:
            columns = self.hazards_.index

        yaxis_locations = list(range(len(columns)))
        symmetric_errors = z * self.standard_errors_[columns].to_frame().squeeze(axis=1).values.copy()
        hazards = self.hazards_.loc[columns].values.copy()

        order = np.argsort(hazards)

        ax.errorbar(hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
        ax.set_ylim(best_ylim)

        tick_labels = [columns[i] for i in order]

        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("log(HR) (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def plot_covariate_groups(self, covariates, values, plot_baseline=True, **kwargs):
        """
        Produces a visual representation comparing the baseline survival curve of the model versus
        what happens when a covariate(s) is varied over values in a group. This is useful to compare
        subjects' survival as we vary covariate(s), all else being held equal. The baseline survival
        curve is equal to the predicted survival curve at all average values in the original dataset.

        Parameters
        ----------
        covariates: string or list
            a string (or list of strings) of the covariate(s) in the original dataset that we wish to vary.
        values: 1d or 2d iterable
            an iterable of the values we wish the covariate(s) to take on.
        plot_baseline: bool
            also display the baseline survival, defined as the survival at the mean of the original dataset.
        kwargs:
            pass in additional plotting commands.

        Returns
        -------
        ax: matplotlib axis, or list of axis'
            the matplotlib axis that be edited.


        Examples
        ---------
        >>> from lifelines import datasets, CoxPHFitter
        >>> rossi = datasets.load_rossi()
        >>> cph = CoxPHFitter().fit(rossi, 'week', 'arrest')
        >>> cph.plot_covariate_groups('prio', values=np.arange(0, 15), cmap='coolwarm')

        >>> # multiple variables at once
        >>> cph.plot_covariate_groups(['prio', 'paro'], values=[[0, 0], [5, 0], [10, 0], [0, 1], [5, 1], [10, 1]], cmap='coolwarm')

        >>> # if you have categorical variables, you can simply things:
        >>> cph.plot_covariate_groups(['dummy1', 'dummy2', 'dummy3'], values=np.eye(3))

        """
        from matplotlib import pyplot as plt

        covariates = _to_list(covariates)
        n_covariates = len(covariates)
        values = _to_array(values)
        if len(values.shape) == 1:
            values = values[None, :].T

        if n_covariates != values.shape[1]:
            raise ValueError("The number of covariates must equal to second dimension of the values array.")

        for covariate in covariates:
            if covariate not in self.hazards_.index:
                raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        set_kwargs_drawstyle(kwargs, "steps-post")

        if self.strata is None:
            axes = kwargs.pop("ax", None) or plt.figure().add_subplot(111)
            x_bar = self._norm_mean.to_frame().T
            X = pd.concat([x_bar] * values.shape[0])

            if np.array_equal(np.eye(n_covariates), values):
                X.index = ["%s=1" % c for c in covariates]
            else:
                X.index = [", ".join("%s=%g" % (c, v) for (c, v) in zip(covariates, row)) for row in values]
            for covariate, value in zip(covariates, values.T):
                X[covariate] = value

            self.predict_survival_function(X).plot(ax=axes, **kwargs)
            if plot_baseline:
                self.baseline_survival_.plot(ax=axes, ls=":", color="k", drawstyle="steps-post")

        else:
            axes = []
            for stratum, baseline_survival_ in self.baseline_survival_.iteritems():
                ax = plt.figure().add_subplot(1, 1, 1)
                x_bar = self._norm_mean.to_frame().T

                for name, value in zip(_to_list(self.strata), _to_tuple(stratum)):
                    x_bar[name] = value

                X = pd.concat([x_bar] * values.shape[0])
                if np.array_equal(np.eye(len(covariates)), values):
                    X.index = ["%s=1" % c for c in covariates]
                else:
                    X.index = [", ".join("%s=%g" % (c, v) for (c, v) in zip(covariates, row)) for row in values]
                for covariate, value in zip(covariates, values.T):
                    X[covariate] = value

                self.predict_survival_function(X).plot(ax=ax, **kwargs)
                if plot_baseline:
                    baseline_survival_.plot(
                        ax=ax, ls=":", label="stratum %s baseline survival" % str(stratum), drawstyle="steps-post"
                    )
                plt.legend()
                axes.append(ax)
        return axes

    def check_assumptions(
        self, training_df, advice=True, show_plots=False, p_value_threshold=0.01, plot_n_bootstraps=10
    ):
        """
        Use this function to test the proportional hazards assumption. See usage example at
        https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html


        Parameters
        -----------

        training_df: DataFrame
            the original DataFrame used in the call to ``fit(...)`` or a sub-sampled version.
        advice: boolean, optional
            display advice as output to the user's screen
        show_plots: boolean, optional
            display plots of the scaled schoenfeld residuals and loess curves. This is an eyeball test for violations.
            This will slow down the function significantly.
        p_value_threshold: float, optional
            the threshold to use to alert the user of violations. See note below.
        plot_n_bootstraps:
            in the plots displayed, also display plot_n_bootstraps bootstrapped loess curves. This will slow down
            the function significantly.


        Examples
        ----------

        >>> from lifelines.datasets import load_rossi
        >>> from lifelines import CoxPHFitter
        >>>
        >>> rossi = load_rossi()
        >>> cph = CoxPHFitter().fit(rossi, 'week', 'arrest')
        >>>
        >>> cph.check_assumptions(rossi)


        Notes
        -------
        The ``p_value_threshold`` is arbitrarily set at 0.01. Under the null, some covariates
        will be below the threshold (i.e. by chance). This is compounded when there are many covariates.

        Similarly, when there are lots of observations, even minor deviances from the proportional hazard
        assumption will be flagged.

        With that in mind, it's best to use a combination of statistical tests and eyeball tests to
        determine the most serious violations.


        References
        -----------
        section 5 in https://socialsciences.mcmaster.ca/jfox/Books/Companion/appendices/Appendix-Cox-Regression.pdf,
        http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf,
        http://eprints.lse.ac.uk/84988/1/06_ParkHendry2015-ReassessingSchoenfeldTests_Final.pdf
        """

        if not training_df.index.is_unique:
            raise IndexError(
                "`training_df` index should be unique for this exercise. Please make it unique or use `.reset_index(drop=True)` to force a unique index"
            )

        residuals = self.compute_residuals(training_df, kind="scaled_schoenfeld")
        test_results = proportional_hazard_test(
            self, training_df, time_transform=["rank", "km"], precomputed_residuals=residuals
        )

        residuals_and_duration = residuals.join(training_df[self.duration_col])

        counter = 0
        n = residuals_and_duration.shape[0]

        for variable in self.hazards_.index:
            minumum_observed_p_value = test_results.summary.loc[variable, "p"].min()
            if np.round(minumum_observed_p_value, 2) > p_value_threshold:
                continue

            counter += 1

            if counter == 1:
                if advice:
                    print(
                        fill(
                            """The ``p_value_threshold`` is set at %g. Even under the null hypothesis of no violations, some covariates will be below the threshold by chance. This is compounded when there are many covariates. Similarly, when there are lots of observations, even minor deviances from the proportional hazard assumption will be flagged."""
                            % p_value_threshold,
                            width=100,
                        )
                    )
                    print()
                    print(
                        fill(
                            """With that in mind, it's best to use a combination of statistical tests and visual tests to determine the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)`` and looking for non-constant lines. See link [A] below for a full example.""",
                            width=100,
                        )
                    )
                    print()
                test_results.print_summary()
                print()

            print()
            print(
                "%d. Variable '%s' failed the non-proportional test: p-value is %s."
                % (counter, variable, format_p_value(4)(minumum_observed_p_value)),
                end="\n\n",
            )

            if advice:
                values = training_df[variable]
                value_counts = values.value_counts()
                n_uniques = value_counts.shape[0]

                # Arbitrary chosen 10 and 4 to check for ability to use strata col.
                # This should capture dichotomous / low cardinality values.
                if n_uniques <= 10 and value_counts.min() >= 5:
                    print(
                        fill(
                            "   Advice: with so few unique values (only {0}), you can include `strata=['{1}', ...]` in the call in `.fit`. See documentation in link [E] below.".format(
                                n_uniques, variable
                            ),
                            width=100,
                        )
                    )
                else:
                    print(
                        fill(
                            """   Advice 1: the functional form of the variable '{var}' might be incorrect. That is, there may be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect functional forms. See documentation in link [D] below on how to specify a functional form.""".format(
                                var=variable
                            ),
                            width=100,
                        ),
                        end="\n\n",
                    )
                    print(
                        fill(
                            """   Advice 2: try binning the variable '{var}' using pd.cut, and then specify it in `strata=['{var}', ...]` in the call in `.fit`. See documentation in link [B] below.""".format(
                                var=variable
                            ),
                            width=100,
                        ),
                        end="\n\n",
                    )
                    print(
                        fill(
                            """   Advice 3: try adding an interaction term with your time variable. See documentation in link [C] below.""".format(
                                var=variable
                            ),
                            width=100,
                        ),
                        end="\n\n",
                    )

            if show_plots:

                from matplotlib import pyplot as plt

                fig = plt.figure()

                # plot variable against all time transformations.
                for i, (transform_name, transformer) in enumerate(TimeTransformers().iter(["rank", "km"]), start=1):
                    p_value = test_results.summary.loc[(variable, transform_name), "p"]

                    ax = fig.add_subplot(1, 2, i)

                    y = residuals_and_duration[variable]
                    tt = transformer(self.durations, self.event_observed, self.weights)[self.event_observed.values]

                    ax.scatter(tt, y, alpha=0.75)

                    y_lowess = lowess(tt.values, y.values)
                    ax.plot(tt, y_lowess, color="k", alpha=1.0, linewidth=2)

                    # bootstrap some possible other lowess lines. This is an approximation of the 100% confidence intervals
                    for _ in range(plot_n_bootstraps):
                        ix = sorted(np.random.choice(n, n))
                        tt_ = tt.values[ix]
                        y_lowess = lowess(tt_, y.values[ix])
                        ax.plot(tt_, y_lowess, color="k", alpha=0.30)

                    best_xlim = ax.get_xlim()
                    ax.hlines(0, 0, tt.max(), linestyles="dashed", linewidths=1)
                    ax.set_xlim(best_xlim)

                    ax.set_xlabel("%s-transformed time\n(p=%.4f)" % (transform_name, p_value), fontsize=10)

                fig.suptitle("Scaled Schoenfeld residuals of '%s'" % variable, fontsize=14)
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)

        if advice and counter > 0:
            print(
                dedent(
                    r"""
                ---
                [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
                [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
                [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
                [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
                [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
            """
                )
            )

        if counter == 0:
            print("Proportional hazard assumption looks okay.")

    @property
    def score_(self):
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the ROC AUC
        to survival data, including censorships.

        For this purpose, the ``score_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset.

        References
        ----------
        https://stats.stackexchange.com/questions/133817/stratified-concordance-index-survivalsurvconcordance

        """
        # pylint: disable=access-member-before-definition
        if not hasattr(self, "_concordance_score_"):
            if self.strata:
                # https://stats.stackexchange.com/questions/133817/stratified-concordance-index-survivalsurvconcordance
                num_correct, num_tied, num_pairs = 0, 0, 0
                for _, _df in self._predicted_partial_hazards_.groupby(self.strata):
                    if _df.shape[0] == 1:
                        continue
                    _num_correct, _num_tied, _num_pairs = _concordance_summary_statistics(
                        _df["T"].values, -_df["P"].values, _df["E"].values
                    )
                    num_correct += _num_correct
                    num_tied += _num_tied
                    num_pairs += _num_pairs
            else:
                df = self._predicted_partial_hazards_
                num_correct, num_tied, num_pairs = _concordance_summary_statistics(
                    df["T"].values, -df["P"].values, df["E"].values
                )

            self._concordance_score_ = _concordance_ratio(num_correct, num_tied, num_pairs)
            return self._concordance_score_
        return self._concordance_score_
