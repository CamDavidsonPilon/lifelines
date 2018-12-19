# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from datetime import datetime
import warnings
import time

import numpy as np
import pandas as pd
from scipy import stats

from numpy import dot, exp
from numpy.linalg import norm, inv
from scipy.linalg import solve as spsolve
from lifelines.fitters import BaseFitter
from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import chisq_test
from lifelines.utils import (
    inv_normal_cdf,
    significance_code,
    normalize,
    significance_codes_as_text,
    pass_for_numeric_dtypes_or_raise,
    check_low_var,
    # check_for_overlapping_intervals,
    check_complete_separation_low_variance,
    ConvergenceWarning,
    StepSizer,
    _get_index,
    check_for_immediate_deaths,
    check_for_instantaneous_events,
    ConvergenceError,
    check_nans_or_infs,
    string_justify,
)


class CoxTimeVaryingFitter(BaseFitter):

    """
    This class implements fitting Cox's time-varying proportional hazard model:

    h(t|x(t)) = h_0(t)*exp(x(t)'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      penalizer: the coefficient of an l2 penalizer in the regression
    """

    def __init__(self, alpha=0.95, penalizer=0.0):
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha parameter must be between 0 and 1.")
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")

        self.alpha = alpha
        self.penalizer = penalizer

    def fit(
        self,
        df,
        id_col,
        event_col,
        start_col="start",
        stop_col="stop",
        weights_col=None,
        show_progress=False,
        step_size=None,
        robust=False,
    ):  # pylint: disable=too-many-arguments
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
          stop_col: the column that contains the end of a subject's time period.
          weights_col: the column that contains (possibly time-varying) weight of each subject-period row.
          show_progress: since the fitter is iterative, show convergence
             diagnostics.
          step_size: set an initial step size for the fitting algorithm.
          robust: Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078


        Returns:
            self, with additional properties: hazards_

        """

        self.robust = robust
        if self.robust:
            raise NotImplementedError("Not available yet.")

        self.event_col = event_col
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        df = df.copy()

        if not (id_col in df and event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the dataframe provided.")

        if weights_col is None:
            assert (
                "__weights" not in df.columns
            ), "__weights is an internal lifelines column, please rename your column first."
            df["__weights"] = 1.0
        else:
            if (df[weights_col] <= 0).any():
                raise ValueError("values in weights_col must be positive.")

        df = df.rename(
            columns={id_col: "id", event_col: "event", start_col: "start", stop_col: "stop", weights_col: "__weights"}
        )
        df = df.set_index("id")
        stop_times_events = df[["event", "stop", "start"]].copy()
        weights = df[["__weights"]].copy().astype(float)
        df = df.drop(["event", "stop", "start", "__weights"], axis=1)
        stop_times_events["event"] = stop_times_events["event"].astype(bool)

        self._check_values(df, stop_times_events)
        df = df.astype(float)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)

        hazards_ = self._newton_rhaphson(
            normalize(df, self._norm_mean, self._norm_std),
            stop_times_events,
            weights,
            show_progress=show_progress,
            step_size=step_size,
        )

        self.hazards_ = pd.DataFrame(hazards_.T, columns=df.columns, index=["coef"]) / self._norm_std
        self.variance_matrix_ = -inv(self._hessian_) / np.outer(self._norm_std, self._norm_std)
        self.standard_errors_ = self._compute_standard_errors(
            normalize(df, self._norm_mean, self._norm_std), stop_times_events, weights
        )
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_cumulative_hazard_ = self._compute_cumulative_baseline_hazard(df, stop_times_events, weights)
        self.baseline_survival_ = self._compute_baseline_survival()
        self.event_observed = stop_times_events["event"]
        self.start_stop_and_events = stop_times_events
        self.weights = weights

        self._n_examples = df.shape[0]
        self._n_unique = df.index.unique().shape[0]
        return self

    @staticmethod
    def _check_values(df, stop_times_events):
        # check_for_overlapping_intervals(df) # this is currenty too slow for production.
        check_nans_or_infs(df)
        check_low_var(df)
        check_complete_separation_low_variance(df, stop_times_events["event"])
        pass_for_numeric_dtypes_or_raise(df)
        check_for_immediate_deaths(stop_times_events)
        check_for_instantaneous_events(stop_times_events)

    def _compute_residuals(self, df, stop_times_events, weights):
        raise NotImplementedError()

    def _compute_delta_beta(self, df, stop_times_events, weights):
        """ approximate change in betas as a result of excluding ith row"""

        score_residuals = self._compute_residuals(df, stop_times_events, weights) * weights[:, None]

        naive_var = inv(self._hessian_)
        delta_betas = -score_residuals.dot(naive_var) / self._norm_std.values

        return delta_betas

    def _compute_sandwich_estimator(self, df, stop_times_events, weights):

        delta_betas = self._compute_delta_beta(df, stop_times_events, weights)

        if self.cluster_col:
            delta_betas = pd.DataFrame(delta_betas).groupby(self._clusters).sum().values

        sandwich_estimator = delta_betas.T.dot(delta_betas)
        return sandwich_estimator

    def _compute_standard_errors(self, df, stop_times_events, weights):
        if self.robust:
            se = np.sqrt(self._compute_sandwich_estimator(df, stop_times_events, weights).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.DataFrame(se[None, :], index=["se"], columns=self.hazards_.columns)

    def _compute_z_values(self):
        return self.hazards_.loc["coef"] / self.standard_errors_.loc["se"]

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        se = self.standard_errors_
        hazards = self.hazards_.values
        return pd.DataFrame(
            np.r_[hazards - alpha2 * se, hazards + alpha2 * se],
            index=["lower-bound", "upper-bound"],
            columns=self.hazards_.columns,
        )

    @property
    def summary(self):
        """
        Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns:
            df: DataFrame, contains columns coef, exp(coef), se(coef), z, p, lower, upper
        """
        df = pd.DataFrame(index=self.hazards_.columns)
        df["coef"] = self.hazards_.loc["coef"].values
        df["exp(coef)"] = exp(self.hazards_.loc["coef"].values)
        df["se(coef)"] = self.standard_errors_.loc["se"].values
        df["z"] = self._compute_z_values()
        df["p"] = self._compute_p_values()
        df["lower %.2f" % self.alpha] = self.confidence_intervals_.loc["lower-bound"].values
        df["upper %.2f" % self.alpha] = self.confidence_intervals_.loc["upper-bound"].values
        return df

    def _newton_rhaphson(
        self, df, stop_times_events, weights, show_progress=False, step_size=None, precision=10e-6, max_steps=50
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Note that data is assumed to be sorted on T!

        Parameters:
            df: (n, d) Pandas DataFrame of observations
            stop_times_events: (n, d) Pandas DataFrame of meta information about the subjects history
            show_progress: True to show verbous output of convergence
            step_size: float > 0 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.

        Returns:
            beta: (1,d) numpy array.
        """
        assert precision <= 1.0, "precision must be less than or equal to 1."

        _, d = df.shape

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
            h, g, ll = self._get_gradients(df, stop_times_events, weights, beta)

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta.T
                h.flat[:: d + 1] -= self.penalizer

            try:
                # reusing a piece to make g * inv(h) * g.T faster later
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
            newton_decrement = g.dot(inv_h_dot_g_T) / 2

            if show_progress:
                print(
                    "Iteration %d: norm_delta = %.5f, step_size = %.5f, ll = %.5f, newton_decrement = %.5f, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll, newton_decrement, time.time() - start)
                )

            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif previous_ll > 0 and abs(ll - previous_ll) / (-previous_ll) < 1e-09:
                # this is what R uses by default
                converging, completed = False, True
            elif newton_decrement < 10e-8:
                converging, completed = False, True
            elif i >= max_steps:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is less than 10 steps
                converging, completed = False, False
            elif step_size <= 0.0001:
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

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        if not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    def _get_gradients(self, df, stops_events, weights, beta):  # pylint: disable=too-many-locals
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

        unique_death_times = np.unique(stops_events["stop"].loc[stops_events["event"]])

        for t in unique_death_times:

            # I feel like this can be made into some tree-like structure
            ix = (stops_events["start"].values < t) & (t <= stops_events["stop"].values)

            df_at_t = df.values[ix]
            weights_at_t = weights.values[ix]
            stops_events_at_t = stops_events["stop"].values[ix]
            events_at_t = stops_events["event"].values[ix]

            phi_i = weights_at_t * exp(dot(df_at_t, beta))
            phi_x_i = phi_i * df_at_t
            phi_x_x_i = dot(df_at_t.T, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = phi_i.sum()
            risk_phi_x = phi_x_i.sum(0)
            risk_phi_x_x = phi_x_x_i

            # Calculate the sums of Tie set
            deaths = events_at_t & (stops_events_at_t == t)

            ties_counts = deaths.sum()  # should always at least 1

            xi_deaths = df_at_t[deaths]
            weights_deaths = weights_at_t[deaths]

            x_death_sum = (weights_deaths * xi_deaths).sum(0)

            if ties_counts > 1:
                # it's faster if we can skip computing these when we don't need to.
                tie_phi = phi_i[deaths].sum()
                tie_phi_x = phi_x_i[deaths].sum(0)
                tie_phi_x_x = dot(xi_deaths.T, phi_i[deaths] * xi_deaths)

            partial_gradient = np.zeros(d)
            weight_count = weights_deaths.sum()
            weighted_average = weight_count / ties_counts

            for l in range(ties_counts):

                if ties_counts > 1:

                    # A good explaination for how Efron handles ties. Consider three of five subjects who fail at the time.
                    # As it is not known a priori that who is the first to fail, so one-third of
                    # (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
                    # of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

                    increasing_proportion = l / ties_counts
                    denom = risk_phi - increasing_proportion * tie_phi
                    numer = risk_phi_x - increasing_proportion * tie_phi_x
                    # Hessian
                    a1 = (risk_phi_x_x - increasing_proportion * tie_phi_x_x) / denom
                else:
                    denom = risk_phi
                    numer = risk_phi_x
                    # Hessian
                    a1 = risk_phi_x_x / denom

                # Gradient
                partial_gradient += weighted_average * numer / denom
                # In case numer and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = np.outer(numer / denom, numer / denom)

                hessian -= weighted_average * (a1 - a2)
                log_lik -= weighted_average * np.log(denom)

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
            pass_for_numeric_dtypes_or_raise(X)

        X = X.astype(float)
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
        Print summary statistics describing the fit, the coefficients, and the error bounds.
        """
        # pylint: disable=unnecessary-lambda
        # Print information about data first
        justify = string_justify(18)
        print(self)
        print("{} = {}".format(justify("event col"), self.event_col))
        print("{} = {}".format(justify("number of subjects"), self._n_unique))
        print("{} = {}".format(justify("number of periods"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.3f}".format(justify("log-likelihood"), self._log_likelihood))
        print("{} = {} UTC".format(justify("time fit was run"), self._time_fit_was_called), end="\n\n")

        print("---")

        df = self.summary
        # Significance codes last
        df[""] = [significance_code(p) for p in df["p"]]
        print(df.to_string(float_format=lambda f: "{:4.4f}".format(f)))
        # Significance code explanation
        print("---")
        print(significance_codes_as_text(), end="\n\n")
        print("Likelihood ratio test = {:.3f} on {} df, p={:.5f}".format(*self._compute_likelihood_ratio_test()))

    def _compute_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Cox model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.

        Conveniently, we can actually use the class itself to do most of the work.

        """

        trivial_dataset = self.start_stop_and_events.groupby(level=0).last()[["event", "stop"]]
        weights = self.weights.groupby(level=0).last()[["__weights"]]
        trivial_dataset = trivial_dataset.join(weights)

        cp_null = CoxPHFitter()
        cp_null.fit(trivial_dataset, "stop", "event", weights_col='__weights', show_progress=False)

        ll_null = cp_null._log_likelihood
        ll_alt = self._log_likelihood

        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.hazards_.shape[1]
        _, p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom, alpha=0.0)
        return test_stat, degrees_freedom, p_value

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

        ax = kwargs.get("ax", None) or plt.figure().add_subplot(111)
        yaxis_locations = range(len(self.hazards_.columns))

        if columns is not None:
            yaxis_locations = range(len(columns))
            summary = self.summary.loc[columns]
            lower_bound = self.confidence_intervals_[columns].loc["lower-bound"].copy()
            upper_bound = self.confidence_intervals_[columns].loc["upper-bound"].copy()
            hazards = self.hazards_[columns].values[0].copy()
        else:
            yaxis_locations = range(len(self.hazards_.columns))
            summary = self.summary
            lower_bound = self.confidence_intervals_.loc["lower-bound"].copy()
            upper_bound = self.confidence_intervals_.loc["upper-bound"].copy()
            hazards = self.hazards_.values[0].copy()

        if standardized:
            se = summary["se(coef)"]
            lower_bound /= se
            upper_bound /= se
            hazards /= se

        order = np.argsort(hazards)
        ax.scatter(upper_bound.values[order], yaxis_locations, marker="|", c="k")
        ax.scatter(lower_bound.values[order], yaxis_locations, marker="|", c="k")
        ax.scatter(hazards[order], yaxis_locations, marker="o", c="k")
        ax.hlines(yaxis_locations, lower_bound.values[order], upper_bound.values[order], color="k", lw=1)

        tick_labels = [c + significance_code(p).strip() for (c, p) in summary["p"][order].iteritems()]
        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("standardized coef" if standardized else "coef")
        return ax

    def _compute_cumulative_baseline_hazard(
        self, tv_data, stop_times_events, weights
    ):  # pylint: disable=too-many-locals
        hazards = self.predict_partial_hazard(tv_data).values

        unique_death_times = np.unique(stop_times_events["stop"].loc[stop_times_events["event"]])
        baseline_hazard_ = pd.DataFrame(
            np.zeros_like(unique_death_times), index=unique_death_times, columns=["baseline hazard"]
        )

        for t in unique_death_times:
            ix = (stop_times_events["start"].values < t) & (t <= stop_times_events["stop"].values)

            events_at_t = stop_times_events["event"].values[ix]
            stops_at_t = stop_times_events["stop"].values[ix]
            weights_at_t = weights.values[ix]
            hazards_at_t = hazards[ix]

            deaths = events_at_t & (stops_at_t == t)

            death_counts = (weights_at_t.squeeze() * deaths).sum()  # should always be atleast 1.
            baseline_hazard_.loc[t] = death_counts / hazards_at_t.sum()

        return baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        survival_df = exp(-self.baseline_cumulative_hazard_)
        survival_df.columns = ["baseline survival"]
        return survival_df

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifelines.%s: fitted with %d periods, %d subjects, %d events>""" % (
                classname,
                self._n_examples,
                self._n_unique,
                self.event_observed.sum(),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s
