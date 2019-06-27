# -*- coding: utf-8 -*-
import collections
from functools import wraps
import sys
import warnings
from datetime import datetime

# pylint: disable=wrong-import-position
warnings.simplefilter(action="ignore", category=FutureWarning)

from textwrap import dedent

import numpy as np
import autograd.numpy as anp
from autograd import hessian, value_and_grad, elementwise_grad as egrad, grad
from autograd.differential_operators import make_jvp_reversemode
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from numpy.linalg import inv, pinv


from lifelines.plotting import _plot_estimate, set_kwargs_drawstyle, set_kwargs_ax
from lifelines.fitters import BaseFitter

from lifelines.utils import (
    qth_survival_times,
    _to_array,
    _to_list,
    safe_zip,
    dataframe_interpolate_at_times,
    ConvergenceError,
    inv_normal_cdf,
    string_justify,
    format_floats,
    format_p_value,
    format_exp_floats,
    coalesce,
    check_nans_or_infs,
    pass_for_numeric_dtypes_or_raise_array,
    check_for_numeric_dtypes_or_raise,
    check_complete_separation,
    check_low_var,
    check_positivity,
    StatisticalWarning,
    StatError,
    median_survival_times,
    normalize,
    concordance_index,
    CensoringType,
)


class DataframeSliceDict:
    def __init__(self, df, mappings):
        self.df = df
        self.mappings = mappings
        self.size = sum(len(v) for v in self.mappings.values())

    def __getitem__(self, key):
        return self.df[self.mappings[key]].values

    def __iter__(self):
        for k in self.mappings:
            yield (k, self[k])

    def filter(self, ix):
        return DataframeSliceDict(self.df.loc[ix], self.mappings)


class ParametericRegressionFitter(BaseFitter):
    def __init__(self, alpha=0.05, penalizer=0.0):
        super(ParametericRegressionFitter, self).__init__(alpha=alpha)
        self._hazard = egrad(self._cumulative_hazard, argnum=1)  # pylint: disable=unexpected-keyword-arg
        self.penalizer = penalizer

    def _check_values(self, df, T, E, weights, entries):
        check_for_numeric_dtypes_or_raise(df)
        check_nans_or_infs(df)
        check_nans_or_infs(T)
        check_nans_or_infs(E)
        check_positivity(T)
        check_complete_separation(df, E, T, self.event_col)

        if self.weights_col:
            if (weights.astype(int) != weights).any() and not self.robust:
                warnings.warn(
                    dedent(
                        """It appears your weights are not integers, possibly propensity or sampling scores then?
                                        It's important to know that the naive variance estimates of the coefficients are biased. Instead a) set `robust=True` in the call to `fit`, or b) use Monte Carlo to
                                        estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"""
                    ),
                    StatisticalWarning,
                )
            if (weights <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

        if self.entry_col:
            count_invalid_rows = (entries > T).sum()
            if count_invalid_rows:
                warnings.warn("""There exist %d rows where entry > duration.""")

    def _log_hazard(self, params, T, *Xs):
        # can be overwritten to improve convergence, see WeibullAFTFitter
        hz = self._hazard(params, T, *Xs)
        hz = anp.clip(hz, 1e-20, np.inf)
        return anp.log(hz)

    def _log_1m_sf(self, params, T, *Xs):
        # equal to log(cdf), but often easier to express with sf.
        cum_haz = self._cumulative_hazard(params, T, *Xs)
        return anp.log1p(-anp.exp(-cum_haz))

    def _survival_function(self, params, T, Xs):
        return anp.clip(anp.exp(-self._cumulative_hazard(params, T, Xs)), 1e-25, 1.0)

    def _log_likelihood_right_censoring(self, params, Ts, E, W, entries, Xs):
        warnings.simplefilter(action="ignore", category=FutureWarning)

        T = Ts[0]
        non_zero_entries = entries > 0

        log_hz = self._log_hazard(params, T, Xs)
        cum_hz = self._cumulative_hazard(params, T, Xs)
        delayed_entries = self._cumulative_hazard(params, entries[non_zero_entries], Xs.filter(non_zero_entries))

        ll = 0
        ll = ll + (W * E * log_hz).sum()
        ll = ll + -(W * cum_hz).sum()
        ll = ll + (W[non_zero_entries] * delayed_entries).sum()
        ll = ll / anp.sum(W)
        return ll

    @CensoringType.right_censoring
    def fit(
        self,
        df,
        duration_col,
        event_col=None,
        regressors=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
    ):
        """
        Fit the accelerated failure time model to a right-censored dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        show_progress: boolean, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        regressors: TODO

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: boolean, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        Returns
        -------
        self:
            self with additional new properties: ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        """
        self.duration_col = duration_col
        self._time_cols = [duration_col]

        df = df.copy()

        T = pass_for_numeric_dtypes_or_raise_array(df.pop(duration_col)).astype(float)
        self.durations = T.copy()

        self._fit(
            self._log_likelihood_right_censoring,
            df,
            (T.values, None),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
        )

        return self

    def _create_Xs_dict(self, df):
        return DataframeSliceDict(df, self.regressors)

    def _fit(
        self,
        log_likelihood_function,
        df,
        Ts,
        regressors,
        event_col=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
    ):

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.weights_col = weights_col
        self.entry_col = entry_col
        self.event_col = event_col
        self._n_examples = df.shape[0]
        self.timeline = timeline
        self.robust = robust
        self.regressors = regressors  # TODO name

        E = (
            pass_for_numeric_dtypes_or_raise_array(df.pop(self.event_col))
            if (self.event_col is not None)
            else pd.Series(np.ones(self._n_examples, dtype=bool), index=df.index, name="E")
        )
        weights = (
            pass_for_numeric_dtypes_or_raise_array(df.pop(self.weights_col)).astype(float)
            if (self.weights_col is not None)
            else pd.Series(np.ones(self._n_examples, dtype=float), index=df.index, name="weights")
        )

        entries = (
            pass_for_numeric_dtypes_or_raise_array(df.pop(entry_col)).astype(float)
            if (entry_col is not None)
            else pd.Series(np.zeros(self._n_examples, dtype=float), index=df.index, name="entry")
        )

        check_nans_or_infs(E)
        E = E.astype(bool)
        self.event_observed = E.copy()
        self.entry = entries.copy()
        self.weights = weights.copy()

        df = df.astype(float)
        self._check_values(df, coalesce(Ts[1], Ts[0]), E, weights, entries)
        check_for_numeric_dtypes_or_raise(df)
        check_nans_or_infs(df)

        _norm_std = df.std(0)
        _norm_std[_norm_std < 1e-8] = 1.0
        df_normalized = normalize(df, 0, _norm_std)

        Xs = self._create_Xs_dict(df_normalized)

        self._LOOKUP_SLICE = self._create_slicer(Xs)

        _index = pd.MultiIndex.from_tuples(
            sum(([(name, col) for col in columns] for name, columns in regressors.items()), [])
        )

        self._norm_std = pd.Series([_norm_std.loc[variable_name] for _, variable_name in _index], index=_index)

        _params, self._log_likelihood, self._hessian_ = self._fit_model(
            log_likelihood_function,
            Ts,
            Xs,
            E.values,
            weights.values,
            entries.values,
            show_progress=show_progress,
            initial_point=initial_point,
        )
        self.params_ = _params / self._norm_std

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors(Ts, E.values, weights.values, entries.values, Xs)
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self._predicted_median = self.predict_median(df)

    def _create_initial_point(self, Xs):
        return np.zeros(Xs.size)

    def _add_penalty(self, ll, *args):
        return ll

    def _wrap_ll(self, ll):
        def f(params_array, *args):
            params_dict = {
                parameter_name: params_array[self._LOOKUP_SLICE[parameter_name]]
                for parameter_name in self._fitted_parameter_names
            }
            return ll(params_dict, *args)

        return f

    def _fit_model(self, likelihood, Ts, Xs, E, weights, entries, show_progress=False, initial_point=None):

        if initial_point is None:
            initial_point = self._create_initial_point(Xs)

        assert initial_point.shape[0] == Xs.size, "initial_point is not the correct shape."

        self._neg_likelihood_with_penalty_function = lambda *args: self._add_penalty(
            -self._wrap_ll(likelihood)(*args), *args
        )

        results = minimize(
            # using value_and_grad is much faster (takes advantage of shared computations) than splitting.
            value_and_grad(self._neg_likelihood_with_penalty_function),
            initial_point,
            method=None,
            jac=True,
            args=(Ts, E, weights, entries, Xs),
            options={"disp": show_progress},
        )
        if show_progress or not results.success:
            print(results)

        if results.success:
            sum_weights = weights.sum()
            # pylint: disable=no-value-for-parameter
            hessian_ = hessian(self._neg_likelihood_with_penalty_function)(results.x, Ts, E, weights, entries, Xs)
            return results.x, -sum_weights * results.fun, sum_weights * hessian_

        name = self._class_name
        raise ConvergenceError(
            dedent(
                """\
            Fitting did not converge. This could be a problem with your dataset:

            0. Are there any lifelines warnings outputted during the `fit`?
            1. Inspect your DataFrame: does everything look as expected?
            2. Is there high-collinearity in the dataset? Try using the variance inflation factor (VIF) to find redundant variables.
            3. Trying adding a small penalizer (or changing it, if already present). Example: `%s(penalizer=0.01).fit(...)`.
            4. Are there any extreme outliers? Try modeling them or dropping them to see if it helps convergence.
        """
                % name
            )
        )

    def _create_slicer(self, Xs):
        lookup = {}
        position = 0

        for name, X in Xs:
            size = X.shape[1]
            lookup[name] = slice(position, position + size)
            position += size

        return lookup

    def _compute_variance_matrix(self):
        try:
            unit_scaled_variance_matrix_ = np.linalg.inv(self._hessian_)
        except np.linalg.LinAlgError:
            unit_scaled_variance_matrix_ = np.linalg.pinv(self._hessian_)
            warning_text = dedent(
                """\
                The Hessian was not invertible. We will instead approximate it using the pseudo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the
                fitted parameters too.

                Some ways to possible ways fix this:

                0. Are there any lifelines warnings outputted during the `fit`?
                1. Inspect your DataFrame: does everything look as expected?
                2. Is there high-collinearity in the dataset? Try using the variance inflation factor (VIF) to find redundant variables.
                3. Trying adding a small penalizer (or changing it, if already present). Example: `%s(penalizer=0.01).fit(...)`.
                4. Are there any extreme outliers? Try modeling them or dropping them to see if it helps convergence.
                """
                % self._class_name
            )
            warnings.warn(warning_text, StatisticalWarning)

        return unit_scaled_variance_matrix_ / np.outer(self._norm_std, self._norm_std)

    def _compute_z_values(self):
        return self.params_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_standard_errors(self, Ts, E, weights, entries, Xs):
        if self.robust:
            se = np.sqrt(self._compute_sandwich_errors(Ts, E, weights, entries, Xs).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.Series(se, name="se", index=self.params_.index)

    def _compute_sandwich_errors(self, Ts, E, weights, entries, Xs):

        with np.errstate(all="ignore"):
            # convergence will fail catastrophically elsewhere.

            ll_gradient = grad(self._neg_likelihood_with_penalty_function)
            params = self.params_.values
            n_params = params.shape[0]
            J = np.zeros((n_params, n_params))

            for ts, e, w, s, xs in zip(safe_zip(*Ts), E, weights, entries, zip(*Xs)):
                score_vector = ll_gradient(params, ts, e, w, s, xs)
                J += np.outer(score_vector, score_vector)

            return self.variance_matrix_ @ J @ self.variance_matrix_

    def _compute_confidence_intervals(self):
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        params = self.params_.values
        return pd.DataFrame(
            np.c_[params - z * se, params + z * se], index=self.params_.index, columns=["lower-bound", "upper-bound"]
        )

    @property
    def _ll_null(self):
        if hasattr(self, "_ll_null_"):
            return self._ll_null_

        initial_point = np.zeros(len(self._fitted_parameter_names))
        regressors = {name: ["intercept"] for name in self._fitted_parameter_names}

        model = self.__class__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if CensoringType.is_right_censoring(self):
                df = pd.DataFrame({"T": self.durations, "E": self.event_observed, "entry": self.entry, "intercept": 1})
                model.fit_right_censoring(
                    df, "T", "E", initial_point=initial_point, entry_col="entry", regressors=regressors
                )
            elif CensoringType.is_interval_censoring(self):
                df = pd.DataFrame(
                    {
                        "lb": self.lower_bound,
                        "ub": self.upper_bound,
                        "E": self.event_observed,
                        "entry": self.entry,
                        "intercept": 1,
                    }
                )
                model.fit_interval_censoring(
                    df, "lb", "ub", "E", initial_point=initial_point, entry_col="entry", regressors=regressors
                )
            if CensoringType.is_left_censoring(self):
                raise NotImplementedError()

        self._ll_null_ = model._log_likelihood
        return self._ll_null_

    def log_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.
        """
        from lifelines.statistics import chisq_test, StatisticalResult

        ll_null = self._ll_null
        ll_alt = self._log_likelihood

        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.params_.shape[0] - 2  # delta in number of parameters between models
        p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value,
            test_stat,
            name="log-likelihood ratio test",
            degrees_freedom=degrees_freedom,
            null_distribution="chi squared",
        )

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper
        """

        ci = 1 - self.alpha
        with np.errstate(invalid="ignore", divide="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = np.exp(self.params_)
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
        alpha: float or iterable
            specify confidence intervals to show
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        # Print information about data first
        justify = string_justify(18)
        print(self)
        if self.event_col:
            print("{} = '{}'".format(justify("event col"), self.event_col))
        if self.weights_col:
            print("{} = '{}'".format(justify("weights col"), self.weights_col))
        if self.penalizer > 0:
            print("{} = {}".format(justify("penalizer"), self.penalizer))

        if self.robust:
            print("{} = {}".format(justify("robust variance"), True))

        print("{} = {}".format(justify("number of subjects"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.{prec}f}".format(justify("log-likelihood"), self._log_likelihood, prec=decimals))
        print("{} = {}".format(justify("time fit was run"), self._time_fit_was_called))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(
            df.to_string(
                float_format=format_floats(decimals),
                formatters={"p": format_p_value(decimals), "exp(coef)": format_exp_floats(decimals)},
            )
        )

        print("---")
        if CensoringType.is_right_censoring(self):
            print("Concordance = {:.{prec}f}".format(self.score_, prec=decimals))

        with np.errstate(invalid="ignore", divide="ignore"):
            sr = self.log_likelihood_ratio_test()
            print(
                "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                    sr.test_statistic, sr.degrees_freedom, -np.log2(sr.p_value), prec=decimals
                )
            )

    def predict_survival_function(self, df, times=None):
        """
        Predict the survival function for individuals, given their covariates. This assumes that the individual
        just entered the study (that is, we do not condition on how long they have already lived for.)

        Parameters
        ----------

        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_X: numpy array or DataFrame, optional
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
        return np.exp(-self.predict_cumulative_hazard(df, times=times))

    def predict_median(self, df):
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
        predict_percentile, predict_expectation

        """
        return self.predict_percentile(df, p=0.5)

    def predict_percentile(self, df, p=0.5):
        return qth_survival_times(p, self.predict_survival_function(df))

    def predict_cumulative_hazard(self, df, times=None):

        times = coalesce(times, self.timeline, np.unique(self.durations))
        n = df.shape[0]
        Xs = self._create_Xs_dict(df)

        params_dict = {
            parameter_name: self.params_.values[self._LOOKUP_SLICE[parameter_name]]
            for parameter_name in self._fitted_parameter_names
        }

        return pd.DataFrame(
            self._cumulative_hazard(params_dict, np.tile(times, (n, 1)).T, Xs), index=times, columns=df.index
        )

    @property
    def score_(self):
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the ROC AUC
        to survival data, including censorships.

        For this purpose, the ``score_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset.

        """
        # pylint: disable=access-member-before-definition
        if hasattr(self, "_predicted_median"):
            self._concordance_score_ = concordance_index(self.durations, self._predicted_median, self.event_observed)
            del self._predicted_median
            return self._concordance_score_
        return self._concordance_score_

    @property
    def median_survival_time_(self):
        return self.predict_median(self._norm_mean.to_frame().T).squeeze()

    @property
    def mean_survival_time_(self):
        return self.predict_expectation(self._norm_mean.to_frame().T).squeeze()

    def plot(self, columns=None, parameter=None, **errorbar_kwargs):
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

        params_ = self.params_.copy()
        standard_errors_ = self.standard_errors_.copy()
        user_supplied_columns = False

        if columns is not None:
            params_ = params_.loc[:, columns]
            standard_errors_ = standard_errors_.loc[:, columns]
            user_supplied_columns = True
        if parameter is not None:
            params_ = params_.loc[parameter]
            standard_errors_ = standard_errors_.loc[parameter]

        columns = params_.index

        hazards = params_.loc[columns].to_frame(name="coefs")
        hazards["se"] = z * standard_errors_.loc[columns]

        if not user_supplied_columns:
            if isinstance(hazards.index, pd.MultiIndex):
                hazards = hazards.groupby(level=0, group_keys=False).apply(
                    lambda x: x.sort_values(by="coefs", ascending=True)
                )
            else:
                hazards = hazards.sort_values(by="coefs", ascending=True)

        yaxis_locations = list(range(len(columns)))

        ax.errorbar(hazards["coefs"], yaxis_locations, xerr=hazards["se"], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
        ax.set_ylim(best_ylim)

        if isinstance(columns[0], tuple):
            tick_labels = ["%s: %s" % (c, p) for (p, c) in hazards.index]
        else:
            tick_labels = [i for i in hazards.index]

        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("log(accelerated failure rate) (%g%% CI)" % ((1 - self.alpha) * 100))

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
            a string (or list of strings) of the covariate in the original dataset that we wish to vary.
        values: 1d or 2d iterable
            an iterable of the values we wish the covariate to take on.
        plot_baseline: bool
            also display the baseline survival, defined as the survival at the mean of the original dataset.
        kwargs:
            pass in additional plotting commands

        Returns
        -------
        ax: matplotlib axis, or list of axis'
            the matplotlib axis that be edited.

        Examples
        ---------

        >>> from lifelines import datasets, WeibullAFTFitter
        >>> rossi = datasets.load_rossi()
        >>> wf = WeibullAFTFitter().fit(rossi, 'week', 'arrest')
        >>> wf.plot_covariate_groups('prio', values=np.arange(0, 15), cmap='coolwarm')

        >>> # multiple variables at once
        >>> wf.plot_covariate_groups(['prio', 'paro'], values=[[0, 0], [5, 0], [10, 0], [0, 1], [5, 1], [10, 1]], cmap='coolwarm')

        >>> # if you have categorical variables, you can simply things:
        >>> wf.plot_covariate_groups(['dummy1', 'dummy2', 'dummy3'], values=np.eye(3))


        """
        from matplotlib import pyplot as plt

        covariates = _to_list(covariates)
        values = _to_array(values)
        if len(values.shape) == 1:
            values = values[None, :].T

        if len(covariates) != values.shape[1]:
            raise ValueError("The number of covariates must equal to second dimension of the values array.")

        original_columns = self.params_.index.get_level_values(1)
        for covariate in covariates:
            if covariate not in original_columns:
                raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        ax = kwargs.pop("ax", None) or plt.figure().add_subplot(111)

        # model X
        x_bar = self._norm_mean.to_frame().T
        X = pd.concat([x_bar] * values.shape[0])
        if np.array_equal(np.eye(len(covariates)), values):
            X.index = ["%s=1" % c for c in covariates]
        else:
            X.index = [", ".join("%s=%g" % (c, v) for (c, v) in zip(covariates, row)) for row in values]
        for covariate, value in zip(covariates, values.T):
            X[covariate] = value

        # model ancillary X
        x_bar_anc = self._norm_mean_ancillary.to_frame().T
        ancillary_X = pd.concat([x_bar_anc] * values.shape[0])
        for covariate, value in zip(covariates, values.T):
            ancillary_X[covariate] = value

        if self.fit_intercept:
            X["_intercept"] = 1.0
            ancillary_X["_intercept"] = 1.0

        self.predict_survival_function(X, ancillary_X=ancillary_X).plot(ax=ax, **kwargs)
        if plot_baseline:
            self.predict_survival_function(x_bar, ancillary_X=x_bar_anc).rename(columns={0: "baseline survival"}).plot(
                ax=ax, ls=":", color="k"
            )
        return ax


class Weibull(ParametericRegressionFitter):

    # TODO: check if _fitted_parameter_names is equal to keys in regressors!!
    _fitted_parameter_names = ["lambda_", "rho_"]

    def _cumulative_hazard(self, params, T, Xs):
        """
        One problem with this is that the dict Xs is going to be very memory consuming. Maybe I can create a new
        "dict" that has shared df and dynamically creates the df on indexing with [], that's cool

        """
        lambda_ = anp.exp(anp.dot(Xs["lambda_"], params["lambda_"]))  # > 0
        rho_ = anp.exp(anp.dot(Xs["rho_"], params["rho_"]))  # > 0

        return ((T) / lambda_) ** rho_


from lifelines.datasets import load_rossi

swf = Weibull(penalizer=1.0)
rossi = load_rossi()
rossi["intercept"] = 1.0

covariates = {
    "lambda_": ["intercept", "fin", "age", "race", "wexp", "mar", "paro", "prio"],  # need a shortcut for all columns?
    "rho_": ["intercept"],
}

swf.fit(rossi, "week", event_col="arrest", regressors=covariates)  # TODO: name
swf.print_summary()
