# -*- coding: utf-8 -*-
import warnings
from textwrap import dedent
from datetime import datetime

from autograd import numpy as np
from autograd import value_and_grad, elementwise_grad as egrad, hessian
from scipy.optimize import minimize
from scipy import stats
from scipy.special import gamma
import pandas as pd

from lifelines.fitters import BaseFitter
from lifelines import WeibullFitter
from lifelines.utils import (
    _get_index,
    concordance_index,
    StatisticalWarning,
    inv_normal_cdf,
    format_floats,
    format_p_value,
    string_justify,
    pass_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation,
    check_nans_or_infs,
    normalize,
    ConvergenceError,
)
from lifelines.statistics import chisq_test


class WeibullAFTFitter(BaseFitter):
    r"""
    This class implements a Weibull model for univariate data. The model has parameterized
    form, with :math:`\lambda = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and optionally, `\rho = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    .. math::  S(t; x) = \exp(-(t / \lambda(x))^\rho(y)),

    which implies the cumulative hazard rate is

    .. math:: H(t) = \left(\frac{t}{\lambda(x)}\right)^\rho(y),

    and the hazard rate is:

    .. math::  h(t) = \frac{\rho}{\lambda}(t/\lambda)^{\rho-1}

    After calling the `.fit` method, you have access to properties like:
    ``params_``, ``print_summary()``. A summary of the fit is available with the method ``print_summary()``.


    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    fit_intercept: boolean, optional (default=True)
        Allow lifelines to add an intercept column of 1s to df, and ancillary_df if applicable.

    penalizer: float, optional (default=0.0)
        the penalizer coefficient to the size of the coefficients. See `l1_ratio`. Must be equal to or greater than 0.

    l1_ratio: float, optional (default=0.0)
        how much of the penalizer should be attributed to an l1 penality (otherwise an l2 penalty). The penalty function looks like
        ``penalizer * l1_ratio * ||w||_1 + 0.5 * penalizer * (1 - l1_ratio) * ||w||^2_2``
    """

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio=0.0, fit_intercept=True):
        super(WeibullAFTFitter, self).__init__(alpha=alpha)
        self._fitted_parameter_names = ["lambda_", "rho_"]
        self._hazard = egrad(self._cumulative_hazard, argnum=1)  # pylint: disable=unexpected-keyword-arg
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept

    def _negative_log_likelihood(self, params, T, E, *Xs):
        n = T.shape[0]

        hz = self._hazard(params, T, *Xs)
        hz = np.clip(hz, 1e-18, np.inf)

        ll = (E * np.log(hz)).sum() - self._cumulative_hazard(params, T, *Xs).sum()
        if self.penalizer > 0:
            penalty = self.l1_ratio * np.abs(params).sum() + 0.5 * (1.0 - self.l1_ratio) * (params ** 2).sum()
        else:
            penalty = 0

        ll = ll / n
        return -ll + self.penalizer * penalty

    def _cumulative_hazard(self, params, T, *Xs):
        lambda_params = params[self._LOOKUP_SLICE["lambda_"]]
        lambda_ = np.exp(np.dot(Xs[0], lambda_params))

        rho_params = params[self._LOOKUP_SLICE["rho_"]]
        rho_ = np.exp(np.dot(Xs[1], rho_params))

        return (T / lambda_) ** rho_

    def fit(self, df, duration_col=None, event_col=None, ancillary_df=None, show_progress=False):
        """
        Fit the Weibull accelerated failure time model to a dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
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

        show_progress: boolean, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        ancillary_df: None, boolean, or DataFrame, optional (default=None)
            Choose to model the ancillary parameters.
            If None or False, explicity do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.

        Returns
        -------
        self: WeibullAFTFitter
            self with additional new properties: ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        Examples
        --------
        >>> from lifelines import WeibullAFTFitter
        >>>
        >>> df = pd.DataFrame({
        >>>     'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        >>>     'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        >>>     'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>> })
        >>>
        >>> aft = WeibullAFTFitter()
        >>> aft.fit(df, 'T', 'E')
        >>> aft.print_summary()
        >>> aft.predict_median(df)
        >>>
        >>> aft = WeibullAFTFitter()
        >>> aft.fit(df, 'T', 'E', ancillary_df=df)
        >>> aft.print_summary()
        >>> aft.predict_median(df)

        """
        if duration_col is None:
            raise TypeError("duration_col cannot be None.")

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self._n_examples = df.shape[0]

        df = df.copy()

        T = df.pop(duration_col).astype(float)
        E = (
            df.pop(self.event_col).astype(bool)
            if (self.event_col is not None)
            else pd.Series(np.ones(self._n_examples), index=df.index, name="E")
        )
        self.durations = T.copy()
        self.event_observed = E.copy()

        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        self._check_values(df, T, E, self.event_col)

        if isinstance(ancillary_df, pd.DataFrame):
            assert ancillary_df.shape[0] == df.shape[0], "ancillary_df must be the same shape[0] as df"

            ancillary_df = ancillary_df.copy().drop([duration_col, event_col], axis=1, errors="ignore")
            self._check_values(ancillary_df, T, E, self.event_col)

        elif (ancillary_df is None) or (ancillary_df is False):
            ancillary_df = pd.DataFrame(index=df.index)
        elif ancillary_df is True:
            ancillary_df = df.copy()

        if self.fit_intercept:
            assert "_intercept" not in ancillary_df
            assert "_intercept" not in df
            ancillary_df["_intercept"] = 1.0
            df["_intercept"] = 1.0

        self._LOOKUP_SLICE = self._create_slicer(len(df.columns), len(ancillary_df.columns))

        _norm_std, _norm_std_ancillary = df.std(0), ancillary_df.std(0)
        # if we included an intercept, we need to fix not divide by zero.
        if self.fit_intercept:
            _norm_std["_intercept"] = 1.0
            _norm_std_ancillary["_intercept"] = 1.0
        else:
            _norm_std[_norm_std < 1e-8] = 1.0
            _norm_std_ancillary[_norm_std_ancillary < 1e-8] = 1.0

        _index = pd.MultiIndex.from_tuples(
            [("lambda_", c) for c in df.columns] + [("rho_", c) for c in ancillary_df.columns]
        )

        self._norm_std = pd.Series(np.append(_norm_std, _norm_std_ancillary), index=_index)

        _params, self._log_likelihood, self._hessian_ = self._fit_model(
            T.values,
            E.values,
            normalize(df, 0, _norm_std).values,
            normalize(ancillary_df, 0, _norm_std_ancillary).values,
            show_progress=show_progress,
        )
        self.params_ = _params / self._norm_std

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self._predicted_median = self.predict_median(df, ancillary_df)

        return self

    def _check_values(self, X, T, E, event_col):
        pass_for_numeric_dtypes_or_raise(X)
        check_nans_or_infs(T)
        check_nans_or_infs(E)
        check_nans_or_infs(X)
        check_complete_separation(X, E, T, event_col)
        if not self.fit_intercept:
            check_low_var(X)

    def _fit_model(self, T, E, *Xs, show_progress=False):
        n_params = sum([X.shape[1] for X in Xs])
        init_values = np.zeros((n_params,))

        results = minimize(
            value_and_grad(self._negative_log_likelihood),
            init_values,
            method=None if self.l1_ratio <= 0.0 else "L-BFGS-B",
            jac=True,
            args=(T, E, *Xs),
            options={"disp": show_progress},
        )
        if show_progress:
            print(results)

        if results.success:
            # pylint: disable=no-value-for-parameter
            hessian_ = hessian(self._negative_log_likelihood)(results.x, T, E, *Xs)
            return results.x, -self._n_examples * results.fun, self._n_examples * hessian_
        print(results)
        raise ConvergenceError(
            dedent(
                """\
            Fitting did not converge. This could be a problem with your data:
            1. Are there any extreme values? (Try modelling them or dropping them to see if it helps convergence)
        """
            )
        )

    def _create_slicer(self, *sizes):

        lookup = {}
        position = 0

        for name, size in zip(self._fitted_parameter_names, sizes):
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
                The hessian was not invertable. We will instead approximate it using the psuedo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the
                fitted parameters too. Perform plots of the cumulative hazard to help understand
                the latter's bias.
                """
            )
            warnings.warn(warning_text, StatisticalWarning)

        return unit_scaled_variance_matrix_ / np.outer(self._norm_std, self._norm_std)

    def _compute_z_values(self):
        return self.params_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_standard_errors(self):
        se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.Series(se, name="se", index=self.params_.index)

    def _compute_confidence_intervals(self):
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        params = self.params_.values
        return pd.DataFrame(
            np.c_[params - z * se, params + z * se], index=self.params_.index, columns=["lower-bound", "upper-bound"]
        )

    def _compute_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Weibull model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.

        """
        ll_null = WeibullFitter().fit(self.durations, self.event_observed)._log_likelihood
        ll_alt = self._log_likelihood

        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.params_.shape[0] - 2  # diff in number of parameters between models
        p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom)
        with np.errstate(invalid="ignore", divide="ignore"):
            return test_stat, degrees_freedom, -np.log2(p_value)

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
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
        print("{} = '{}'".format(justify("duration col"), self.duration_col))
        print("{} = '{}'".format(justify("event col"), self.event_col))
        if self.penalizer > 0:
            print("{} = {}".format(justify("penalizer"), self.penalizer))
            print("{} = {}".format(justify("l1_ratio"), self.l1_ratio))

        print("{} = {}".format(justify("number of subjects"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.{prec}f}".format(justify("log-likelihood"), self._log_likelihood, prec=decimals))
        print("{} = {}".format(justify("time fit was run"), self._time_fit_was_called))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        # Significance codes as last column
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))

        # Significance code explanation
        print("---")
        print("Concordance = {:.{prec}f}".format(self.score_, prec=decimals))
        print(
            "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                *self._compute_likelihood_ratio_test(), prec=decimals
            )
        )

    def predict_survival_function(self, X, times=None, ancillary_X=None):
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
        return np.exp(-self.predict_cumulative_hazard(X, times=times, ancillary_X=ancillary_X))

    def predict_percentile(self, X, ancillary_X=None, p=0.5):
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
        ancillary_X: numpy array or DataFrame, optional
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
        X = X.copy()

        if ancillary_X is None:
            ancillary_X = pd.DataFrame(np.ones((X.shape[0], 1)), columns=["_intercept"])
        elif isinstance(ancillary_X, pd.DataFrame):
            ancillary_X = ancillary_X.copy()
            if self.fit_intercept:
                ancillary_X["_intercept"] = 1.0
            ancillary_X = ancillary_X[self.params_.loc["rho_"].index]
        else:
            assert ancillary_X.shape[1] == (self.params_.loc["rho_"].shape[0] + 1)  # 1 for _intercept

        if isinstance(X, pd.DataFrame):
            if self.fit_intercept:
                X["_intercept"] = 1.0
            X = X[self.params_.loc["lambda_"].index]
        else:
            assert X.shape[1] == (self.params_.loc["lambda_"].shape[0] + 1)  # 1 for _intercept

        lambda_params = self.params_[self._LOOKUP_SLICE["lambda_"]]
        lambda_ = np.exp(np.dot(X, lambda_params))

        rho_params = self.params_[self._LOOKUP_SLICE["rho_"]]
        rho_ = np.exp(np.dot(ancillary_X, rho_params))
        subjects = _get_index(X)

        return pd.DataFrame(lambda_ * np.power(-np.log(p), 1 / rho_), index=subjects)

    def predict_expectation(self, X, ancillary_X=None):
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

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

        Returns
        -------
        percentiles: DataFrame
            the median lifetimes for the individuals. If the survival curve of an
            individual does not cross 0.5, then the result is infinity.


        See Also
        --------
        predict_median
        """
        X = X.copy()

        if ancillary_X is None:
            ancillary_X = pd.DataFrame(np.ones((X.shape[0], 1)), columns=["_intercept"])
        elif isinstance(ancillary_X, pd.DataFrame):
            ancillary_X = ancillary_X.copy()
            if self.fit_intercept:
                ancillary_X["_intercept"] = 1.0
            ancillary_X = ancillary_X[self.params_.loc["rho_"].index]
        else:
            assert ancillary_X.shape[1] == (self.params_.loc["rho_"].shape[0] + 1)  # 1 for _intercept

        if isinstance(X, pd.DataFrame):
            if self.fit_intercept:
                X["_intercept"] = 1.0
            X = X[self.params_.loc["lambda_"].index]
        else:
            assert X.shape[1] == (self.params_.loc["lambda_"].shape[0] + 1)  # 1 for _intercept

        lambda_params = self.params_[self._LOOKUP_SLICE["lambda_"]]
        lambda_ = np.exp(np.dot(X, lambda_params))

        rho_params = self.params_[self._LOOKUP_SLICE["rho_"]]
        rho_ = np.exp(np.dot(ancillary_X, rho_params))
        subjects = _get_index(X)
        return pd.DataFrame((lambda_ * gamma(1 + 1 / rho_)), index=subjects)

    def predict_median(self, X, ancillary_X=None):
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

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

        Returns
        -------
        percentiles: DataFrame
            the median lifetimes for the individuals. If the survival curve of an
            individual does not cross 0.5, then the result is infinity.


        See Also
        --------
        predict_percentile, predict_expectation

        """
        return self.predict_percentile(X, p=0.5, ancillary_X=ancillary_X)

    def predict_cumulative_hazard(self, X, ancillary_X=None, times=None):
        """
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
        cumulative_hazard_ : DataFrame
            the cumulative hazard of individuals over the timeline
        """
        X = X.copy()
        if times is None:
            times = np.unique(self.durations)

        if ancillary_X is None:
            ancillary_X = pd.DataFrame(np.ones((X.shape[0], 1)), columns=["_intercept"])
        elif isinstance(ancillary_X, pd.DataFrame):
            ancillary_X = ancillary_X.copy()
            if self.fit_intercept:
                ancillary_X["_intercept"] = 1.0
            ancillary_X = ancillary_X[self.params_.loc["rho_"].index]
        else:
            assert ancillary_X.shape[1] == (self.params_.loc["rho_"].shape[0] + 1)  # 1 for _intercept

        if isinstance(X, pd.DataFrame):
            if self.fit_intercept:
                X["_intercept"] = 1.0
            X = X[self.params_.loc["lambda_"].index]
        else:
            assert X.shape[1] == (self.params_.loc["lambda_"].shape[0] + 1)  # 1 for _intercept

        lambda_params = self.params_[self._LOOKUP_SLICE["lambda_"]]
        lambda_ = np.exp(np.dot(X, lambda_params))

        rho_params = self.params_[self._LOOKUP_SLICE["rho_"]]
        rho_ = np.exp(np.dot(ancillary_X, rho_params))
        cols = _get_index(X)
        return pd.DataFrame(np.outer(times, 1 / lambda_) ** rho_, columns=cols, index=times)

    @property
    def score_(self):
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the AUC
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
