# -*- coding: utf-8 -*-
import autograd.numpy as np
from lifelines.utils import coalesce, _get_index, CensoringType
from lifelines.fitters import ParametricRegressionFitter
import pandas as pd
from lifelines.utils.safe_exp import safe_exp


class PiecewiseExponentialRegressionFitter(ParametricRegressionFitter):
    """
    TODO: docs



    Examples
    ----------

    See blog post <here https://dataorigami.net/blogs/napkin-folding/churn>_.


    """

    # about 50% faster than BFGS
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-6, "maxiter": 200}

    def __init__(self, breakpoints, alpha=0.05, penalizer=0.0):
        super(PiecewiseExponentialRegressionFitter, self).__init__(alpha=alpha)

        breakpoints = np.sort(breakpoints)
        if len(breakpoints) and not (breakpoints[-1] < np.inf):
            raise ValueError("Do not add inf to the breakpoints.")

        if len(breakpoints) and breakpoints[0] < 0:
            raise ValueError("First breakpoint must be greater than 0.")

        self.breakpoints = np.append(breakpoints, [np.inf])
        self.n_breakpoints = len(self.breakpoints)

        self.penalizer = penalizer
        self._fitted_parameter_names = ["lambda_%d_" % i for i in range(self.n_breakpoints)]

    def _add_penalty(self, params, neg_ll):
        params_stacked = np.stack(params.values())
        coef_penalty = 0

        if self.penalizer > 0:
            for i in range(params_stacked.shape[1]):
                if not self._constant_cols[i]:
                    coef_penalty = coef_penalty + (params_stacked[:, i]).var()

        return neg_ll + self.penalizer * coef_penalty

    def _cumulative_hazard(self, params, T, Xs):
        n = T.shape[0]
        T = T.reshape((n, 1))
        M = np.minimum(np.tile(self.breakpoints, (n, 1)), T)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])
        lambdas_ = np.array([safe_exp(-np.dot(Xs[param], params[param])) for param in self._fitted_parameter_names])
        return (M * lambdas_.T).sum(1)

    def _log_hazard(self, params, T, X):
        hz = self._hazard(params, T, X)
        hz = np.clip(hz, 1e-20, np.inf)
        return np.log(hz)

    def _prep_inputs_for_prediction_and_return_parameters(self, X):
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X = X[self.params_["lambda_0_"].index]

        return np.array([np.exp(np.dot(X, self.params_["lambda_%d_" % i])) for i in range(self.n_breakpoints)])

    def predict_cumulative_hazard(self, df, times=None, conditional_after=None):
        """
        Return the cumulative hazard rate of subjects in X at time points.

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

        if isinstance(df, pd.Series):
            return self.predict_cumulative_hazard(df.to_frame().T)

        if conditional_after is not None:
            raise NotImplementedError()

        times = np.atleast_1d(coalesce(times, self.timeline, np.unique(self.durations))).astype(float)
        n = times.shape[0]
        times = times.reshape((n, 1))

        lambdas_ = self._prep_inputs_for_prediction_and_return_parameters(df)

        bp = self.breakpoints
        M = np.minimum(np.tile(bp, (n, 1)), times)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])

        return pd.DataFrame(np.dot(M, (1 / lambdas_)), columns=_get_index(df), index=times[:, 0])

    @property
    def _ll_null(self):
        if hasattr(self, "_ll_null_"):
            return self._ll_null_

        initial_point = np.zeros(len(self._fitted_parameter_names))

        model = self.__class__(breakpoints=self.breakpoints[:-1], penalizer=self.penalizer)
        regressors = {param_name: ["_intercept"] for param_name in self._fitted_parameter_names}
        if CensoringType.is_right_censoring(self):
            df = pd.DataFrame({"T": self.durations, "E": self.event_observed, "entry": self.entry, "_intercept": 1.0})
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
                    "_intercept": 1.0,
                }
            )
            model.fit_interval_censoring(
                df, "lb", "ub", "E", initial_point=initial_point, entry_col="entry", regressors=regressors
            )
        if CensoringType.is_left_censoring(self):
            raise NotImplementedError()

        self._ll_null_ = model.log_likelihood_
        return self._ll_null_
