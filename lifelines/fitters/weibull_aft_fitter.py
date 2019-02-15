
from autograd import numpy as np
from autograd import value_and_grad, elementwise_grad as egrad, jacobian, hessian
from scipy.optimize import minimize
from lifelines.datasets import load_regression_dataset
import pandas as pd
from datetime import datetime
from lifelines.fitters import BaseFitter
from lifelines import WeibullFitter

from scipy import stats
from lifelines.utils import (_get_index, qth_survival_times, concordance_index, StatisticalWarning, inv_normal_cdf, format_floats, format_p_value, string_justify,
    pass_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation,
    check_nans_or_infs,
)

from lifelines.statistics import chisq_test


class WeibullAFTFitter(BaseFitter):
    r"""
    This class implements a Weibull model for univariate data. The model has parameterized
    form, with :math:`\lambda = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and optionally, `\rho = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    .. math::  S(t) = exp(-(t / \lambda)^\rho),

    which implies the cumulative hazard rate is

    .. math:: H(t) = (t / \lambda)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \frac{\rho}{\lambda}(t/\lambda)^{\rho-1}

    After calling the `.fit` method, you have access to properties like:
    ``params_``, ``print_summary()``.

    A summary of the fit is available with the method 'print_summary()'
    """

    def __init__(self, fit_intercept=True, alpha=0.95, penalizer=0.0):
        super(WeibullAFTFitter, self).__init__(alpha=alpha)
        self._fitted_parameter_names = ['lambda_', 'rho_']
        self._hazard = egrad(self._cumulative_hazard, argnum=1) # diff w.r.t. time
        self.penalizer = penalizer


    def _negative_log_likelihood(self, params, T, E, *Xs):
        n = T.shape[0]

        hz = self._hazard(params, T, *Xs)
        hz = np.clip(hz, 1e-18, np.inf)

        ll = (
            (E * np.log(hz)).sum()
            - self._cumulative_hazard(params, T, *Xs).sum()
        )
        return -ll / n + self.penalizer * (params**2).sum()


    def _cumulative_hazard(self, params, T, *Xs):
        lambda_params = params[self._LOOKUP_SLICE['lambda_']]
        lambda_ =  np.exp(np.dot(Xs[0], lambda_params))

        rho_params = params[self._LOOKUP_SLICE['rho_']]
        rho_ =  np.exp(np.dot(Xs[1], rho_params))

        return (T / lambda_) ** rho_

    def fit(
        self,
        df,
        duration_col,
        event_col=None,
        ancillary_df=None, # (or None, or True, or False - same as None)
        show_progress=False
    ):

        if duration_col is None:
            raise TypeError("duration_col cannot be None.")

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self._n_examples = df.shape[0]


        df = df.copy()

        T = df.pop(duration_col).astype(float)
        E = df.pop(event_col).astype(bool)


        self.durations = T.copy()
        self.event_observed = E.copy()

        if isinstance(ancillary_df, pd.DataFrame):
            assert ancillary_df.shape[0] == df.shape[0], 'ancillary_df must be the same shape[0] as df'
            ancillary_df = ancillary_df.copy()
            ancillary_df = ancillary_df.drop([duration_col, event_col], axis=1, errors='ignore')
            self._check_values(ancillary_df, T, E, self.event_col)
        elif (ancillary_df is None) or (ancillary_df == False):
            ancillary_df = pd.DataFrame(index=df.index)
        elif ancillary_df == True:
            ancillary_df = df.copy()

        self._check_values(df, T, E, self.event_col)

        assert ('_intercept' not in ancillary_df) and ('_intercept' not in df)

        ancillary_df['_intercept'], df['_intercept'] = 1.0, 1.0

        self._LOOKUP_SLICE = self._create_slicer(
            len(df.columns),
            len(ancillary_df.columns)
        )

        _params, self._log_likelihood, self._hessian_ = self._fit_model(T.values, E.values, df.values, ancillary_df.values, show_progress=show_progress)

        self.params_ = pd.Series(_params, index=pd.MultiIndex.from_tuples(
            [('lamba_', c) for c in df.columns] +
            [('rho_', c) for c in ancillary_df.columns]
        ), name='coef')

        try:
            self.variance_matrix_ = inv(self._hessian_)
        except np.linalg.LinAlgError:
            self.variance_matrix_ = pinv(self._hessian_)
            warning_text = dedent(
                """\
                The hessian was not invertable. We will instead approximate it using the psuedo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the
                fitted parameters too. Perform plots of the cumulative hazard to help understand
                the latter's bias.
                """
            )
            warnings.warn(warning_text, StatisticalWarning)

        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()

        return self

    @staticmethod
    def _check_values(X, T, E, event_col):
        pass_for_numeric_dtypes_or_raise(X)
        check_nans_or_infs(T)
        check_nans_or_infs(E)
        check_nans_or_infs(X)
        check_low_var(X)
        check_complete_separation(X, E, T, event_col)

    def _fit_model(self, T, E, *Xs, show_progress=False):

        n_params = sum([X.shape[1] for X in Xs])
        init_values = np.zeros((n_params,))

        results = minimize(
            value_and_grad(self._negative_log_likelihood),
            init_values,
            jac=True,
            args=(T, E, *Xs),
            options={"disp": show_progress},
        )

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

    def _compute_z_values(self):
        return self.params_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_standard_errors(self):
        se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.Series(se, name="se", index=self.params_.index)

    def _compute_confidence_intervals(self):
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        se = self.standard_errors_
        params = self.params_.values
        return pd.DataFrame(
            np.c_[params - alpha2 * se, params + alpha2 * se],
            index=self.params_.index,
            columns=["lower-bound", "upper-bound"],
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
        degrees_freedom = self.params_.shape[0] - 2 # diff in number of parameters between models
        p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom)
        with np.errstate(invalid="ignore", divide="ignore"):
            return test_stat, degrees_freedom, -np.log2(p_value)


    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""

        with np.errstate(invalid="ignore", divide="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = np.exp(self.params_)
            df["se(coef)"] = self.standard_errors_
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -np.log2(df["p"])
            df["lower %.2f" % self.alpha] = self.confidence_intervals_["lower-bound"]
            df["upper %.2f" % self.alpha] = self.confidence_intervals_["upper-bound"]
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
        #print("Concordance = {:.{prec}f}".format(self.score_, prec=decimals))
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

    def predict_percentile(self, X, p=0.5, ancillary_X=None):
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
        subjects = _get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X, ancillary_X=ancillary_X)[subjects]).T

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
        predict_percentile

        """
        return self.predict_percentile(X, 0.5, ancillary_X=ancillary_X)

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
        timeline = np.linspace(0, 25)

        if ancillary_X is None:
            ancillary_X = np.ones((X.shape[0],1))
        elif isinstance(ancillary_X, pd.DataFrame):
            ancillary_X['_intercept'] = 1.0
            ancillary_X = ancillary_X[wf.params.loc['rho_'].index]
        else:
            assert ancillary_X.shape[1] == (wf.params.loc['rho_'].shape[0] + 1) # 1 for _intercept

        if isinstance(X, pd.DataFrame):
            X['_intercept'] = 1.0
            X = X[wf.params.loc['lambda_'].index]
        else:
            assert X.shape[1] == (wf.params.loc['lambda_'].shape[0] + 1) # 1 for _intercept


        cols = _get_index(X)

        lambda_params = self.params[self._LOOKUP_SLICE['lambda_']]
        lambda_ =  np.exp(np.dot(X, lambda_params))

        rho_params = self.params[self._LOOKUP_SLICE['rho_']]
        rho_ =  np.exp(np.dot(ancillary_X, rho_params))

        return pd.DataFrame(np.outer(timeline, lambda_) ** rho_, columns=cols, index=timeline)



df = load_regression_dataset()

wf = WeibullFitter().fit(df['T'], df['E'])

aft = WeibullAFTFitter().fit(df, 'T', 'E', ancillary_df=df)
