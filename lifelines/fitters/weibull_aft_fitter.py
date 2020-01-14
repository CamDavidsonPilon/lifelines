# -*- coding: utf-8 -*-
from autograd import numpy as np
from autograd.builtins import DictBox
from autograd.numpy.numpy_boxes import ArrayBox
from numpy import ndarray
from pandas.core.frame import DataFrame
from typing import Dict, List, Optional, Union
from scipy.special import gamma
import pandas as pd

from lifelines.utils import _get_index
from lifelines.fitters import ParametericAFTRegressionFitter
from lifelines.utils.safe_exp import safe_exp
from lifelines.utils import DataframeSliceDict
from lifelines.statistics import proportional_hazard_test


class WeibullAFTFitter(ParametericAFTRegressionFitter):
    r"""
    This class implements a Weibull AFT model. The model has parameterized
    form, with :math:`\lambda(x) = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and optionally, :math:`\rho(y) = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    .. math::  S(t; x, y) = \exp\left(-\left(\frac{t}{\lambda(x)}\right)^{\rho(y)}\right),

    With no covariates, the Weibull model's parameters has the following interpretations: The :math:`\lambda` (scale) parameter has an
    applicable interpretation: it represent the time when 37% of the population has died.
    The :math:`\rho` (shape) parameter controls if the cumulative hazard (see below) is convex or concave, representing accelerating or decelerating
    hazards.

    The cumulative hazard rate is

    .. math:: H(t; x, y) = \left(\frac{t}{\lambda(x)} \right)^{\rho(y)},

    After calling the ``.fit`` method, you have access to properties like:
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
        how much of the penalizer should be attributed to an l1 penalty (otherwise an l2 penalty). The penalty function looks like
        ``penalizer * l1_ratio * ||w||_1 + 0.5 * penalizer * (1 - l1_ratio) * ||w||^2_2``

    model_ancillary: optional (default=False)
        set the model instance to always model the ancillary parameter with the supplied Dataframe.
        This is useful for grid-search optimization.

    Attributes
    ----------
    params_ : DataFrame
        The estimated coefficients
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the coefficients
    durations: Series
        The event_observed variable provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    standard_errors_: Series
        the standard errors of the estimates
    score_: float
        the concordance index of the model.
    """

    # about 25% faster than BFGS
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-10, "maxiter": 200}

    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        model_ancillary: bool = False,
    ) -> None:
        self._ancillary_parameter_name = "rho_"
        self._primary_parameter_name = "lambda_"
        super(WeibullAFTFitter, self).__init__(alpha, penalizer, l1_ratio, fit_intercept, model_ancillary)

    def _cumulative_hazard(
        self, params: Union[DictBox, Dict[str, ndarray]], T: Union[float, ndarray], Xs: DataframeSliceDict
    ) -> Union[ndarray, ArrayBox]:
        lambda_params = params["lambda_"]
        log_lambda_ = Xs["lambda_"] @ lambda_params

        rho_params = params["rho_"]
        rho_ = safe_exp(Xs["rho_"] @ rho_params)

        return safe_exp(rho_ * (np.log(np.clip(T, 1e-25, np.inf)) - log_lambda_))

    def _log_hazard(
        self, params: Union[DictBox, Dict[str, ndarray]], T: Union[float, ndarray], Xs: DataframeSliceDict
    ) -> Union[ndarray, ArrayBox]:
        lambda_params = params["lambda_"]
        log_lambda_ = Xs["lambda_"] @ lambda_params

        rho_params = params["rho_"]
        log_rho_ = Xs["rho_"] @ rho_params

        return log_rho_ - log_lambda_ + np.expm1(log_rho_) * (np.log(T) - log_lambda_)

    def predict_percentile(
        self,
        df: DataFrame,
        *,
        ancillary_df: Optional[DataFrame] = None,
        p: float = 0.5,
        conditional_after: Optional[ndarray] = None
    ) -> DataFrame:
        """
        Returns the median lifetimes for the individuals, by default. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions

        Parameters
        ----------
        df:  DataFrame
            a (n,d)  DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_df: DataFrame, optional
            a (n,d) DataFrame. If a DataFrame, columns
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

        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary_df)

        if conditional_after is None and len(df.shape) == 2:
            conditional_after = np.zeros(df.shape[0])
        elif conditional_after is None and len(df.shape) == 1:
            conditional_after = np.zeros(1)

        return pd.DataFrame(
            lambda_ * np.power(-np.log(p) + (conditional_after / lambda_) ** rho_, 1 / rho_) - conditional_after,
            index=_get_index(df),
        )

    def predict_expectation(self, df: DataFrame, ancillary_df: Optional[DataFrame] = None) -> DataFrame:
        """
        Predict the expectation of lifetimes, :math:`E[T | x]`.

        Parameters
        ----------
        df: DataFrame
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_df:  DataFrame, optional
            a (n,d) DataFrame. If a DataFrame, columns
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
        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary_df)
        return pd.DataFrame((lambda_ * gamma(1 + 1 / rho_)), index=_get_index(df))

    def compute_residuals(self, training_dataframe: DataFrame, kind: str) -> pd.DataFrame:
        """
        TODO: move me to more general class.

        Parameters
        ----------
        training_dataframe : DataFrame
            the same training DataFrame given in `fit`
        kind : string
            {'schoenfeld', 'scaled_schoenfeld'}

        """
        ALLOWED_RESIDUALS = {"schoenfeld", "scaled_schoenfeld"}
        assert kind in ALLOWED_RESIDUALS, "kind must be in %s" % ALLOWED_RESIDUALS

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X, T, E, weights, shuffled_original_index, _ = self._preprocess_dataframe(training_dataframe)

        resids = getattr(self, "_compute_%s" % kind)(X, T, E, weights, index=shuffled_original_index)
        return resids

    def _compute_schoenfeld(self, X, T, E, weights, index=None):
        pass

    def _compute_scaled_schoenfeld(self, X, T, E, weights, index=None):
        pass

    def check_assumptions(
        self,
        training_df: DataFrame,
        advice: bool = True,
        show_plots: bool = False,
        p_value_threshold: float = 0.01,
        plot_n_bootstraps: int = 10,
        columns: Optional[List[str]] = None,
    ) -> None:
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
        columns: list, optional
            specify a subset of columns to test.


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
        will be below the threshold by chance. This is compounded when there are many covariates.

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

        for variable in self.params_.index.intersection(columns or self.params_.index):
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
                            """   Advice 3: try adding an interaction term with your time variable. See documentation in link [C] below.""",
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
