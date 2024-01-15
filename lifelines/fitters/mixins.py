# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Any, Iterable
from textwrap import dedent, fill
from autograd import numpy as anp
import numpy as np
from pandas import DataFrame, Series
from lifelines.exceptions import ProportionalHazardAssumptionError
from lifelines.statistics import proportional_hazard_test, TimeTransformers
from lifelines.utils import format_p_value
from lifelines.utils.lowess import lowess


class SplineFitterMixin:
    @staticmethod
    def relu(x: np.ndarray):
        return anp.maximum(0, x)

    def basis(self, x: np.ndarray, knot: float, min_knot: float, max_knot: float):
        lambda_ = (max_knot - knot) / (max_knot - min_knot)
        return self.relu(x - knot) ** 3 - (lambda_ * self.relu(x - min_knot) ** 3 + (1 - lambda_) * self.relu(x - max_knot) ** 3)


class ProportionalHazardMixin:
    def check_assumptions(
        self,
        training_df: DataFrame,
        advice: bool = True,
        show_plots: bool = False,
        p_value_threshold: float = 0.01,
        plot_n_bootstraps: int = 15,
        columns: Optional[List[str]] = None,
        raise_on_fail: bool = False,
    ) -> None:
        """
        Use this function to test the proportional hazards assumption. See usage example at
        https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html


        Parameters
        -----------

        training_df: DataFrame
            the original DataFrame used in the call to ``fit(...)`` or a sub-sampled version.
        advice: bool, optional
            display advice as output to the user's screen
        show_plots: bool, optional
            display plots of the scaled Schoenfeld residuals and loess curves. This is an eyeball test for violations.
            This will slow down the function significantly.
        p_value_threshold: float, optional
            the threshold to use to alert the user of violations. See note below.
        plot_n_bootstraps:
            in the plots displayed, also display plot_n_bootstraps bootstrapped loess curves. This will slow down
            the function significantly.
        columns: list, optional
            specify a subset of columns to test.
        raise_on_fail: bool, optional
            throw a ``ProportionalHazardAssumptionError`` if the test fails. Default: False.

        Returns
        --------
            A list of list of axes objects.


        Examples
        ----------

        .. code:: python

            from lifelines.datasets import load_rossi
            from lifelines import CoxPHFitter

            rossi = load_rossi()
            cph = CoxPHFitter().fit(rossi, 'week', 'arrest')

            axes = cph.check_assumptions(rossi, show_plots=True)


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
        test_results = proportional_hazard_test(self, training_df, time_transform=["rank", "km"], precomputed_residuals=residuals)

        residuals_and_duration = residuals.join(training_df[self.duration_col])
        Xs = self.regressors.transform_df(training_df)

        counter = 0
        n = residuals_and_duration.shape[0]
        axes = []

        for variable in self.params_.index.intersection(columns or self.params_.index):
            minimum_observed_p_value = test_results.summary.loc[variable, "p"].min()

            # plot is done (regardless of test result) whenever `show_plots = True`
            if show_plots:
                axes.append([])
                print()
                print("   Bootstrapping lowess lines. May take a moment...")
                print()
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
                    axes[-1].append(ax)

                fig.suptitle("Scaled Schoenfeld residuals of '%s'" % variable, fontsize=14)
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)

            if np.round(minimum_observed_p_value, 2) > p_value_threshold:
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
                % (counter, variable, format_p_value(4)(minimum_observed_p_value)),
                end="\n\n",
            )

            if advice:
                values = Xs["beta_"][variable]
                value_counts = values.value_counts()
                n_uniques = value_counts.shape[0]

                # Arbitrary chosen to check for ability to use strata col.
                # This should capture dichotomous / low cardinality values.
                if n_uniques <= 6 and value_counts.min() >= 5:
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
        #################

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
        elif raise_on_fail:
            raise ProportionalHazardAssumptionError()
        return axes

    @property
    def hazard_ratios_(self):
        return Series(np.exp(self.params_), index=self.params_.index, name="exp(coef)")

    def compute_followup_hazard_ratios(self, training_df: DataFrame, followup_times: Iterable) -> DataFrame:
        """
        Recompute the hazard ratio at different follow-up times (lifelines handles accounting for updated censoring and updated durations).
        This is useful because we need to remember that the hazard ratio is actually a weighted-average of period-specific hazard ratios.

        Parameters
        ----------

        training_df: pd.DataFrame
            The same dataframe used to train the model
        followup_times: Iterable
            a list/array of follow-up times to recompute the hazard ratio at.


        """
        results = {}
        for t in sorted(followup_times):
            assert t <= training_df[self.duration_col].max(), "all follow-up times must be less than max observed duration"
            df = training_df.copy()
            # if we "rollback" the df to time t, who is dead and who is censored
            df[self.event_col] = (df[self.duration_col] <= t) & df[self.event_col]
            df[self.duration_col] = np.minimum(df[self.duration_col], t)

            model = self.__class__(penalizer=self.penalizer, l1_ratio=self.l1_ratio).fit(
                df, self.duration_col, self.event_col, weights_col=self.weights_col, entry_col=self.entry_col
            )
            results[t] = model.hazard_ratios_
        return DataFrame(results).T
