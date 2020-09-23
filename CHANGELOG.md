## Changelog

#### 0.25.5 - 2020-09-23

##### API Changes
 - `check_assumptions` now returns a list of list of axes that can be manipulated

##### Bug fixes
 - fixed error when using `plot_partial_effects` with categorical data in AFT models
 - improved warning when Hessian matrix contains NaNs.

#### 0.25.4 - 2020-08-26

##### New features
 - New baseline estimator for Cox models: ``piecewise``
 - Performance improvements for parametric models `log_likelihood_ratio_test()` and `print_summary()`
 - Better step-size defaults for Cox model -> more robust convergence.


##### Bug fixes
 - fix `check_assumptions` when using formulas.


#### 0.25.3 - 2020-08-24

##### New features
 - `survival_difference_at_fixed_point_in_time_test` now accepts fitters instead of raw data, meaning that you can use this function on left, right or interval censored data.

##### API Changes
 - See note on `survival_difference_at_fixed_point_in_time_test` above.

##### Bug fixes
 - fix `StatisticalResult` printing in notebooks
 - fix Python error when calling `plot_covariate_groups`
 - fix dtype mismatches in `plot_partial_effects_on_outcome`.


#### 0.25.2 - 2020-08-08

##### New features
 - Spline `CoxPHFitter` can now use `strata`.

##### API Changes
 - a small parameterization change of the spline `CoxPHFitter`. The linear term in the spline part was moved to a new `Intercept` term in the `beta_`.
 - `n_baseline_knots` in the spline `CoxPHFitter` now refers to _all_ knots, and not just interior knots (this was confusing to me, the author.). So add 2 to `n_baseline_knots` to recover the identical model as previously.

##### Bug fixes
 - fix splines `CoxPHFitter` with  when `predict_hazard` was called.
 - fix some exception imports I missed.
 - fix log-likelihood p-value in splines `CoxPHFitter`


#### 0.25.1 - 2020-08-01

##### Bug fixes
 - ok _actually_ ship the out-of-sample calibration code
 - fix `labels=False` in `add_at_risk_counts`
 - allow for specific rows to be shown in `add_at_risk_counts`
 - put `patsy` as a proper dependency.
 - suppress some Pandas 1.1 warnings.


#### 0.25.0 - 2020-07-27

##### New features
 - Formulas! *lifelines* now supports R-like formulas in regression models. See docs [here](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#fitting-the-regression).
 - `plot_covariate_group` now can plot other y-values like hazards and cumulative hazards (default: survival function).
 - `CoxPHFitter` now accepts late entries via `entry_col`.
 - `calibration.survival_probability_calibration` now works with out-of-sample data.
 - `print_summary` now accepts a `column` argument to filter down the displayed values. This helps with clutter in notebooks, latex, or on the terminal.
 - `add_at_risk_counts` now follows the cool new KMunicate suggestions


##### API Changes
 - With the introduction of formulas, all models can be using formulas under the hood.
    - For both custom regression models or non-AFT regression models, this means that you no longer need to add a constant column to your DataFrame (instead add a `1` as a formula string in the `regressors` dict). You may also need to remove the T and E columns from `regressors`. I've updated the models in the `\examples` folder with examples of this new model building.
 - Unfortunately, if using formulas, your model will not be able to be pickled. This is a problem with an upstream library, and I hope to have it resolved in the near future.
 - `plot_covariate_groups` has been deprecated in favour of `plot_partial_effects_on_outcome`.
 - The baseline in `plot_covariate_groups` has changed from the *mean* observation (including dummy-encoded categorical variables) to *median* for ordinal (including continuous) and *mode* for categorical.
 - Previously, *lifelines* used the label `"_intercept"` to when it added a constant column in regressions. To align with Patsy, we are now using `"Intercept"`.
 - In AFT models, `ancillary_df` kwarg has been renamed to `ancillary`. This reflects the more general use of the kwarg (not always a DataFrame, but could be a boolean or string now, too).
 - Some column names in datasets shipped with lifelines have changed.
 - The never used "lifelines.metrics" is deleted.
 - With the introduction of formulas, `plot_covariate_groups` (now called `plot_partial_effects_on_outcome`) behaves differently for transformed variables. Users no longer need to add "derivatives" features, and encoding is done implicitly. See docs [here](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#plotting-the-effect-of-varying-a-covariate).
 - all exceptions and warnings have moved to `lifelines.exceptions`

##### Bug fixes
 - The p-value of the log-likelihood ratio test for the CoxPHFitter with splines was returning the wrong result because the degrees of freedom was incorrect.
 - better `print_summary` logic in IDEs and Jupyter exports. Previously it should not be displayed.
 - p-values have been corrected in the `SplineFitter`. Previously, the "null hypothesis" was no coefficient=0, but coefficient=0.01. This is now set to the former.
 - fixed NaN bug in `survival_table_from_events` with intervals when no events would occur in a interval.

#### 0.24.16 - 2020-07-09

##### New features
 - improved algorithm choice for large DataFrames for Cox models. Should see a significant performance boost.

##### Bug fixes
- fixed `utils.median_survival_time` not accepting Pandas Series.

#### 0.24.15 - 2020-07-07

##### Bug fixes
- fixed an edge case in `KaplanMeierFitter` where a really late entry would occur after all other population had died.
- fixed `plot` in `BreslowFlemingtonHarrisFitter`
- fixed bug where using `conditional_after` and `times` in `CoxPHFitter("spline")` prediction methods would be ignored.


#### 0.24.14 - 2020-07-02

##### Bug fixes
- fixed a bug where using `conditional_after` and `times` in prediction methods would result in a shape error
- fixed a bug where `score` was not able to be used in splined `CoxPHFitter`
- fixed a bug where some columns would not be displayed in `print_summary`

#### 0.24.13 - 2020-06-22

##### Bug fixes
- fixed a bug where `CoxPHFitter` would ignore inputed `alpha` levels for confidence intervals
- fixed a bug where `CoxPHFitter` would fail with working with `sklearn_adapter`


#### 0.24.12 - 2020-06-20

##### New features
 - improved convergence of `GeneralizedGamma(Regression)Fitter`.


#### 0.24.11 - 2020-06-17

##### New features
 - new spline regression model `CRCSplineFitter` based on the paper "A flexible parametric accelerated failure time model" by Michael J. Crowther, Patrick Royston, Mark Clements.
 - new survival probability calibration tool `lifelines.calibration.survival_probability_calibration` to help validate regression models. Based on “Graphical calibration curves and the integrated calibration index (ICI) for survival models” by P. Austin, F. Harrell, and D. van Klaveren.

##### API Changes
 - (and bug fix) scalar parameters in regression models were not being penalized by `penalizer` - we now penalizing everything except intercept terms in linear relationships.


#### 0.24.10 - 2020-06-16

##### New features
 - New improvements when using splines model in CoxPHFitter - it should offer much better prediction and baseline-hazard estimation, including extrapolation and interpolation.

##### API Changes
 - Related to above: the fitted spline parameters are now available in the `.summary` and `.print_summary` methods.

##### Bug fixes
- fixed a bug in initialization of some interval-censoring models -> better convergence.


#### 0.24.9 - 2020-06-05

##### New features
 - Faster NPMLE for interval censored data
 - New weightings available in the `logrank_test`: `wilcoxon`, `tarone-ware`, `peto`, `fleming-harrington`. Thanks @sean-reed
 - new interval censored dataset: `lifelines.datasets.load_mice`

##### Bug fixes
 - Cleared up some mislabeling in `plot_loglogs`. Thanks @sean-reed!
 - tuples are now able to be used as input in univariate models.

#### 0.24.8 - 2020-05-17

##### New features
 - Non parametric interval censoring is now available, _experimentally_. Not all edge cases are fully checked, and some features are missing. Try it under `KaplanMeierFitter.fit_interval_censoring`


#### 0.24.7 - 2020-05-17

##### New features
 - `find_best_parametric_model` can handle left and interval censoring. Also allows for more fitting options.
 - `AIC_` is a property on parametric models, and `AIC_partial_` is a property on Cox models.
 - `penalizer` in all regression models can now be an array instead of a float. This enables new functionality and better
 control over penalization. This is similar (but not identical) to `penalty.factors` in glmnet in R.
 - some convergence tweaks which should help recent performance regressions.

#### 0.24.6 - 2020-05-05

##### New features
 - At the cost of some performance, convergence is improved in many models.
 - New `lifelines.plotting.plot_interval_censored_lifetimes` for plotting interval censored data - thanks @sean-reed!

##### Bug fixes
 - fixed bug where `cdf_plot` and `qq_plot` were not factoring in the weights correctly.

#### 0.24.5 - 2020-05-01

##### New features
 - `plot_lifetimes` accepts pandas Series.

##### Bug fixes
 - Fixed important bug in interval censoring models. Users using interval censoring are strongly advised to upgrade.
 - Improved `at_risk_counts` for subplots.
 - More data validation checks for `CoxTimeVaryingFitter`

#### 0.24.4 - 2020-04-13

##### Bug fixes
 - Improved stability of interval censoring in parametric models.
 - setting a dataframe in `ancillary_df` works for interval censoring
 - `.score` works for interval censored models

#### 0.24.3 - 2020-03-25

##### New features
 - new `logx` kwarg in plotting curves
 - PH models have `compute_followup_hazard_ratios` for simulating what the hazard ratio would be at previous times. This is useful because the final hazard ratio is some weighted average of these.

##### Bug fixes
 - Fixed error in HTML printer that was hiding concordance index information.


#### 0.24.2 - 2020-03-15

##### Bug fixes
 - Fixed bug when no covariates were passed into `CoxPHFitter`. See #975
 - Fixed error in `StatisticalResult` where the test name was not displayed correctly.
 - Fixed a keyword bug in `plot_covariate_groups` for parametric models.


#### 0.24.1 - 2020-03-05

##### New features
 - Stability improvements for GeneralizedGammaRegressionFitter and CoxPHFitter with spline estimation.

##### Bug fixes
 - Fixed bug with plotting hazards in NelsonAalenFitter.


#### 0.24.0 - 2020-02-20

This version and future versions of lifelines no longer support py35. Pandas 1.0 is fully supported, along with previous versions. Minimum Scipy has been bumped to 1.2.0.

##### New features
 - `CoxPHFitter` and `CoxTimeVaryingFitter` has support for an elastic net penalty, which includes L1 and L2 regression.
 - `CoxPHFitter` has new baseline survival estimation methods. Specifically, `spline` now estimates the coefficients and baseline survival using splines. The traditional method, `breslow`, is still the default however.
 - Regression models have a new `score` method that will score your model against a dataset (ex: a testing or validation dataset). The default is to evaluate the log-likelihood, but also the concordance index can be chose.
 - New `MixtureCureFitter` for quickly creating univariate mixture models.
 - Univariate parametric models have a `plot_density`, `density_at_times`, and property `density_` that computes the probability density function estimates.
 - new dataset for interval regression involving *C. Botulinum*.
 - new `lifelines.fitters.mixins.ProportionalHazardMixin` that implements proportional hazard checks.

##### API Changes
 - Models' prediction method that return a single array now return a Series (use to return a DataFrame). This includes `predict_median`, `predict_percentile`, `predict_expectation`, `predict_log_partial_hazard`, and possibly others.
 - The penalty in Cox models is now scaled by the number of observations. This makes it invariant to changing sample sizes. This change also make the penalty magnitude behave the same as any parametric regression model.
 - `score_` on models has been renamed `concordance_index_`
 - models' `.variance_matrix_` is now a DataFrame.
 - `CoxTimeVaryingFitter` no longer requires an `id_col`. It's optional, and some checks may be done for integrity if provided.
 - Significant changes to `utils.k_fold_cross_validation`.
 - removed automatically adding `inf` from `PiecewiseExponentialRegressionFitter.breakpoints` and `PiecewiseExponentialFitter.breakpoints`
 - `tie_method` was dropped from Cox models (it was always Efron anyways...)
 - Mixins are moved to `lifelines.fitters.mixins`
 - `find_best_parametric_model` `evaluation` kwarg has been changed to `scoring_method`.
 - removed `_score_` and `path` from Cox model.

##### Bug fixes
 - Fixed `show_censors` with `KaplanMeierFitter.plot_cumulative_density` see issue #940.
 - Fixed error in `"BIC"` code path in `find_best_parametric_model`
 - Fixed a bug where left censoring in AFT models was not converging well
 - Cox models now incorporate any penalizers in their `log_likelihood_`


#### 0.23.9 - 2020-01-28

##### Bug fixes
 - fixed important error when a parametric regression model would not assign the correct labels to fitted
parameters' variances. See more here: https://github.com/CamDavidsonPilon/lifelines/issues/931. Users of `GeneralizedGammaRegressionFitter` and any custom regression models should update their code as soon as possible.

#### 0.23.8 - 2020-01-21

##### Bug fixes
 - fixed important error when a parametric regression model would not assign the correct labels to fitted
parameters. See more here: https://github.com/CamDavidsonPilon/lifelines/issues/931. Users of `GeneralizedGammaRegressionFitter` and any custom regression models should update their code as soon as possible.

#### 0.23.7 - 2020-01-14

Bug fixes for py3.5.

#### 0.23.6 - 2020-01-07

##### New features
 - New univariate model, `SplineFitter`, that uses cubic splines to model the cumulative hazard.
 - To aid users with selecting the best parametric model, there is a new `lifelines.utils.find_best_parametric_model` function that will iterate through the models and return the model with the lowest AIC (by default).
 - custom parametric regression models can now do left and interval censoring.


#### 0.23.5 - 2020-01-05

##### New features
 - New `predict_hazard` for parametric regression models.
 - New lymph node cancer dataset, originally from *H.F. for the German Breast Cancer Study Group (GBSG) (1994)*

##### Bug fixes
 - fixes error thrown when converge of regression models fails.
 - `kwargs` is now used in `plot_covariate_groups`
 - fixed bug where large exponential numbers in `print_summary` were not being suppressed correctly.

#### 0.23.4 - 2019-12-15

 - Bug fix for PyPI

#### 0.23.3 - 2019-12-11

##### New features
 - `StatisticalResult.print_summary` supports html output.

##### Bug fixes
 - fix import in `printer.py`
 - fix html printing with Univariate models.


#### 0.23.2 - 2019-12-07

##### New features
 - new `lifelines.plotting.rmst_plot` for pretty figures of survival curves and RMSTs.
 - new variance calculations for `lifelines.utils.resticted_mean_survival_time`
 - performance improvements on regression models' preprocessing. Should make datasets with
 high number of columns more performant.

##### Bug fixes
 - fixed `print_summary` for AAF class.
 - fixed repr for `sklearn_adapter` classes.
 - fixed `conditional_after` in Cox model with strata was used.


#### 0.23.1 - 2019-11-27

##### New features
 - new `print_summary` option `style` to print HTML, LaTeX or ASCII output
 - performance improvements for `CoxPHFitter` - up to 30% performance improvements for some datasets.

##### Bug fixes
 - fixed bug where computed statistics were not being shown in `print_summary` for HTML output.
 - fixed bug where "None" was displayed in models' `__repr__`
 - fixed bug in `StatisticalResult.print_summary`
 - fixed bug when using `print_summary` with left censored models.
 - lots of minor bug fixes.

#### 0.23.0 - 2019-11-17

##### New features
 - new `print_summary` abstraction that allows HTML printing in Jupyter notebooks!
 - silenced some warnings.

##### Bug fixes
 - The "comparison" value of some parametric univariate models wasn't standard, so the null hypothesis p-value may have been wrong. This is now fixed.
 - fixed a NaN error in confidence intervals for KaplanMeierFitter

##### API Changes

 - To align values across models, the column names for the confidence intervals in parametric univariate models `summary` have changed.
 - Fixed typo in `ParametricUnivariateFitter` name.
 - `median_` has been removed in favour of `median_survival_time_`.
 - `left_censorship` in `fit` has been removed in favour of `fit_left_censoring`.


#### 0.22.10 - 2019-11-08

The tests were re-factored to be shipped with the package. Let me know if this causes problems.


##### Bug fixes
 - fixed error in plotting models with "lower" or "upper" was in the label name.
 - fixed bug in plot_covariate_groups for AFT models when >1d arrays were used for values arg.


#### 0.22.9 - 2019-10-30


##### Bug fixes
 - fixed `predict_` methods in AFT models when `timeline` was not specified.
 - fixed error in `qq_plot`
 - fixed error when submitting a model in `qth_survival_time`
 - `CoxPHFitter` now displays correct columns values when changing alpha param.


#### 0.22.8 - 2019-10-06

##### New features
 - Serializing lifelines is better supported. Packages like joblib and pickle are now supported. Thanks @AbdealiJK!
 - `conditional_after` now available in `CoxPHFitter.predict_median`
 - Suppressed some unimportant warnings.

##### Bug fixes
 - fixed initial_point being ignored in AFT models.


#### 0.22.7 - 2019-09-29

##### New features
 - new `ApproximationWarning` to tell you if the package is making an potentially mislead approximation.

##### Bug fixes
 - fixed a bug in parametric prediction for interval censored data.
 - realigned values in `print_summary`.
 - fixed bug in `survival_difference_at_fixed_point_in_time_test`

##### API Changes

 - `utils.qth_survival_time` no longer takes a `cdf` argument - users should take the compliment (1-cdf).
 - Some previous `StatisticalWarnings` have been replaced by `ApproximationWarning`

#### 0.22.6 - 2019-09-25

##### New features
 - `conditional_after` works for `CoxPHFitter` prediction models 😅

##### Bug fixes

##### API Changes
 - `CoxPHFitter.baseline_cumulative_hazard_`'s column is renamed `"baseline cumulative hazard"` - previously it was `"baseline hazard"`. (Only applies if the model has no strata.)
 - `utils.dataframe_interpolate_at_times` renamed to `utils.interpolate_at_times_and_return_pandas`.


#### 0.22.5 - 2019-09-20

##### New features
 - Improvements to the __repr__ of models that takes into accounts weights.
 - Better support for predicting on Pandas Series

##### Bug fixes
 - Fixed issue where `fit_interval_censoring` wouldn't accept lists.
 - Fixed an issue with `AalenJohansenFitter` failing to plot confidence intervals.

##### API Changes
 - `_get_initial_value` in parametric univariate models is renamed `_create_initial_point`


#### 0.22.4 - 2019-09-04

##### New features
 - Some performance improvements to regression models.
 - lifelines will avoid penalizing the intercept (aka bias) variables in regression models.
 - new `utils.restricted_mean_survival_time` that approximates the RMST using numerical integration against survival functions.

##### API changes
 - `KaplanMeierFitter.survival_function_`'s' index is no longer given the name "timeline".

##### Bug fixes
 - Fixed issue where `concordance_index` would never exit if NaNs in dataset.


#### 0.22.3 - 2019-08-08

##### New features
 - model's now expose a `log_likelihood_` property.
 - new `conditional_after` argument on `predict_*` methods that make prediction on censored subjects easier.
 - new `lifelines.utils.safe_exp` to make `exp` overflows easier to handle.
 - smarter initial conditions for parametric regression models.
 - New regression model: `GeneralizedGammaRegressionFitter`

##### API changes
 - removed `lifelines.utils.gamma` - use `autograd_gamma` library instead.
 - removed bottleneck as a dependency. It offered slight performance gains only in Cox models, and only a small fraction of the API was being used.

##### Bug fixes
 - AFT log-likelihood ratio test was not using weights correctly.
 - corrected (by bumping) scipy and autograd dependencies
 - convergence is improved for most models, and many `exp` overflow warnings have been eliminated.
 - Fixed an error in the `predict_percentile` of `LogLogisticAFTFitter`. New tests have been added around this.


#### 0.22.2 - 2019-07-25

##### New features
 - lifelines is now compatible with scipy>=1.3.0

##### Bug fixes
 - fixed printing error when using robust=True in regression models
 - `GeneralizedGammaFitter` is more stable, maybe.
 - lifelines was allowing old version of numpy (1.6), but this caused errors when using the library. The correctly numpy has been pinned (to 1.14.0+)



#### 0.22.1 - 2019-07-14

##### New features
 - New univariate model, `GeneralizedGammaFitter`. This model contains many sub-models, so it is a good model to check fits.
 - added a warning when a time-varying dataset had instantaneous deaths.
 - added a `initial_point` option in univariate parametric fitters.
 - `initial_point` kwarg is present in parametric univariate fitters `.fit`
 - `event_table` is now an attribute on all univariate fitters (if right censoring)
 - improvements to `lifelines.utils.gamma`

##### API changes
 - In AFT models, the column names in `confidence_intervals_` has changed to include the alpha value.
 - In AFT models, some column names in `.summary` and `.print_summary` has changed to include the alpha value.
 - In AFT models, some column names in `.summary` and `.print_summary` includes confidence intervals for the exponential of the value.

##### Bug fixes
 - when using `censors_show` in plotting functions, the censor ticks are now reactive to the estimate being shown.
 - fixed an overflow bug in `KaplanMeierFitter` confidence intervals
 - improvements in data validation for `CoxTimeVaryingFitter`


#### 0.22.0 - 2019-07-03

##### New features
 - Ability to create custom parametric regression models by specifying the cumulative hazard. This enables new and extensions of AFT models.
 - `percentile(p)` method added to univariate models that solves the equation `p = S(t)` for `t`
 - for parametric univariate models, the `conditional_time_to_event_` is now exact instead of an approximation.

##### API changes
 - In Cox models, the attribute `hazards_` has been renamed to `params_`. This aligns better with the other regression models, and is more clear (what is a hazard anyways?)
 - In Cox models, a new `hazard_ratios_` attribute is available which is the exponentiation of `params_`.
 - In Cox models, the column names in `confidence_intervals_` has changed to include the alpha value.
 - In Cox models, some column names in `.summary` and `.print_summary` has changed to include the alpha value.
 - In Cox models, some column names in `.summary` and `.print_summary` includes confidence intervals for the exponential of the value.
 - Significant changes to internal AFT code.
 - A change to how `fit_intercept` works in AFT models. Previously one could set `fit_intercept` to False and not have to set `ancillary_df` - now one must specify a DataFrame.

##### Bug fixes
 - for parametric univariate models, the `conditional_time_to_event_` is now exact instead of an approximation.
 - fixed a name error bug in `CoxTimeVaryingFitter.plot`

#### 0.21.5 - 2019-06-22

I'm skipping 0.21.4 version because of deployment issues.

##### New features
 - `scoring_method` now a kwarg on `sklearn_adapter`

##### Bug fixes
 - fixed an implicit import of scikit-learn. scikit-learn is an optional package.
 - fixed visual bug that misaligned x-axis ticks and at-risk counts. Thanks @christopherahern!


#### 0.21.3 - 2019-06-04

##### New features
 - include in lifelines is a scikit-learn adapter so lifeline's models can be used with scikit-learn's API. See [documentation here](https://lifelines.readthedocs.io/en/latest/Compatibility%20with%20scikit-learn.html).
 - `CoxPHFitter.plot` now accepts a `hazard_ratios` (boolean) parameter that will plot the hazard ratios (and CIs) instead of the log-hazard ratios.
 - `CoxPHFitter.check_assumptions` now accepts a `columns` parameter to specify only checking a subset of columns.

##### Bug fixes
 - `covariates_from_event_matrix` handle nulls better


#### 0.21.2 - 2019-05-16

##### New features
 - New regression model: `PiecewiseExponentialRegressionFitter` is available. See blog post here: https://dataorigami.net/blogs/napkin-folding/churn
 - Regression models have a new method `log_likelihood_ratio_test` that computes, you guessed it, the log-likelihood ratio test. Previously this was an internal API that is being exposed.

##### API changes
 - The default behavior of the `predict` method on non-parametric estimators (`KaplanMeierFitter`, etc.) has changed from (previous) linear interpolation to (new) return last value. Linear interpolation is still possible with the `interpolate` flag.
 - removing `_compute_likelihood_ratio_test` on regression models. Use `log_likelihood_ratio_test` now.

##### Bug fixes


#### 0.21.1 - 2019-04-26

##### New features
 - users can provided their own start and stop column names in `add_covariate_to_timeline`
 - PiecewiseExponentialFitter now allows numpy arrays as breakpoints

##### API changes
 - output of `survival_table_from_events` when collapsing rows to intervals now removes the "aggregate" column multi-index.

##### Bug fixes
 - fixed bug in CoxTimeVaryingFitter when ax is provided, thanks @j-i-l!

#### 0.21.0 - 2019-04-12

##### New features
 - `weights` is now a optional kwarg for parametric univariate models.
 - all univariate and multivariate parametric models now have ability to handle left, right and interval censored data (the former two being special cases of the latter). Users can use the `fit_right_censoring` (which is an alias for `fit`), `fit_left_censoring` and `fit_interval_censoring`.
 - a new interval censored dataset is available under `lifelines.datasets.load_diabetes`

##### API changes
 - `left_censorship` on all univariate fitters has been deprecated. Please use the new
 api `model.fit_left_censoring(...)`.
 - `invert_y_axis` in `model.plot(...` has been removed.
 - `entries` property in multivariate parametric models has a new Series name: `entry`

##### Bug fixes
 - lifelines was silently converting any NaNs in the event vector to True. An error is now thrown instead.
 - Fixed an error that didn't let users use Numpy arrays in prediction for AFT models


#### 0.20.5 - 2019-04-08

##### New features
 - performance improvements for `print_summary`.

##### API changes
 - `utils.survival_events_from_table` returns an integer weight vector as well as durations and censoring vector.
 - in `AalenJohansenFitter`, the `variance` parameter is renamed to `variance_` to align with the usual lifelines convention.

##### Bug fixes
 - Fixed an error in the `CoxTimeVaryingFitter`'s likelihood ratio test when using strata.
 - Fixed some plotting bugs with `AalenJohansenFitter`


#### 0.20.4 - 2019-03-27

##### New features
 - left-truncation support in AFT models, using the `entry_col` kwarg in `fit()`
 - `generate_datasets.piecewise_exponential_survival_data` for generating piecewise exp. data
 - Faster `print_summary` for AFT models.

##### API changes
 - Pandas is now correctly pinned to >= 0.23.0. This was always the case, but not specified in setup.py correctly.

##### Bug fixes
 - Better handling for extremely large numbers in `print_summary`
 - `PiecewiseExponentialFitter` is available with `from lifelines import *`.


#### 0.20.3 - 2019-03-23

##### New features
 - Now `cumulative_density_` & `survival_function_` are _always_ present on a fitted `KaplanMeierFitter`.
 - New attributes/methods on `KaplanMeierFitter`: `plot_cumulative_density()`, `confidence_interval_cumulative_density_`, `plot_survival_function` and `confidence_interval_survival_function_`.


#### 0.20.2 - 2019-03-21

##### New features
 - Left censoring is now supported in univariate parametric models: `.fit(..., left_censorship=True)`. Examples are in the docs.
 - new dataset: `lifelines.datasets.load_nh4()`
 - Univariate parametric models now include, by default, support for the cumulative density function: `.cumulative_density_`, `.confidence_interval_cumulative_density_`, `plot_cumulative_density()`, `cumulative_density_at_times(t)`.
 -  add a `lifelines.plotting.qq_plot` for univariate parametric models that handles censored data.

##### API changes
 - `plot_lifetimes` no longer reverses the order when plotting. Thanks @vpolimenov!
 - The `C` column in `load_lcd` dataset is renamed to `E`.

##### Bug fixes
 - fixed a naming error in `KaplanMeierFitter` when `left_censorship` was set to True, `plot_cumulative_density_()` is now `plot_cumulative_density()`.
 - added some error handling when passing in timedeltas. Ideally, users don't pass in timedeltas, as the scale is ambiguous. However, the error message before was not obvious, so we do some conversion, warn the user, and pass it through.
 - `qth_survival_times` for a truncated CDF would return `np.inf` if the q parameter was below the truncation limit. This should have been `-np.inf`


#### 0.20.1 - 2019-03-16

 - Some performance improvements to `CoxPHFitter` (about 30%). I know it may seem silly, but we are now about the same or slighty faster than the Cox model in R's `survival` package (for some testing datasets and some configurations). This is a big deal, because 1) lifelines does more error checking prior, 2) R's cox model is written in C, and we are still pure Python/NumPy, 3) R's cox model has decades of development.
 - suppressed unimportant warnings

##### API changes
 - Previously, lifelines _always_ added a 0 row to `cph.baseline_hazard_`, even if there were no event at this time. This is no longer the case. A 0 will still be added if there is a duration (observed or not) at 0 occurs however.


#### 0.20.0 - 2019-03-05

 - Starting with 0.20.0, only Python3 will be supported. Over 75% of recent installs where Py3.
 - Updated minimum dependencies, specifically Matplotlib and Pandas.

##### New features
 -  smarter initialization for AFT models which should improve convergence.

##### API changes
 - `inital_beta` in Cox model's `.fit` is now `initial_point`.
 - `initial_point` is now available in AFT models and `CoxTimeVaryingFitter`
 - the DataFrame `confidence_intervals_` for univariate models is transposed now (previous parameters where columns, now parameters are rows).

##### Bug fixes
 - Fixed a bug with plotting and `check_assumptions`.



#### 0.19.5 - 2019-02-26

##### New features
 -  `plot_covariate_group` can accept multiple covariates to plot. This is useful for columns that have implicit correlation like polynomial features or categorical variables.
 - Convergence improvements for AFT models.

#### 0.19.4 - 2019-02-25

##### Bug fixes
 - remove some bad print statements in `CoxPHFitter`.

#### 0.19.3 - 2019-02-25

##### New features
 - new AFT models: `LogNormalAFTFitter` and `LogLogisticAFTFitter`.
 - AFT models now accept a `weights_col` argument to `fit`.
 - Robust errors (sandwich errors) are now avilable in AFT models using the `robust=True` kwarg in `fit`.
 - Performance increase to `print_summary` in the `CoxPHFitter` and `CoxTimeVaryingFitter` model.

#### 0.19.2 - 2019-02-22

##### New features
 - `ParametricUnivariateFitters`, like `WeibullFitter`, have smoothed plots when plotting (vs stepped plots)

##### Bug fixes
 - The `ExponentialFitter` log likelihood _value_ was incorrect - inference was correct however.
 - Univariate fitters are more flexiable and can allow 2-d and DataFrames as inputs.

#### 0.19.1 - 2019-02-21

##### New features
 - improved stability of `LogNormalFitter`
 - Matplotlib for Python3 users are not longer forced to use 2.x.

##### API changes
 - **Important**: we changed the parameterization of the `PiecewiseExponential` to the same as `ExponentialFitter` (from `\lambda * t` to `t / \lambda`).


#### 0.19.0 - 2019-02-20

##### New features
 - New regression model `WeibullAFTFitter` for fitting accelerated failure time models. Docs have been added to our [documentation](https://lifelines.readthedocs.io/) about how to use `WeibullAFTFitter` (spoiler: it's API is similar to the other regression models) and how to interpret the output.
 - `CoxPHFitter` performance improvements (about 10%)
 - `CoxTimeVaryingFitter` performance improvements (about 10%)


##### API changes
 - **Important**: we changed the `.hazards_` and `.standard_errors_` on Cox models to be pandas Series (instead of Dataframes). This felt like a more natural representation of them. You may need to update your code to reflect this. See notes here: https://github.com/CamDavidsonPilon/lifelines/issues/636
 - **Important**: we changed the `.confidence_intervals_` on Cox models to be transposed. This felt like a more natural representation of them. You may need to update your code to reflect this. See notes here: https://github.com/CamDavidsonPilon/lifelines/issues/636
 - **Important**: we changed the parameterization of the `WeibullFitter` and `ExponentialFitter` from `\lambda * t` to `t / \lambda`. This was for a few reasons: 1) it is a more common parameterization in literature, 2) it helps in convergence.
 - **Important**: in models where we add an intercept (currently only `AalenAdditiveModel`), the name of the added column has been changed from `baseline` to `_intercept`
 - **Important**: the meaning of `alpha` in all fitters has changed to be the standard interpretation of alpha in confidence intervals. That means that the _default_ for alpha is set to 0.05 in the latest lifelines, instead of 0.95 in previous versions.

##### Bug Fixes
 - Fixed a bug in the `_log_likelihood_` property of `ParametericUnivariateFitter` models. It was showing the "average" log-likelihood (i.e. scaled by 1/n) instead of the total. It now displays the total.
 - In model `print_summary`s, correct a label erroring. Instead of "Likelihood test", it should have read "Log-likelihood test".
 - Fixed a bug that was too frequently rejecting the dtype of `event` columns.
 - Fixed a calculation bug in the concordance index for stratified Cox models. Thanks @airanmehr!
 - Fixed some Pandas <0.24 bugs.

#### 0.18.6 - 2019-02-13

 - some improvements to the output of `check_assumptions`. `show_plots` is turned to `False` by default now. It only shows `rank` and `km` p-values now.
 - some performance improvements to `qth_survival_time`.

#### 0.18.5 - 2019-02-11

 - added new plotting methods to parametric univariate models: `plot_survival_function`, `plot_hazard` and `plot_cumulative_hazard`. The last one is an alias for `plot`.
 - added new properties to parametric univarite models: `confidence_interval_survival_function_`, `confidence_interval_hazard_`, `confidence_interval_cumulative_hazard_`. The last one is an alias for `confidence_interval_`.
 - Fixed some overflow issues with `AalenJohansenFitter`'s variance calculations when using large datasets.
 - Fixed an edgecase in `AalenJohansenFitter` that causing some datasets with to be jittered too often.
 - Add a new kwarg to  `AalenJohansenFitter`, `calculate_variance` that can be used to turn off variance calculations since this can take a long time for large datasets. Thanks @pzivich!

#### 0.18.4 - 2019-02-10

 - fixed confidence intervals in cumulative hazards for parametric univarite models. They were previously
   serverly depressed.
 - adding left-truncation support to parametric univarite models with the `entry` kwarg in `.fit`

#### 0.18.3 - 2019-02-07

 - Some performance improvements to parametric univariate models.
 - Suppressing some irrelevant NumPy and autograd warnings, so lifeline warnings are more noticeable.
 - Improved some warning and error messages.

#### 0.18.2 - 2019-02-05

 - New univariate fitter `PiecewiseExponentialFitter` for creating a stepwise hazard model. See docs online.
 - Ability to create novel parametric univariate models using the new `ParametericUnivariateFitter` super class. See docs online for how to do this.
 - Unfortunately, parametric univariate fitters are not serializable with `pickle`. The library `dill` is still useable.
 - Complete overhaul of all internals for parametric univariate fitters. Moved them all (most) to use `autograd`.
 - `LogNormalFitter` no longer models `log_sigma`.


#### 0.18.1 - 2019-02-02
 - bug fixes in `LogNormalFitter` variance estimates
 - improve convergence of `LogNormalFitter`. We now model the log of sigma internally, but still expose sigma externally.
 - use the `autograd` lib to help with gradients.
 - New `LogLogisticFitter` univariate fitter available.

#### 0.18.0 - 2019-01-31

 - `LogNormalFitter` is a new univariate fitter you can use.
 - `WeibullFitter` now correctly returns the confidence intervals (previously returned only NaNs)
 - `WeibullFitter.print_summary()` displays p-values associated with its parameters not equal to 1.0 - previously this was (implicitly) comparing against 0, which is trivially always true (the parameters must be greater than 0)
 - `ExponentialFitter.print_summary()` displays p-values associated with its parameters not equal to 1.0 - previously this was (implicitly) comparing against 0, which is trivially always true (the parameters must be greater than 0)
 - `ExponentialFitter.plot` now displays the cumulative hazard, instead of the survival function. This is to make it easier to compare to `WeibullFitter` and `LogNormalFitter`
 - Univariate fitters' `cumulative_hazard_at_times`, `hazard_at_times`, `survival_function_at_times` return pandas Series now (use to be numpy arrays)
 - remove `alpha` keyword from all statistical functions. This was never being used.
 - Gone are astericks and dots in `print_summary` functions that represent signficance thresholds.
 - In models' `summary` (including `print_summary`), the `log(p)` term has changed to `-log2(p)`. This is known as the s-value. See https://lesslikely.com/statistics/s-values/
 - introduce new statistical tests between univariate datasets: `survival_difference_at_fixed_point_in_time_test`,...
 - new warning message when Cox models detects possible non-unique solutions to maximum likelihood.
 - Generally: clean up lifelines exception handling. Ex: catch `LinAlgError: Matrix is singular.` and report back to the user advice.

#### 0.17.5 - 2019-01-25

 - more bugs in `plot_covariate_groups` fixed when using non-numeric strata.

#### 0.17.4 -2019-01-25

 - Fix bug in `plot_covariate_groups` that wasn't allowing for strata to be used.
 - change name of `multicenter_aids_cohort_study` to `load_multicenter_aids_cohort_study`
 - `groups` is now called `values` in `CoxPHFitter.plot_covariate_groups`

#### 0.17.3 - 2019-01-24
 - Fix in `compute_residuals` when using `schoenfeld` and the minumum duration has only censored subjects.

#### 0.17.2 2019-01-22
 - Another round of serious performance improvements for the Cox models. Up to 2x faster for CoxPHFitter and CoxTimeVaryingFitter. This was mostly the result of using NumPy's `einsum` to simplify a previous `for` loop. The downside is the code is more esoteric now. I've added comments as necessary though 🤞

#### 0.17.1 - 2019-01-20

 - adding bottleneck as a dependency. This library is highly-recommended by Pandas, and in lifelines we see some nice performance improvements with it too. (~15% for `CoxPHFitter`)
 - There was a small bug in `CoxPHFitter` when using `batch_mode` that was causing coefficients to deviate from their MLE value. This bug eluded tests, which means that it's discrepancy was less than 0.0001 difference. It's fixed now, and even more accurate tests are added.
 - Faster `CoxPHFitter._compute_likelihood_ratio_test()`
 - Fixes a Pandas performance warning in `CoxTimeVaryingFitter`.
 - Performances improvements to `CoxTimeVaryingFitter`.

#### 0.17.0 - 2019-01-11

 - corrected behaviour in `CoxPHFitter` where `score_` was not being refreshed on every new `fit`.
 - Reimplentation of `AalenAdditiveFitter`. There were significant changes to it:
   - implementation is at least 10x faster, and possibly up to 100x faster for some datasets.
   - memory consumption is way down
   - removed the time-varying component from `AalenAdditiveFitter`. This will return in a future release.
   - new `print_summary`
   - `weights_col` is added
   - `nn_cumulative_hazard` is removed (may add back)
 - some plotting improvemnts to `plotting.plot_lifetimes`


#### 0.16.3 - 2019-01-03

 - More `CoxPHFitter` performance improvements. Up to a 40% reduction vs 0.16.2 for some datasets.

#### 0.16.2 - 2019-01-02

 - Fixed `CoxTimeVaryingFitter` to allow more than one variable to be stratafied
 - Significant performance improvements for `CoxPHFitter` with dataset has lots of duplicate times. See https://github.com/CamDavidsonPilon/lifelines/issues/591

#### 0.16.1 - 2019-01-01
 - Fixed py2 division error in `concordance` method.

#### 0.16.0 - 2019-01-01

 - Drop Python 3.4 support.
 - introduction of residual calculations in `CoxPHFitter.compute_residuals`. Residuals include "schoenfeld", "score", "delta_beta", "deviance", "martingale", and "scaled_schoenfeld".
 - removes `estimation` namespace for fitters. Should be using `from lifelines import xFitter` now. Thanks @usmanatron
 - removes `predict_log_hazard_relative_to_mean` from Cox model. Thanks @usmanatron
 - `StatisticalResult` has be generalized to allow for multiple results (ex: from pairwise comparisons). This means a slightly changed API that is mostly backwards compatible. See doc string for how to use it.
 - `statistics.pairwise_logrank_test` now returns a `StatisticalResult` object instead of a nasty NxN DataFrame 💗
 - Display log(p-values) as well as p-values in `print_summary`. Also, p-values below thesholds will be truncated. The orignal p-values are still recoverable using `.summary`.
 - Floats `print_summary` is now displayed to 2 decimal points. This can be changed using the `decimal` kwarg.
 - removed `standardized` from `Cox` model plotting. It was confusing.
 - visual improvements to Cox models `.plot`
 - `print_summary` methods accepts kwargs to also be displayed.
 - `CoxPHFitter` has a new human-readable method, `check_assumptions`, to check the assumptions of your Cox proportional hazard model.
 - A new helper util to "expand" static datasets into long-form: `lifelines.utils.to_episodic_format`.
 - `CoxTimeVaryingFitter` now accepts `strata`.

#### 0.15.4

 - bug fix for the Cox model likelihood ratio test when using non-trivial weights.

#### 0.15.3 - 2018-12-18
 - Only allow matplotlib less than 3.0.

#### 0.15.2 - 2018-11-23
 - API changes to `plotting.plot_lifetimes`
 - `cluster_col` and `strata` can be used together in `CoxPHFitter`
 - removed `entry` from `ExponentialFitter` and `WeibullFitter` as it was doing nothing.

#### 0.15.1 - 2018-11-23
 - Bug fixes for v0.15.0
 - Raise NotImplementedError if the `robust` flag is used in `CoxTimeVaryingFitter` - that's not ready yet.

#### 0.15.0 - 2018-11-22
 - adding `robust` params to `CoxPHFitter`'s `fit`. This enables atleast i) using non-integer weights in the model (these could be sampling weights like IPTW), and ii) mis-specified models (ex: non-proportional hazards). Under the hood it's a sandwich estimator. This does not handle ties, so if there are high number of ties, results may significantly differ from other software.
 - `standard_errors_` is now a property on fitted `CoxPHFitter` which describes the standard errors of the coefficients.
 - `variance_matrix_` is now a property on fitted `CoxPHFitter` which describes the variance matrix of the coefficients.
 - new criteria for convergence of `CoxPHFitter` and `CoxTimeVaryingFitter` called the Newton-decrement. Tests show it is as accurate (w.r.t to previous coefficients) and typically shaves off a single step, resulting in generally faster convergence. See https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/Newton_methods.pdf. Details about the Newton-decrement are added to the `show_progress` statements.
 - Minimum suppport for scipy is 1.0
 - Convergence errors in models that use Newton-Rhapson methods now throw a `ConvergenceError`, instead of a `ValueError` (the former is a subclass of the latter, however).
 - `AalenAdditiveModel` raises `ConvergenceWarning` instead of printing a warning.
 - `KaplanMeierFitter` now has a cumulative plot option. Example `kmf.plot(invert_y_axis=True)`
 - a `weights_col` option has been added to `CoxTimeVaryingFitter` that allows for time-varying weights.
 - `WeibullFitter` has a new `show_progress` param and additional information if the convergence fails.
 - `CoxPHFitter`, `ExponentialFitter`, `WeibullFitter` and `CoxTimeVaryFitter` method `print_summary` is updated with new fields.
 - `WeibullFitter` has renamed the incorrect `_jacobian` to `_hessian_`.
 - `variance_matrix_` is now a property on fitted `WeibullFitter` which describes the variance matrix of the parameters.
 - The default `WeibullFitter().timeline` has changed from integers between the min and max duration to _n_ floats between the max and min durations, where _n_ is the number of observations.
 - Performance improvements for `CoxPHFitter` (~20% faster)
 - Performance improvements for `CoxTimeVaryingFitter` (~100% faster)
 - In Python3, Univariate models are now serialisable with `pickle`. Thanks @dwilson1988 for the contribution. For Python2, `dill` is still the preferred method.
 - `baseline_cumulative_hazard_` (and derivatives of that) on `CoxPHFitter` now correctly incorporate the `weights_col`.
 - Fixed a bug in `KaplanMeierFitter` when late entry times lined up with death events. Thanks @pzivich
 - Adding `cluster_col` argument to `CoxPHFitter` so users can specify groups of subjects/rows that may be correlated.
 - Shifting the "signficance codes" for p-values down an order of magnitude. (Example, p-values between 0.1 and 0.05 are not noted at all and p-values between 0.05 and 0.1 are noted with `.`, etc.). This deviates with how they are presented in other software. There is an argument to be made to remove p-values from lifelines altogether (_become the changes you want to see in the world_ lol), but I worry that people could compute the p-values by hand incorrectly, a worse outcome I think. So, this is my stance. P-values between 0.1 and 0.05 offer _very_ little information, so they are removed. There is a growing movement in statistics to shift "signficant" findings to p-values less than 0.01 anyways.
 - New fitter for cumulative incidence of multiple risks `AalenJohansenFitter`. Thanks @pzivich! See "Methodologic Issues When Estimating Risks in Pharmacoepidemiology" for a nice overview of the model.

#### 0.14.6 - 2018-07-02
 - fix for n > 2 groups in `multivariate_logrank_test` (again).
 - fix bug for when `event_observed` column was not boolean.

#### 0.14.5 - 2018-06-29
 - fix for n > 2 groups in `multivariate_logrank_test`
 - fix weights in KaplanMeierFitter when using a pandas Series.

#### 0.14.4 - 2018-06-14
 - Adds `baseline_cumulative_hazard_` and `baseline_survival_` to `CoxTimeVaryingFitter`. Because of this, new prediction methods are available.
 - fixed a bug in `add_covariate_to_timeline` when using `cumulative_sum` with multiple columns.
 - Added `Likelihood ratio test` to `CoxPHFitter.print_summary` and `CoxTimeVaryingFitter.print_summary`
 - New checks in `CoxTimeVaryingFitter` that check for immediate deaths and redundant rows.
 - New `delay` parameter in `add_covariate_to_timeline`
 - removed `two_sided_z_test` from `statistics`

#### 0.14.3 - 2018-05-24
 - fixes a bug when subtracting or dividing two `UnivariateFitters` with labels.
 - fixes an import error with using `CoxTimeVaryingFitter` predict methods.
 - adds a `column` argument to `CoxTimeVaryingFitter` and `CoxPHFitter` `plot` method to plot only a subset of columns.

#### 0.14.2 - 2018-05-18
 - some quality of life improvements for working with `CoxTimeVaryingFitter` including new `predict_` methods.

#### 0.14.1 - 2018-04-01
 - fixed bug with using weights and strata in `CoxPHFitter`
 - fixed bug in using non-integer weights in `KaplanMeierFitter`
 - Performance optimizations in `CoxPHFitter` for up to 40% faster completion of `fit`.
    - even smarter `step_size` calculations for iterative optimizations.
    - simple code optimizations & cleanup in specific hot spots.
 - Performance optimizations in `AalenAdditiveFitter` for up to 50% faster completion of `fit` for large dataframes, and up to 10% faster for small dataframes.


#### 0.14.0 - 2018-03-03
 - adding `plot_covariate_groups` to `CoxPHFitter` to visualize what happens to survival as we vary a covariate, all else being equal.
 - `utils` functions like `qth_survival_times` and `median_survival_times` now return the transpose of the DataFrame compared to previous version of lifelines. The reason for this is that we often treat survival curves as columns in DataFrames, and functions of the survival curve as index (ex: KaplanMeierFitter.survival_function_ returns a survival curve _at_ time _t_).
 - `KaplanMeierFitter.fit` and `NelsonAalenFitter.fit` accept a `weights` vector that can be used for pre-aggregated datasets. See this [issue](https://github.com/CamDavidsonPilon/lifelines/issues/396).
 - Convergence errors now return a custom `ConvergenceWarning` instead of a `RuntimeWarning`
 - New checks for complete separation in the dataset for regressions.

#### 0.13.0 - 2017-12-22
 - removes `is_significant` and `test_result` from `StatisticalResult`. Users can instead choose their significance level by comparing to `p_value`. The string representation of this class has changed aswell.
 - `CoxPHFitter` and `AalenAdditiveFitter` now have a `score_` property that is the concordance-index of the dataset to the fitted model.
 - `CoxPHFitter` and `AalenAdditiveFitter` no longer have the `data` property. It was an _almost_ duplicate of the training data, but was causing the model to be very large when serialized.
 - Implements a new fitter `CoxTimeVaryingFitter` available under the `lifelines` namespace. This model implements the Cox model for time-varying covariates.
 - Utils for creating time varying datasets available in `utils`.
 - less noisy check for complete separation.
 - removed `datasets` namespace from the main `lifelines` namespace
 - `CoxPHFitter` has a slightly more intelligent (barely...) way to pick a step size, so convergence should generally be faster.
 - `CoxPHFitter.fit` now has accepts a `weight_col` kwarg so one can pass in weights per observation. This is very useful if you have many subjects, and the space of covariates is not large. Thus you can group the same subjects together and give that observation a weight equal to the count. Altogether, this means a much faster regression.

#### 0.12.0
 - removes  `include_likelihood` from `CoxPHFitter.fit` - it was not slowing things down much (empirically), and often I wanted it for debugging (I suppose others do too). It's also another exit condition, so we many exit from the NR iterations faster.
 - added `step_size` param to `CoxPHFitter.fit` - the default is good, but for extremely large or small datasets this may want to be set manually.
 - added a warning to `CoxPHFitter` to check for complete seperation: https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/
 - Additional functionality to `utils.survival_table_from_events` to bin the index to make the resulting table more readable.

#### 0.11.3
 - No longer support matplotlib 1.X
 - Adding `times` argument to `CoxPHFitter`'s `predict_survival_function` and `predict_cumulative_hazard` to predict the estimates at, instead uses the default times of observation or censorship.
 - More accurate prediction methods parametrics univariate models.

#### 0.11.2
 - Changing liscense to valilla MIT.
 - Speed up `NelsonAalenFitter.fit` considerably.

#### 0.11.1 - 2017-06-22
 - Python3 fix for `CoxPHFitter.plot`.

#### 0.11.0 - 2017-06-21
 - fixes regression in `KaplanMeierFitter.plot` when using Seaborn and lifelines.
 - introduce a new `.plot` function to a fitted `CoxPHFitter` instance. This plots the hazard coefficients and their confidence intervals.
 - in all plot methods, the `ix` kwarg has been deprecated in favour of a new `loc` kwarg. This is to align with Pandas deprecating `ix`

#### 0.10.1 - 2017-06-05
 - fix in internal normalization for `CoxPHFitter` predict methods.

#### 0.10.0
 - corrected bug that was returning the wrong baseline survival and hazard values in `CoxPHFitter` when `normalize=True`.
 - removed  `normalize` kwarg in `CoxPHFitter`. This was causing lots of confusion for users, and added code complexity. It's really nice to be able to remove it.
 - correcting column name in `CoxPHFitter.baseline_survival_`
 - `CoxPHFitter.baseline_cumulative_hazard_` is always centered, to mimic R's `basehaz` API.
 - new `predict_log_partial_hazards` to `CoxPHFitter`

#### 0.9.4
 - adding `plot_loglogs` to `KaplanMeierFitter`
 - added a (correct) check to see if some columns in a dataset will cause convergence problems.
 - removing `flat` argument in `plot` methods. It was causing confusion. To replicate it, one can set `ci_force_lines=True` and `show_censors=True`.
 - adding `strata` keyword argument to `CoxPHFitter` on initialization (ex: `CoxPHFitter(strata=['v1', 'v2'])`. Why? Fitters initialized with `strata` can now be passed into `k_fold_cross_validation`, plus it makes unit testing `strata` fitters easier.
 - If using `strata` in `CoxPHFitter`, access to strata specific baseline hazards and survival functions are available (previously it was a blended valie). Prediction also uses the specific baseline hazards/survivals.
 - performance improvements in `CoxPHFitter` - should see at least a 10% speed improvement in `fit`.

#### 0.9.2
 - deprecates Pandas versions before 0.18.
 - throw an error if no admissable pairs in the c-index calculation. Previously a NaN was returned.

#### 0.9.1
 - add two summary functions to Weibull and Exponential fitter, solves #224

#### 0.9.0
 - new prediction function in `CoxPHFitter`, `predict_log_hazard_relative_to_mean`, that mimics what R's `predict.coxph` does.
 - removing the `predict` method in CoxPHFitter and AalenAdditiveFitter. This is because the choice of `predict_median` as a default was causing too much confusion, and no other natual choice as a default was available. All other `predict_` methods remain.
 - Default predict method in `k_fold_cross_validation` is now `predict_expectation`

#### 0.8.1 - 2015-08-01
 - supports matplotlib 1.5.
 - introduction of a param `nn_cumulative_hazards` in AalenAdditiveModel's `__init__` (default True). This parameter will truncate all non-negative cumulative hazards in prediction methods to 0.
 - bug fixes including:
    - fixed issue where the while loop in `_newton_rhaphson` would break too early causing a variable not to be set properly.
    - scaling of smooth hazards in NelsonAalenFitter was off by a factor of 0.5.


#### 0.8.0
 - reorganized lifelines directories:
    - moved test files out of main directory.
    - moved `utils.py` into it's own directory.
    - moved all estimators `fitters` directory.
 - added a `at_risk` column to the output of `group_survival_table_from_events` and `survival_table_from_events`
 - added sample size and power calculations for statistical tests. See `lifeline.statistics. sample_size_necessary_under_cph` and `lifelines.statistics. power_under_cph`.
 - fixed a bug when using KaplanMeierFitter for left-censored data.


#### 0.7.1
- addition of a l2 `penalizer` to `CoxPHFitter`.
- dropped Fortran implementation of efficient Python version. Lifelines is pure python once again!
- addition of `strata` keyword argument to `CoxPHFitter` to allow for stratification of a single or set of
categorical variables in your dataset.
- `datetimes_to_durations` now accepts a list as `na_values`, so multiple values can be checked.
- fixed a bug in `datetimes_to_durations` where `fill_date` was not properly being applied.
- Changed warning in `datetimes_to_durations` to be correct.
- refactor each fitter into it's own submodule. For now, the tests are still in the same file. This will also *not* break the API.


#### 0.7.0 - 2015-03-01
- allow for multiple fitters to be passed into `k_fold_cross_validation`.
- statistical tests in `lifelines.statistics`. now return a `StatisticalResult` object with properties like `p_value`, `test_results`, and `summary`.
- fixed a bug in how log-rank statistical tests are performed. The covariance matrix was not being correctly calculated. This resulted in slightly different p-values.
- `WeibullFitter`, `ExponentialFitter`, `KaplanMeierFitter` and `BreslowFlemingHarringtonFitter` all have a `conditional_time_to_event_` property that measures  the median duration remaining until the death event, given survival up until time t.

#### 0.6.1

- addition of `median_` property to `WeibullFitter` and `ExponentialFitter`.
- `WeibullFitter` and `ExponentialFitter` will use integer timelines instead of float provided by `linspace`. This is
so if your work is to sum up the survival function (for expected values or something similar), it's more difficult to
make a mistake.

#### 0.6.0 - 2015-02-04

- Inclusion of the univariate fitters `WeibullFitter` and `ExponentialFitter`.
- Removing `BayesianFitter` from lifelines.
- Added new penalization scheme to AalenAdditiveFitter. You can now add a smoothing penalizer
that will try to keep subsequent values of a hazard curve close together. The penalizing coefficient
is `smoothing_penalizer`.
- Changed `penalizer` keyword arg to `coef_penalizer` in AalenAdditiveFitter.
- new `ridge_regression` function in `utils.py` to perform linear regression with l2 penalizer terms.
- Matplotlib is no longer a mandatory dependency.
- `.predict(time)` method on univariate fitters can now accept a scalar (and returns a scalar) and an iterable (and returns a numpy array)
- In `KaplanMeierFitter`, `epsilon` has been renamed to `precision`.


#### 0.5.1 - 2014-12-24

- New API for `CoxPHFitter` and `AalenAdditiveFitter`: the default arguments for `event_col` and `duration_col`. `duration_col` is now mandatory, and `event_col` now accepts a column, or by default, `None`, which assumes all events are observed (non-censored).
- Fix statistical tests.
- Allow negative durations in Fitters.
- New API in `survival_table_from_events`: `min_observations` is replaced by `birth_times` (default `None`).
- New API in `CoxPHFitter` for summary: `summary` will return a dataframe with statistics, `print_summary()` will print the dataframe (plus some other statistics) in a pretty manner.
- Adding "At Risk" counts option to univariate fitter `plot` methods, `.plot(at_risk_counts=True)`, and the function `lifelines.plotting.add_at_risk_counts`.
- Fix bug Epanechnikov kernel.

#### 0.5.0 - 2014-12-07

- move testing to py.test
- refactor tests into smaller files
- make `test_pairwise_logrank_test_with_identical_data_returns_inconclusive` a better test
- add test for summary()
- Alternate metrics can be used for `k_fold_cross_validation`.


#### 0.4.4 - 2014-11-27

 - Lots of improvements to numerical stability (but something things still need work)
 - Additions to `summary` in CoxPHFitter.
 - Make all prediction methods output a DataFrame
 - Fixes bug in 1-d input not returning in CoxPHFitter
 - Lots of new tests.

#### 0.4.3 - 2014-07-23
 - refactoring of `qth_survival_times`: it can now accept an iterable (or a scalar still) of probabilities in the q argument, and will return a DataFrame with these as columns. If len(q)==1 and a single survival function is given, will return a scalar, not a DataFrame. Also some good speed improvements.
 - KaplanMeierFitter and NelsonAalenFitter now have a `_label` property that is passed in during the fit.
 - KaplanMeierFitter/NelsonAalenFitter's inital `alpha` value is overwritten if a new `alpha` value is passed
 in during the `fit`.
 - New method for KaplanMeierFitter: `conditional_time_to`. This returns a DataFrame of the estimate:
    med(S(t | T>s)) - s, human readable: the estimated time left of living, given an individual is aged s.
- Adds option `include_likelihood` to CoxPHFitter fit method to save the final log-likelihood value.

#### 0.4.2 - 2014-06-19

 - Massive speed improvements to CoxPHFitter.
 - Additional prediction method: `predict_percentile` is available on CoxPHFitter and AalenAdditiveFitter. Given a percentile, p, this function returns the value t such that *S(t | x) = p*. It is a generalization of `predict_median`.
 - Additional kwargs in `k_fold_cross_validation` that will accept different prediction methods (default is `predict_median`).
- Bug fix in CoxPHFitter `predict_expectation` function.
- Correct spelling mistake in newton-rhapson algorithm.
- `datasets` now contains functions for generating the respective datasets, ex: `generate_waltons_dataset`.
- Bumping up the number of samples in statistical tests to prevent them from failing so often (this a stop-gap)
- pep8 everything

#### 0.4.1.1

- Ability to specify default printing in statistical tests with the `suppress_print` keyword argument (default False).
- For the multivariate log rank test, the inverse step has been replaced with the generalized inverse. This seems to be what other packages use.
- Adding more robust cross validation scheme based on issue #67.
- fixing `regression_dataset` in `datasets`.


#### 0.4.1 - 2014-06-11

 - `CoxFitter` is now known as `CoxPHFitter`
 - refactoring some tests that used redundant data from `lifelines.datasets`.
 - Adding cross validation: in `utils` is a new `k_fold_cross_validation` for model selection in regression problems.
 - Change CoxPHFitter's fit method's `display_output` to `False`.
 - fixing bug in CoxPHFitter's `_compute_baseline_hazard` that errored when sending Series objects to
   `survival_table_from_events`.
 - CoxPHFitter's `fit` now looks to columns with too low variance, and halts NR algorithm if a NaN is found.
 - Adding a Changelog.
 - more sanitizing for the statistical tests =)

#### 0.4.0 - 2014-06-08

 - `CoxFitter` implements Cox Proportional Hazards model in lifelines.
 - lifelines moves the wheels distributions.
 - tests in the `statistics` module now prints the summary (and still return the regular values)
 - new `BaseFitter` class is inherited from all fitters.
