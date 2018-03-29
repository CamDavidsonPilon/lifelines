### Changelogs

#### 0.15.0
 - fixed bug with using weights and strata in `CoxPHFitter`
 - Performance optimizations in `CoxPHFitter` for up to 40% faster completion of `fit`.
    - even smarter `step_size` calculations for iterative optimizations. 
    - simple code optimizations & cleanup in specific hot spots.
 - Performance optimizations in `AalenAdditiveFitter` for up to 50% faster completion of `fit` for large dataframes, and up to 10% faster for small dataframes. 


#### 0.14.0
 - adding `plot_covariate_groups` to `CoxPHFitter` to visualize what happens to survival as we vary a covariate, all else being equal.
 - `utils` functions like `qth_survival_times` and `median_survival_times` now return the transpose of the DataFrame compared to previous version of lifelines. The reason for this is that we often treat survival curves as columns in DataFrames, and functions of the survival curve as index (ex: KaplanMeierFitter.survival_function_ returns a survival curve _at_ time _t_).
 - `KaplanMeierFitter.fit` and `NelsonAalenFitter.fit` accept a `weights` vector that can be used for pre-aggregated datasets. See this [issue](https://github.com/CamDavidsonPilon/lifelines/issues/396).
 - Convergence errors now return a custom `ConvergenceWarning` instead of a `RuntimeWarning`
 - New checks for complete separation in the dataset for regressions.

#### 0.13.0
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

#### 0.11.1
 - Python3 fix for `CoxPHFitter.plot`.

#### 0.11.0
 - fixes regression in `KaplanMeierFitter.plot` when using Seaborn and lifelines.
 - introduce a new `.plot` function to a fitted `CoxPHFitter` instance. This plots the hazard coefficients and their confidence intervals. 
 - in all plot methods, the `ix` kwarg has been deprecated in favour of a new `loc` kwarg. This is to align with Pandas deprecating `ix`

#### 0.10.1
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

#### 0.8.1
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


#### 0.7.0
- allow for multiple fitters to be passed into `k_fold_cross_validation`. 
- statistical tests in `lifelines.statistics`. now return a `StatisticalResult` object with properties like `p_value`, `test_results`, and `summary`.  
- fixed a bug in how log-rank statistical tests are performed. The covariance matrix was not being correctly calculated. This resulted in slightly different p-values. 
- `WeibullFitter`, `ExponentialFitter`, `KaplanMeierFitter` and `BreslowFlemingHarringtonFitter` all have a `conditional_time_to_event_` property that measures  the median duration remaining until the death event, given survival up until time t.

#### 0.6.1

- addition of `median_` property to `WeibullFitter` and `ExponentialFitter`. 
- `WeibullFitter` and `ExponentialFitter` will use integer timelines instead of float provided by `linspace`. This is 
so if your work is to sum up the survival function (for expected values or something similar), it's more difficult to 
make a mistake. 

#### 0.6.0

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


#### 0.5.1

- New API for `CoxPHFitter` and `AalenAdditiveFitter`: the default arguments for `event_col` and `duration_col`. `duration_col` is now mandatory, and `event_col` now accepts a column, or by default, `None`, which assumes all events are observed (non-censored).
- Fix statistical tests.
- Allow negative durations in Fitters.
- New API in `survival_table_from_events`: `min_observations` is replaced by `birth_times` (default `None`).
- New API in `CoxPHFitter` for summary: `summary` will return a dataframe with statistics, `print_summary()` will print the dataframe (plus some other statistics) in a pretty manner.
- Adding "At Risk" counts option to univariate fitter `plot` methods, `.plot(at_risk_counts=True)`, and the function `lifelines.plotting.add_at_risk_counts`.
- Fix bug Epanechnikov kernel.  

#### 0.5.0

- move testing to py.test
- refactor tests into smaller files
- make `test_pairwise_logrank_test_with_identical_data_returns_inconclusive` a better test
- add test for summary()
- Alternate metrics can be used for `k_fold_cross_validation`.


#### 0.4.4

 - Lots of improvements to numerical stability (but something things still need work)
 - Additions to `summary` in CoxPHFitter.
 - Make all prediction methods output a DataFrame
 - Fixes bug in 1-d input not returning in CoxPHFitter
 - Lots of new tests. 

####0.4.3
 - refactoring of `qth_survival_times`: it can now accept an iterable (or a scalar still) of probabilities in the q argument, and will return a DataFrame with these as columns. If len(q)==1 and a single survival function is given, will return a scalar, not a DataFrame. Also some good speed improvements.
 - KaplanMeierFitter and NelsonAalenFitter now have a `_label` property that is passed in during the fit.
 - KaplanMeierFitter/NelsonAalenFitter's inital `alpha` value is overwritten if a new `alpha` value is passed
 in during the `fit`.
 - New method for KaplanMeierFitter: `conditional_time_to`. This returns a DataFrame of the estimate:
    med(S(t | T>s)) - s, human readable: the estimated time left of living, given an individual is aged s.
- Adds option `include_likelihood` to CoxPHFitter fit method to save the final log-likelihood value.

####0.4.2

 - Massive speed improvements to CoxPHFitter. 
 - Additional prediction method: `predict_percentile` is available on CoxPHFitter and AalenAdditiveFitter. Given a percentile, p, this function returns the value t such that *S(t | x) = p*. It is a generalization of `predict_median`. 
 - Additional kwargs in `k_fold_cross_validation` that will accept different prediction methods (default is `predict_median`). 
- Bug fix in CoxPHFitter `predict_expectation` function. 
- Correct spelling mistake in newton-rhapson algorithm.
- `datasets` now contains functions for generating the respective datasets, ex: `generate_waltons_dataset`.
- Bumping up the number of samples in statistical tests to prevent them from failing so often (this a stop-gap)
- pep8 everything

####0.4.1.1

- Ability to specify default printing in statsitical tests with the `suppress_print` keyword argument (default False).
- For the multivariate log rank test, the inverse step has been replaced with the generalized inverse. This seems to be what other packages use.
- Adding more robust cross validation scheme based on issue #67.
- fixing `regression_dataset` in `datasets`.
 

####0.4.1

 - `CoxFitter` is now known as `CoxPHFitter`
 - refactoring some tests that used redundant data from `lifelines.datasets`. 
 - Adding cross validation: in `utils` is a new `k_fold_cross_validation` for model selection in regression problems.
 - Change CoxPHFitter's fit method's `display_output` to `False`.
 - fixing bug in CoxPHFitter's `_compute_baseline_hazard` that errored when sending Series objects to
   `survival_table_from_events`.
 - CoxPHFitter's `fit` now looks to columns with too low variance, and halts NR algorithm if a NaN is found.
 - Adding a Changelog.
 - more sanitizing for the statistical tests =)

####0.4.0

 - `CoxFitter` implements Cox Proportional Hazards model in lifelines. 
 - lifelines moves the wheels distributions.
 - tests in the `statistics` module now prints the summary (and still return the regular values)
 - new `BaseFitter` class is inherited from all fitters. 
