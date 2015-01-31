### Changelogs

#### 0.6.0

- Added new penalization scheme to AalenAdditiveFitter. You can now add a smoothing penalizer
that will try to keep subsequent values of a hazard curve close together. The penalizing coefficient
is `smoothing_penalizer`. 
- Changed `penalizer` keyword arg to `coef_penalizer` in AalenAdditiveFitter.
- new `ridge_regression` function in `utils.py` to perform linear regression with l2 penalizer terms.
- Matplotlib is no longer a mandatory dependency. 
- `.predict(time)` method on univariate fitters can now accept a scalar (and returns a scalar) and an iterable (and returns a numpy array)

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
