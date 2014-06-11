### Changelogs

####0.4.2

- Ability to specify default printing in statsitical tests with the `suppress_print` keyword argument (default False).
- For the multivariate log rank test, the inverse step has been replaced with the generalized inverse. This seems to be what other packages use.


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