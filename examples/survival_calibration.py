# -*- coding: utf-8 -*-
"""
Smoothed calibration curves for time-to-event models

https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8570

"""


def ccl(p):
    return np.log(-np.log(1 - p))


from lifelines.datasets import load_regression_dataset

df = load_regression_dataset()
T = "T"
E = "E"

T_0 = 15


# fit original model and make survival predictions
cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=4).fit(df, T, E)
predictions_at_T_0 = 1 - cph.predict_survival_function(df, times=[T_0]).T.squeeze()
cll_predictions_at_T_0 = ccl(predictions_at_T_0)

# create new dataset with the predictions
prediction_df = pd.DataFrame({"ccl_at_%d" % T_0: cll_predictions_at_T_0, "constant": 1, "week": df[T], "arrest": df[E]})

# fit new dataset to flexible spline model
# this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!

regressors = {
    "beta_": ["ccl_at_%d" % T_0],
    "gamma0_": ["constant"],
    "gamma1_": ["constant"],
    "gamma2_": ["constant"],
    "gamma3_": ["constant"],
}
# this model is from examples/royson_crowther_clements_splines.py
crc = CRCSplineFitter(4).fit(prediction_df, "week", "arrest", regressors=regressors)

# predict new model at values 0 to 1, but remember to ccl it!
x = np.linspace(np.clip(predictions_at_T_0.min() - 0.05, 0, 1), np.clip(predictions_at_T_0.max() + 0.05, 0, 1), 100)
y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_%d" % T_0: ccl(x), "constant": 1}), times=[T_0]).T.squeeze()


# plot our results
fig, ax = plt.subplots()
ax.set_title("Smoothed calibration curve of predicted vs observed probabilities")
plt.ylim(x[0], x[-1])
plt.xlim(x[0], x[-1])


color = "tab:red"
ax.plot(x, y, label="smoothed calibration curve", color=color)
ax.set_xlabel("Predicted probability of \nt=%d mortality" % T_0)
ax.set_ylabel("Observed probability of \nt=%d mortality" % T_0, color=color)
ax.tick_params(axis="y", labelcolor=color)

# plot x=y line
ax.plot(x, x, c="k", ls="--")


# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax.twinx()
twin_ax.set_ylabel("histogram of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)


twin_ax.hist(predictions_at_T_0, alpha=0.3, bins="sqrt", color=color)

ax.legend()
plt.tight_layout()
