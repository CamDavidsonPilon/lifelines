# -*- coding: utf-8 -*-
import numpy as np
import scipy
from matplotlib import pyplot as plt
from lifelines import WeibullFitter, KaplanMeierFitter, LogNormalFitter, LogLogisticFitter
from lifelines.plotting import left_censorship_cdf_plot, qq_plot

plt.style.use("bmh")


N = 2500

T_actual = scipy.stats.fisk(8, 0, 1).rvs(N)

MIN_0 = np.percentile(T_actual, 5)
MIN_1 = np.percentile(T_actual, 10)
MIN_2 = np.percentile(T_actual, 30)
MIN_3 = np.percentile(T_actual, 50)

T = T_actual.copy()
ix = np.random.randint(4, size=N)

T = np.where(ix == 0, np.maximum(T, MIN_0), T)
T = np.where(ix == 1, np.maximum(T, MIN_1), T)
T = np.where(ix == 2, np.maximum(T, MIN_2), T)
T = np.where(ix == 3, np.maximum(T, MIN_3), T)
E = T_actual == T

fig, axes = plt.subplots(2, 2, figsize=(9, 5))
axes = axes.reshape(4)

for i, model in enumerate([WeibullFitter(), KaplanMeierFitter(), LogNormalFitter(), LogLogisticFitter()]):
    if isinstance(model, KaplanMeierFitter):
        model.fit(T, E, left_censorship=True, label=model.__class__.__name__)
    else:
        model.fit(T, E, left_censorship=True, label=model.__class__.__name__)

    model.plot_cumulative_density(ax=axes[i])
plt.tight_layout()

for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter()]):
    model.fit(T, E, left_censorship=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    left_censorship_cdf_plot(model, ax=axes[0])
    qq_plot(model, ax=axes[1])


plt.show()
