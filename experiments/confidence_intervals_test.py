# -*- coding: utf-8 -*-
import numpy as np
from lifelines import WeibullFitter

lambda_, rho_ = 2, 0.5
N = 10000

T_actual = lambda_ * np.random.exponential(1, size=N) ** (1 / rho_)
T_censor = lambda_ * np.random.exponential(1, size=N) ** (1 / rho_)
T = np.minimum(T_actual, T_censor)
E = T_actual < T_censor

time = [1.0]

# lifelines computed confidence interval
print(wf.fit(T, E, timeline=time).confidence_interval_cumulative_hazard_)


bootstrap_samples = 10000
results = []

for _ in range(bootstrap_samples):
    ix = np.random.randint(0, 10000, 10000)
    wf = WeibullFitter().fit(T[ix], E[ix], timeline=time)
    results.append(wf.cumulative_hazard_at_times(time).values[0])
    print(np.percentile(results, [2.5, 97.5]))
