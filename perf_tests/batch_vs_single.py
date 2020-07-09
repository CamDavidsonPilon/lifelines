# -*- coding: utf-8 -*-
from time import time
from itertools import product
import pandas as pd
import numpy as np
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
import statsmodels.api as sm

# This compares the batch algorithm (in CTV) vs the single iteration algorithm (original in CPH)
# N vs (% ties == unique(T) / N)


ROSSI = load_rossi()
ROSSI_ROWS = ROSSI.shape[0]
results = {}


for n_copies, additional_x_vars, fraction in product(
    [1, 3, 6, 10, 50, 500, 2500], [0, 10, 20], np.logspace(-6, np.log10(0.99), 8)
):
    try:
        print(n_copies, additional_x_vars, fraction)

        n = n_copies * ROSSI_ROWS
        df = pd.concat([ROSSI] * n_copies)
        n_unique_durations = int(df.shape[0] * fraction) + 1
        unique_durations = np.round(np.random.exponential(10, size=n_unique_durations), 5)

        df["week"] = np.tile(unique_durations, int(np.ceil(1 / fraction)))[: df.shape[0]]

        for i in range(additional_x_vars):
            df["%i" % i] = np.random.randn(n)

        batch_results = []
        for _ in range(3):
            cph_batch = CoxPHFitter()
            start_time = time()
            cph_batch.fit(df, "week", "arrest", batch_mode=True)
            batch_results.append(time() - start_time)

        single_results = []
        for _ in range(3):
            cph_single = CoxPHFitter()
            start_time = time()
            cph_single.fit(df, "week", "arrest", batch_mode=False)
            single_results.append(time() - start_time)

        batch_time = min(batch_results)
        single_time = min(single_results)
        print({"batch": batch_time, "single": single_time})
        results[(n, fraction, df.shape[1] - 2)] = {"batch": batch_time, "single": single_time}
    except KeyboardInterrupt:
        break

results = pd.DataFrame(results).T.sort_index()
results = results.reset_index()
results = results.rename(columns={"level_0": "N", "level_1": "frac", "level_2": "N_vars"})
results["ratio"] = results["batch"] / results["single"]

print(results)
results.to_csv("perf_results.csv", index=False)


results["N * frac"] = results["N"] * results["frac"]
results["N**2"] = results["N"] ** 2
results["frac**2"] = results["frac"] ** 2
results["N_vars * N"] = results["N"] * results["N_vars"]

X = results[["N", "frac", "N * frac", "frac**2", "N**2", "N_vars", "N_vars * N"]]
X = sm.add_constant(X)

Y = results["ratio"]


model = sm.OLS(Y, X).fit()
print(model.summary())
print(model.params)
