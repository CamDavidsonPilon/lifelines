# -*- coding: utf-8 -*-
from time import time
from itertools import product
import pandas as pd
import numpy as np
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
import statsmodels.api as sm

# This compares the batch algorithm vs the single iteration algorithm
# N vs (% ties == unique(T) / N)


REPLICATES = 1
ROSSI = load_rossi()
ROSSI_ROWS = ROSSI.shape[0]
results = {}


for n_copies, additional_x_vars, fraction in product([25000], [0], np.logspace(-2, np.log10(0.99), 2)):
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
        for _ in range(REPLICATES):
            cph_batch = CoxPHFitter()
            start_time = time()
            cph_batch.fit(df, "week", "arrest", batch_mode=True)
            batch_results.append(time() - start_time)

        single_results = []
        for _ in range(REPLICATES):
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


# appending!
results.to_csv("batch_vs_single_perf_results.csv", index=False, mode="a", header=False)

# write from scratch!
# results.to_csv("batch_vs_single_perf_results.csv", index=False)

results = pd.read_csv("batch_vs_single_perf_results.csv")

results["log_frac"] = np.log(results["frac"])
results["N * log_frac"] = results["N"] * results["log_frac"]
results["N**2"] = results["N"] ** 2
results["log_frac**2"] = results["log_frac"] ** 2
results["N_vars * N"] = results["N"] * results["N_vars"]

X = results[["N", "log_frac", "N * log_frac", "log_frac**2", "N**2", "N_vars", "N_vars * N"]]
X = sm.add_constant(X)

Y = results["ratio"]


model = sm.OLS(Y, X).fit()
print(model.summary())
print(model.params)
