# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi, load_regression_dataset

    reps = 250
    df = load_rossi()
    df = pd.concat([df] * reps)
    df.to_csv("rossi_2500.csv", index=False)
    print(df.shape)

    cph = CoxPHFitter(baseline_estimation_method="breslow", n_baseline_knots=2)
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=True)
    print("--- %s seconds ---" % (time.time() - start_time))
