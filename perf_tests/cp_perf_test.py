# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi, load_regression_dataset

    reps = 2
    df = load_rossi()
    df = pd.concat([df] * reps)
    print(df.shape)

    cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=2)
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=False)
    cph.print_summary()

    df["entry"] = 0
    cph.fit(df, duration_col="week", event_col="arrest", entry_col="entry", show_progress=False)
    cph.print_summary()

    print("--- %s seconds ---" % (time.time() - start_time))
