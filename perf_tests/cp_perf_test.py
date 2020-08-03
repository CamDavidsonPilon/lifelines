# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi, load_regression_dataset

    reps = 22222
    df = load_rossi()
    df = pd.concat([df] * reps)
    print(df.shape)

    cph = CoxPHFitter()
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=True)
    # cph.print_summary()
    print("--- %s seconds ---" % (time.time() - start_time))

    """
    cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=2)
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=True)
    cph.print_summary()
    print("--- %s seconds ---" % (time.time() - start_time))
    """
