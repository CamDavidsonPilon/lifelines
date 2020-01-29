# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi, load_regression_dataset

    reps = 100
    df = load_rossi()
    df = pd.concat([df] * reps)
    cp = CoxPHFitter(penalizer=0.0, l1_ratio=0, baseline_estimation_method="spline")
    start_time = time.time()
    cp.fit(df, duration_col="week", event_col="arrest")
    print("--- %s seconds ---" % (time.time() - start_time))
    cp.print_summary(2)
    print(cp._ll_null_)
