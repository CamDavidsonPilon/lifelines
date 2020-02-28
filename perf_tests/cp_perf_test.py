# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi, load_regression_dataset

    reps = 1
    df = load_rossi()
    df = pd.concat([df] * reps)
    cp_breslow = CoxPHFitter(penalizer=0.01, l1_ratio=0.0, baseline_estimation_method="spline")
    start_time = time.time()
    cp_breslow.fit(df, duration_col="week", event_col="arrest", show_progress=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    cp_breslow.print_summary(2)
    print(cp_breslow.score(df))
    print(cp_breslow.score(df, scoring_method="concordance_index"))
