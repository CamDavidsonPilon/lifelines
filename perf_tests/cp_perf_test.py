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
    cph = CoxPHFitter()
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    cph.print_summary(2)
    print(cph.compute_followup_hazard_ratios(df, [15, 20, 30, 40, 50, 52]))
    print(cph.hazard_ratios_)
    cph.compute_followup_hazard_ratios(df, [15, 20, 30, 40, 50, 52]).plot()
