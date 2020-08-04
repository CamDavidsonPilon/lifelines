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
    # df['s'] = "a"
    df = pd.concat([df] * reps)
    print(df.shape)

    cph = CoxPHFitter(strata=["wexp"])
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=True)
    cph.print_summary()
    r = cph.baseline_survival_
    print(r)
    print("--- %s seconds ---" % (time.time() - start_time))
