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

    cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=2, strata=["wexp"])
    start_time = time.time()
    cph.fit(df, duration_col="week", event_col="arrest", show_progress=True, timeline=np.linspace(1, 60, 100))
    cph.print_summary()
    cph.plot_partial_effects_on_outcome(covariates=["age"], values=np.arange(20, 50, 10))
    print("--- %s seconds ---" % (time.time() - start_time))
