# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    reps = 1
    df = load_rossi()
    df = pd.concat([df] * reps)
    cp = CoxPHFitter(penalizer=0.05, l1_ratio=1.0)
    start_time = time.time()
    cp.fit(df, duration_col="week", event_col="arrest", batch_mode=True, show_progress=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    cp.print_summary(8)
