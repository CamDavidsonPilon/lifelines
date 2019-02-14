# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 20)
    # df = df.reset_index()
    # df['week'] = np.random.exponential(1, size=df.shape[0])
    cp = CoxPHFitter()
    cp.fit(df, duration_col="week", event_col="arrest", batch_mode=True)
    start_time = time.time()
    print(cp.predict_median(df))
    print("--- %s seconds ---" % (time.time() - start_time))
    cp.print_summary(4)
