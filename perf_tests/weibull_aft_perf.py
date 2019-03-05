# -*- coding: utf-8 -*-
# weibull aft


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import WeibullAFTFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 500)
    # df = df.reset_index()
    # df['week'] = np.random.exponential(1, size=df.shape[0])
    wp = WeibullAFTFitter()
    start_time = time.time()
    wp.fit(df, duration_col="week", event_col="arrest")
    print("--- %s seconds ---" % (time.time() - start_time))
