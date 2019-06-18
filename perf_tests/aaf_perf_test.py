# -*- coding: utf-8 -*-
# aalen additive


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time

    from lifelines.fitters.aalen_additive_fitter import AalenAdditiveFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 1)
    # df['week'] = np.random.exponential(size=df.shape[0])
    aaf = AalenAdditiveFitter()
    start_time = time.time()
    aaf.fit(df, duration_col="week", event_col="arrest")
    print("--- %s seconds ---" % (time.time() - start_time))
    aaf.print_summary(5)
