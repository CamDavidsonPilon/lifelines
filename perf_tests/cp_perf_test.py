# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 40)
    # df['week'] = np.random.exponential(1, size=df.shape[0])
    cp = CoxPHFitter()
    start_time = time.time()
    cp.fit(df, duration_col="week", event_col="arrest", batch_mode=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    cp.print_summary()
