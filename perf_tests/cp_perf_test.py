# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time

    from lifelines.fitters.coxph_fitter import CoxPHFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 20)
    cp = CoxPHFitter()
    start_time = time.time()
    cp.fit(df, duration_col="week", event_col="arrest")
    print("--- %s seconds ---" % (time.time() - start_time))
