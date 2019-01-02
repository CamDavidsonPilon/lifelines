# -*- coding: utf-8 -*-
if __name__ == "__main__":
    import time
    import pandas as pd
    from lifelines import CoxTimeVaryingFitter
    from lifelines.datasets import load_rossi
    from lifelines.utils import to_long_format

    df = load_rossi()
    df = pd.concat([df] * 20)
    df = df.reset_index()
    df = to_long_format(df, duration_col='week')
    ctv = CoxTimeVaryingFitter()
    start_time = time.time()
    ctv.fit(df, id_col="index", event_col="arrest", start_col="start", stop_col="stop", strata=['wexp', 'prio'])
    time_took = time.time() - start_time
    print("--- %s seconds ---" % time_took)
    ctv.print_summary()
