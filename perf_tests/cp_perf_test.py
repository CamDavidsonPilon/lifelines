# -*- coding: utf-8 -*-
# cox regression


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 1)
    cp = CoxPHFitter()
    start_time = time.time()
    cp.fit(df, duration_col="week", event_col="arrest", batch_mode=True, show_progress=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    cp.print_summary()
    print(cp.path)
