# -*- coding: utf-8 -*-
# aalen additive


if __name__ == "__main__":
    import pandas as pd
    import time

    from lifelines import AalenAdditiveFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 5).reset_index(drop=True)
    print("Size: ", df.shape)
    aaf = AalenAdditiveFitter()
    start_time = time.time()
    aaf.fit(df, duration_col="week", event_col="arrest")
    print("--- %s seconds ---" % (time.time() - start_time))
    print(aaf.score_)
