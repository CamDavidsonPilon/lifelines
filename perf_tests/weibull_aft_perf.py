# -*- coding: utf-8 -*-
# weibull aft


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines import WeibullAFTFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df = pd.concat([df] * 1)

    # df["start"] = df["week"]
    # df["stop"] = np.where(df["arrest"], df["start"], np.inf)
    # df = df.drop("week", axis=1)

    wp = WeibullAFTFitter()
    start_time = time.time()
    wp.fit_right_censoring(df, "week", event_col="arrest")
    print("--- %s seconds ---" % (time.time() - start_time))
    wp.print_summary()
    print(wp.score(df, scoring_method="log_likelihood"))
    print(wp.score(df, scoring_method="concordance_index"))
