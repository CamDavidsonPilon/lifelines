# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time

    from lifelines import WeibullFitter

    data = (
        [{"start": 0, "stop": 2, "E": False}] * (1000 - 376)
        + [{"start": 2, "stop": 5, "E": False}] * (376 - 82)
        + [{"start": 5, "stop": 10, "E": False}] * (82 - 7)
        + [{"start": 10, "stop": 1e10, "E": False}] * 7
    )

    df = pd.DataFrame.from_records(data)
    print(df)

    df = df.groupby(["start", "stop", "E"]).size().reset_index()
    print(df)

    wb = WeibullFitter()
    start_time = time.time()
    wb.fit_interval_censoring(df["start"], df["stop"], df["E"], weights=df[0])
    print("--- %s seconds ---" % (time.time() - start_time))
    wb.print_summary(5)
