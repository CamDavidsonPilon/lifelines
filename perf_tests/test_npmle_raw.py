# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines.fitters.npmle import npmle
    from lifelines.datasets import load_diabetes

    df = pd.read_csv("mice.csv", index_col=[0])

    reps = 1
    # df = load_diabetes()
    # df = pd.concat([df] * reps)
    left, right = df["l"], df["u"]
    start_time = time.time()
    npmle(left, right, verbose=False)
    print("--- %s seconds ---" % (time.time() - start_time))
