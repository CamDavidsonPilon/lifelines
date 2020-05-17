# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy as np

    from lifelines.fitters.npmle import npmle
    from lifelines.datasets import load_diabetes

    reps = 1
    df = load_diabetes()
    df = pd.concat([df] * reps)
    left, right = df["left"], df["right"]
    start_time = time.time()
    npmle(left, right)  # verbose=True)
    print("--- %s seconds ---" % (time.time() - start_time))
