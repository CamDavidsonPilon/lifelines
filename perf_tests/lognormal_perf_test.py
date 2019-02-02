# -*- coding: utf-8 -*-
# aalen additive


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time

    from lifelines import LogNormalFitter

    np.random.seed(1)
    N = 250000
    mu = 3 * np.random.randn()
    sigma = np.random.uniform(0.1, 3.0)

    X, C = np.exp(sigma * np.random.randn(N) + mu), np.exp(np.random.randn(N) + mu)
    E = X <= C
    T = np.minimum(X, C)

    lnf = LogNormalFitter()
    start_time = time.time()
    lnf.fit(T, E)
    print("--- %s seconds ---" % (time.time() - start_time))
    lnf.print_summary(5)
