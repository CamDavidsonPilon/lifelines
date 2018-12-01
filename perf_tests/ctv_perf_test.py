if __name__ == "__main__":
    import time
    import pandas as pd
    from lifelines.estimation import CoxTimeVaryingFitter
    from lifelines.datasets import load_stanford_heart_transplants

    dfcv = load_stanford_heart_transplants()
    dfcv = pd.concat([dfcv] * 50)
    ctv = CoxTimeVaryingFitter()
    start_time = time.time()
    ctv.fit(
        dfcv,
        id_col="id",
        event_col="event",
        start_col="start",
        stop_col="stop",
    )
    time_took = time.time() - start_time
    print("--- %s seconds ---" % time_took)
    ctv.print_summary()
