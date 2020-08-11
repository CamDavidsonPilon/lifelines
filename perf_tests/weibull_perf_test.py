# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time
    from lifelines.datasets import load_rossi

    from lifelines import WeibullFitter

    rossi = load_rossi()
    wb = WeibullFitter()
    start_time = time.time()
    wb.fit_right_censoring(rossi["week"], rossi["arrest"], show_progress=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    wb.print_summary(5)
