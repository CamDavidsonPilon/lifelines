from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from lifelines.datasets import load_rossi

df = load_rossi()

# Specifying the position of the knots: One week, one month and one year.
my_knots = [1, 5, 52]
cph = CoxPHFitter(baseline_estimation_method="spline", knots=my_knots)
cph.fit(df, 'week', 'arrest', strata=['wexp'])
cph.print_summary()
