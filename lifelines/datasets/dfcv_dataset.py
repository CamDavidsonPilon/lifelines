# -*- coding: utf-8 -*-
import pandas as pd
from lifelines.utils import add_covariate_to_timeline
from lifelines.utils import to_long_format

df = pd.DataFrame(
    [
        [1, 3, True, 1],
        [6, 4, False, 0],
        [3, 5, True, 1],
        [2, 5, False, 1],
        [4, 6, True, 1],
        [7, 7, True, 0],
        [8, 8, False, 0],
        [5, 8, False, 1],
        [9, 9, True, 0],
        [10, 10, True, 0],
    ],
    columns=["id", "time", "event", "group"],
)


df = to_long_format(df, "time")

cv = pd.DataFrame.from_records(
    [
        {"id": 1, "z": 0, "time": 0},
        {"id": 6, "z": 1, "time": 0},
        {"id": 3, "z": 1, "time": 0},
        {"id": 2, "z": 0, "time": 0},
        {"id": 4, "z": 0, "time": 0},
        {"id": 7, "z": 0, "time": 0},
        {"id": 8, "z": 0, "time": 0},
        {"id": 5, "z": 0, "time": 0},
        {"id": 9, "z": 0, "time": 0},
        {"id": 10, "z": 0, "time": 0},
        {"id": 1, "z": 0, "time": 3},
        {"id": 6, "z": 1, "time": 3},
        {"id": 3, "z": 1, "time": 3},
        {"id": 2, "z": 0, "time": 3},
        {"id": 4, "z": 0, "time": 3},
        {"id": 7, "z": 0, "time": 3},
        {"id": 8, "z": 0, "time": 3},
        {"id": 5, "z": 0, "time": 3},
        {"id": 9, "z": 0, "time": 3},
        {"id": 10, "z": 1, "time": 3},
        {"id": 6, "z": 1, "time": 4},
        {"id": 3, "z": 1, "time": 4},
        {"id": 2, "z": 0, "time": 4},
        {"id": 4, "z": 0, "time": 4},
        {"id": 7, "z": 0, "time": 4},
        {"id": 8, "z": 0, "time": 4},
        {"id": 5, "z": 0, "time": 4},
        {"id": 9, "z": 0, "time": 4},
        {"id": 10, "z": 1, "time": 4},
        {"id": 3, "z": 1, "time": 5},
        {"id": 2, "z": 0, "time": 5},
        {"id": 4, "z": 0, "time": 5},
        {"id": 7, "z": 1, "time": 5},
        {"id": 8, "z": 0, "time": 5},
        {"id": 5, "z": 0, "time": 5},
        {"id": 9, "z": 1, "time": 5},
        {"id": 10, "z": 1, "time": 5},
        {"id": 4, "z": 0, "time": 6},
        {"id": 7, "z": 1, "time": 6},
        {"id": 8, "z": 0, "time": 6},
        {"id": 5, "z": 1, "time": 6},
        {"id": 9, "z": 1, "time": 6},
        {"id": 10, "z": 1, "time": 6},
        {"id": 7, "z": 1, "time": 7},
        {"id": 8, "z": 0, "time": 7},
        {"id": 5, "z": 1, "time": 7},
        {"id": 9, "z": 1, "time": 7},
        {"id": 10, "z": 1, "time": 7},
        {"id": 8, "z": 0, "time": 8},
        {"id": 5, "z": 1, "time": 8},
        {"id": 9, "z": 1, "time": 8},
        {"id": 10, "z": 1, "time": 8},
        {"id": 9, "z": 1, "time": 9},
        {"id": 10, "z": 1, "time": 9},
    ]
)

dfcv = add_covariate_to_timeline(df, cv, "id", "time", "event", add_enum=False)
