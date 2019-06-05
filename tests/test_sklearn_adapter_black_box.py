# -*- coding: utf-8 -*-

import unittest
import numpy as np

from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.datasets import load_regression_dataset, load_rossi
from lifelines.utils import k_fold_cross_validation
from lifelines.utils.sklearn_adapter import *
from sklearn.model_selection import cross_validate, cross_val_predict
import numpy as np
from scipy.stats import ttest_ind


class TestSKLearnAdapter(unittest.TestCase):
    def test_Cox_cv_black_box(self):
        rossi = load_rossi()
        cph = CoxPHFitter()
        folds = 10
        resOwnCV = np.array(k_fold_cross_validation(cph, rossi, "week", event_col="arrest", k=folds))
        print("Own CV: ", resOwnCV)

        f = LifelinesSKLearnAdapter(CoxPHFitter, call_params={"duration_col": "week", "event_col": "arrest"})
        sklearnCV = cross_validate(f, rossi, cv=folds)["test_score"]
        tRes = ttest_ind(resOwnCV, sklearnCV)

        print("sklearn CV: ", sklearnCV)
        print("t-test:", tRes)
        self.assertGreaterEqual(tRes.pvalue, 0.25)

    
    def test_cv_predict(self):
        regression_dataset = load_regression_dataset()
        for fitter in (CoxPHFitter, WeibullAFTFitter):
            with self.subTest(fitter=fitter):
                cph = fitter()
                folds = 10
                f = LifelinesSKLearnAdapter(cph, call_params={"duration_col": "T", "event_col": "E"})
                res = cross_val_predict(f, regression_dataset, cv=folds)
                print(fitter.__name__, res)


if __name__ == "__main__":
    unittest.main()
