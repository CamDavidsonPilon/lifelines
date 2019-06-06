# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.datasets import load_rossi

rossi = load_rossi()

Y = rossi.pop("week").values
X = rossi

base_class = sklearn_adapter(CoxPHFitter, duration_col="week", event_col="arrest")
cph = base_class(penalizer=1.0)

cph.fit(X, Y)

scores = cross_val_score(cph, X, Y, cv=5)
print(scores)

base_model = sklearn_adapter(WeibullAFTFitter, duration_col="week", event_col="arrest")

clf = GridSearchCV(base_model(), {"penalizer": [0, 1, 10], "model_ancillary": [True, False]}, cv=5)
clf.fit(X, Y)
print(clf)
