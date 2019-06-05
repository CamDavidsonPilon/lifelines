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
print(cph.predict(X))
print(cph.score(X, Y))

scores = cross_val_score(cph, X, Y, cv=5)
print(scores)


clf = GridSearchCV(cph, {"penalizer": [0, 1, 10]}, cv=5)
clf.fit(X, Y)
print(clf)
