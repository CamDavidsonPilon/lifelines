.. image:: http://i.imgur.com/EOowdSD.png

-------------------------------------


Compatibility with scikit-learn
###################################

New to lifelines in version 0.21.3 is a wrapper that allows you to use lifeline's regression models with scikit-learn's APIs.


.. code:: python

    from lifelines.utils.sklearn_adapter import sklearn_adapter

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    X = load_rossi().drop('week', axis=1)
    Y = load_rossi().pop('week')

    CoxRegression = sklearn_adapter(CoxPHFitter, duration_col='week', event_col='arrest')
    # CoxRegression is like LinearRegression, or SVC class in scikit-learn

    cph = CoxRegression(penalizer=1.0)
    cph.fit(X, Y)
    print(cph)

    """
    CoxPHFitter(alpha=0.05, penalizer=1.0, strata=None, tie_method='Efron')
    """

    cph.predict(X)
    cph.score(X, Y)



The wrapped classes can even be used in more complex scikit-learn functions and classes


.. code:: python

    from sklearn.model_selection import cross_val_score


    base_class = sklearn_adapter(CoxPHFitter, duration_col='week', event_col='arrest')
    cph = base_class(penalizer=1.0)

    scores = cross_val_score(cph, X, Y, cv=5)
    print(scores)

    """
    [0.69253438 0.55414668 0.588 0.64797196 0.52120917]
    """



    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(cph, {'penalizer': [0, 1, 10]}, cv=5)
    clf.fit(X, Y)
    print(clf)

    """
    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=CoxPHFitter(alpha=0.05, penalizer=1.0, strata=None,
                                       tie_method='Efron'),
                 iid='warn', n_jobs=None, param_grid={'penalizer': [0, 1, 10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)

    """
