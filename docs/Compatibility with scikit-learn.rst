.. image:: http://i.imgur.com/EOowdSD.png

-------------------------------------


Compatibility with scikit-learn
###################################

New to lifelines in version 0.21.3 is a wrapper that allows you to use lifeline's regression models with scikit-learn's APIs.

.. note:: The X variable still needs to be a DataFrame, and should contains the event-occurred column (``event_col``) if it exists.



.. code:: python

    from lifelines.utils.sklearn_adapter import sklearn_adapter

    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    X = load_rossi().drop('week', axis=1)
    Y = load_rossi().pop('week')

    CoxRegression = sklearn_adapter(CoxPHFitter, event_col='arrest')
    # CoxRegression is a class like the `LinearRegression` class or `SVC` class in scikit-learn

    sk_cph = CoxRegression(penalizer=1.0)
    sk_cph.fit(X, Y)
    print(sk_cph)

    """
    CoxPHFitter(alpha=0.05, penalizer=1.0, strata=None, tie_method='Efron')
    """

    sk_cph.predict(X)
    sk_cph.score(X, Y)


If needed, the original lifeline's instance is available as the ``lifelines_model`` attribute.

.. code:: python

    sk_cph.lifelines_model.print_summary()



The wrapped classes can even be used in more complex scikit-learn functions (ex: ``cross_val_score``) and classes (ex: ``GridSearchCV``):


.. code:: python

    from lifelines import WeibullAFTFitter
    from sklearn.model_selection import cross_val_score


    base_class = sklearn_adapter(WeibullAFTFitter, event_col='arrest')
    wf = base_class()

    scores = cross_val_score(wf, X, Y, cv=5)
    print(scores)

    """
    [0.59037328 0.503427   0.55454545 0.59689534 0.62311068]
    """



    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(wf, {
       "penalizer": 10.0 ** np.arange(-2, 3),
       "l1_ratio": [0, 1/3, 2/3],
       "model_ancillary": [True, False],
    }, cv=4)
    clf.fit(X, Y)

    print(clf.best_estimator_)

    """
    SkLearnWeibullAFTFitter(alpha=0.05, fit_intercept=True,
                            l1_ratio=0.66666, model_ancillary=True,
                            penalizer=0.01)

    """


.. note:: The ``sklearn_adapter`` is currently only designed to work with right-censored data.
