.. _code_directive:

Quickstart
'''''''''''''''''''''''''''''''''''''''

Kaplan-Meier and Nelson-Aalen
---------------------------------------


Let's start by importing some data. We need the durations that individuals are observed for, and whether they "died" or not. 

.. code:: python

    from lifelines.datasets import waltons_data    

    T = waltons_data['T']
    E = waltons_data['E']

``T`` is an array of durations, ``E`` is a either boolean or binary array representing whether the "death" was observed (alternatively an individual can be censored). 

.. note:: By default, *lifelines* assumes all "deaths" are observed. 


.. code:: python

    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)

After calling the ``fit`` method, we have access to new properties like ``survival_function_`` and methods like ``plot()``. The latter is a wrapper around Pandas internal plotting library (see `here <http://lifelines.readthedocs.org/en/latest/examples.html#plotting-options-and-styles>`__ for examples). 

.. code:: python
    
    kmf.plot()


.. image:: images/quickstart_kmf.png


Multiple groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python
    
    groups = waltons_data['group']
    ix = groups == 'control'

    kmf.fit(T[ix], E[ix], label='control')
    ax = kmf.plot()

    kmf.fit(T[~ix], E[~ix], label='miR-137')
    kmf.plot(ax=ax)

.. image:: images/quickstart_multi.png   

Similar functionality exists for the ``NelsonAalenFitter``:


.. code:: python

    from lifelines import NelsonAalenFitter
    naf = NelsonAalenFitter()
    naf.fit(T, event_observed=E)

but instead of a ``survival_function_`` being exposed, a ``cumulative_hazard_`` is. 

.. note:: Similar to Scikit-Learn, all estimated quanities in lifelines appends an underscore to the name. 

Survival Regression
---------------------------------

While the above ``KaplanMeierFitter`` and ``NelsonAalenFitter`` are useful, they only give us an "average" view of the population. Often we have specific data at the individual level, either continuous or categorical, that we would like to use. For this, we turn to **survival regression**, specifically ``AalenAdditiveFitter``.

.. code:: python
    
    from lifelines.datasets import regression_data

    regression_data.head()



The input of the ``fit`` method's API on ``AalenAdditiveFitter`` is different than above. All the data, including durations, censorships and covariates must be contained in **a Pandas DataFrame** (yes, it must be a DataFrame). The duration column and event occured column must be specified in the call to ``fit``. 

.. code:: python
    
    from lifelines import AalenAdditiveFitter

    aaf = AalenAdditiveFitter(fit_intercept=False)
    aaf.fit(regression_data, duration_col='T', event_col='E')


After fitting, you'll have access to properties like ``cumulative_hazards_`` and methods like ``plot``, ``predict_cumulative_hazards``, and ``predict_survival_function``. The latter two methods require an additional argument of individual covariates:

.. code:: python
    
    x = regression_data[regression_data.columns - ['E','T']]
    aaf.predict_survival_function(x.ix[10:12]).plot() #get the unique survival functions of the first two subjects 


Like the above estimators, there is also a built-in plotting method:

.. code:: python

    aaf.plot()

.. image:: images/quickstart_aaf.png  
