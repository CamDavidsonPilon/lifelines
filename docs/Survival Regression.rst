.. image:: http://i.imgur.com/EOowdSD.png

-------------------------------------

Survival Regression
=====================================

Often we have additional data aside from the durations, and if
applicable any censorships that occurred. In the regime dataset, we have
the type of government the political leader was part of, the country
they were head of, and the year they were elected. Can we use this data
in survival analysis?

Yes, the technique is called *survival regression* -- the name implies
we regress covariates (eg: year elected, country, etc.) against a
another variable -- in this case durations and lifetimes. Similar to the
logic in the first part of this tutorial, we cannot use traditional
methods like linear regression.

There are two popular competing techniques in survival regression: Cox's
model and Aalen's additive model. Both models attempt to represent the
hazard rate :math:`\lambda(t)`. In Cox's model, the relationship is
defined:

.. math:: \lambda(t) = b_0(t)\exp\left( b_1x_1 + ... + b_Nx_n\right)

On the other hand, Aalen's additive model assumes the following form:

.. math:: \lambda(t) = b_0(t) + b_1(t)x_1 + ... + b_N(t)x_T



Aalen's Additive model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: This model is still experimental.

The estimator to fit unknown coefficients in Aalen's additive model is
located in ``estimators`` under ``AalenAdditiveFitter``. For this
exercise, we will use the regime dataset and include the categorical
variables ``un_continent_name`` (eg: Asia, North America,...), the
``regime`` type (eg: monarchy, civilan,...) and the year the regime
started in, ``start_year``.

Aalen's additive model typically does not estimate the individual
:math:`b_i(t)` but instead estimates :math:`\int_0^t b_i(s) \; ds`
(similar to the estimate of the hazard rate using ``NelsonAalenFitter``
above). This is important to keep in mind when analzying the output.

.. code:: python

    from lifelines import AalenAdditiveFitter
    data.head()


.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ctryname</th>
          <th>cowcode2</th>
          <th>politycode</th>
          <th>un_region_name</th>
          <th>un_continent_name</th>
          <th>ehead</th>
          <th>leaderspellreg</th>
          <th>democracy</th>
          <th>regime</th>
          <th>start_year</th>
          <th>duration</th>
          <th>observed</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td> Afghanistan</td>
          <td> 700</td>
          <td> 700</td>
          <td> Southern Asia</td>
          <td> Asia</td>
          <td>   Mohammad Zahir Shah</td>
          <td> Mohammad Zahir Shah.Afghanistan.1946.1952.Mona...</td>
          <td> Non-democracy</td>
          <td>      Monarchy</td>
          <td> 1946</td>
          <td>  7</td>
          <td> 1</td>
        </tr>
        <tr>
          <th>2</th>
          <td> Afghanistan</td>
          <td> 700</td>
          <td> 700</td>
          <td> Southern Asia</td>
          <td> Asia</td>
          <td> Sardar Mohammad Daoud</td>
          <td> Sardar Mohammad Daoud.Afghanistan.1953.1962.Ci...</td>
          <td> Non-democracy</td>
          <td> Civilian Dict</td>
          <td> 1953</td>
          <td> 10</td>
          <td> 1</td>
        </tr>
        <tr>
          <th>3</th>
          <td> Afghanistan</td>
          <td> 700</td>
          <td> 700</td>
          <td> Southern Asia</td>
          <td> Asia</td>
          <td>   Mohammad Zahir Shah</td>
          <td> Mohammad Zahir Shah.Afghanistan.1963.1972.Mona...</td>
          <td> Non-democracy</td>
          <td>      Monarchy</td>
          <td> 1963</td>
          <td> 10</td>
          <td> 1</td>
        </tr>
        <tr>
          <th>4</th>
          <td> Afghanistan</td>
          <td> 700</td>
          <td> 700</td>
          <td> Southern Asia</td>
          <td> Asia</td>
          <td> Sardar Mohammad Daoud</td>
          <td> Sardar Mohammad Daoud.Afghanistan.1973.1977.Ci...</td>
          <td> Non-democracy</td>
          <td> Civilian Dict</td>
          <td> 1973</td>
          <td>  5</td>
          <td> 0</td>
        </tr>
        <tr>
          <th>5</th>
          <td> Afghanistan</td>
          <td> 700</td>
          <td> 700</td>
          <td> Southern Asia</td>
          <td> Asia</td>
          <td>   Nur Mohammad Taraki</td>
          <td> Nur Mohammad Taraki.Afghanistan.1978.1978.Civi...</td>
          <td> Non-democracy</td>
          <td> Civilian Dict</td>
          <td> 1978</td>
          <td>  1</td>
          <td> 0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 12 columns</p>
    </div>


I'm using the lovely library `patsy <https://github.com/pydata/patsy>`__ here to create a
covariance matrix from my original dataframe.

.. code:: python

    import patsy
    # the '-1' term refers to not adding an intercept column (a column of all 1s).
    X = patsy.dmatrix('un_continent_name + regime + start_year -1', data, return_type='dataframe') 

.. code:: python

    X.columns



.. parsed-literal::

    ['un_continent_name[Africa]',
     'un_continent_name[Americas]',
     'un_continent_name[Asia]',
     'un_continent_name[Europe]',
     'un_continent_name[Oceania]',
     'regime[T.Military Dict]',
     'regime[T.Mixed Dem]',
     'regime[T.Monarchy]',
     'regime[T.Parliamentary Dem]',
     'regime[T.Presidential Dem]',
     'start_year']


Below we create our fitter class. Since we did not supply an intercept
column in our matrix we have included the keyword ``fit_intercept=True``
(``True`` by default) which will append the column of ones to our
matrix. (Sidenote: the intercept term, :math:`b_0(t)` in survival
regression is often referred to as the *baseline* hazard.)

We have also included the ``coef_penalizer`` option. During the estimation, a
linear regression is computed at each step. Often the regression can be
unstable (due to high
`co-linearity <http://camdp.com/blogs/machine-learning-counter-examples-pt1>`__
or small sample sizes) -- adding a penalizer term controls the stability. I recommend always starting with a small penalizer term -- if the estimates still appear to be too unstable, try increasing it.

.. code:: python

    aaf = AalenAdditiveFitter(coef_penalizer=1.0, fit_intercept=True)

An instance of ``AalenAdditiveFitter``
includes a ``fit`` method that performs the inference on the coefficients. This method accepts a pandas DataFrame: each row is an individual and columns are the covariates and 
two special columns: a *duration* column and a boolean *event occured* column (where event occured refers to the event of interest - expulsion from government in this case)


.. code:: python
    
    data = lifelines.datasets.load_dd()

    X['T'] = data['duration']
    X['E'] = data['observed'] 


.. code:: python

    aaf.fit(X, 'T', event_col='E')



After fitting, the instance exposes a ``cumulative_hazards_`` DataFrame
containing the estimates of :math:`\int_0^t b_i(s) \; ds`:

.. code:: python

    figsize(12.5,8)
    aaf.cumulative_hazards_.head()


.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>un_continent_name[Africa]</th>
          <th>un_continent_name[Americas]</th>
          <th>un_continent_name[Asia]</th>
          <th>un_continent_name[Europe]</th>
          <th>un_continent_name[Oceania]</th>
          <th>regime[T.Military Dict]</th>
          <th>regime[T.Mixed Dem]</th>
          <th>regime[T.Monarchy]</th>
          <th>regime[T.Parliamentary Dem]</th>
          <th>regime[T.Presidential Dem]</th>
          <th>start_year</th>
          <th>baseline</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>-0.051595</td>
          <td>-0.082406</td>
          <td> 0.010666</td>
          <td> 0.154493</td>
          <td>-0.060438</td>
          <td> 0.075333</td>
          <td> 0.086274</td>
          <td>-0.133938</td>
          <td> 0.048077</td>
          <td> 0.127171</td>
          <td> 0.000116</td>
          <td>-0.029280</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.014713</td>
          <td>-0.039471</td>
          <td> 0.095668</td>
          <td> 0.194251</td>
          <td>-0.092696</td>
          <td> 0.115033</td>
          <td> 0.358702</td>
          <td>-0.226233</td>
          <td> 0.168783</td>
          <td> 0.121862</td>
          <td> 0.000053</td>
          <td> 0.143039</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 0.007389</td>
          <td>-0.064758</td>
          <td> 0.115121</td>
          <td> 0.170549</td>
          <td> 0.069371</td>
          <td> 0.161490</td>
          <td> 0.677347</td>
          <td>-0.271183</td>
          <td> 0.328483</td>
          <td> 0.146234</td>
          <td> 0.000004</td>
          <td> 0.297672</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.058418</td>
          <td> 0.011399</td>
          <td> 0.091784</td>
          <td> 0.205824</td>
          <td> 0.125722</td>
          <td> 0.220028</td>
          <td> 0.932674</td>
          <td>-0.294900</td>
          <td> 0.365604</td>
          <td> 0.422617</td>
          <td> 0.000002</td>
          <td> 0.376311</td>
        </tr>
        <tr>
          <th>5</th>
          <td>-0.099282</td>
          <td> 0.106641</td>
          <td> 0.112083</td>
          <td> 0.150708</td>
          <td> 0.091900</td>
          <td> 0.241575</td>
          <td> 1.123860</td>
          <td>-0.391103</td>
          <td> 0.536185</td>
          <td> 0.743913</td>
          <td> 0.000057</td>
          <td> 0.362049</td>
        </tr>
      </tbody>
    </table>
    </div>



``AalenAdditiveFitter`` also has built in plotting:

.. code:: python

  aaf.plot(columns=['regime[T.Presidential Dem]', 'baseline', 'un_continent_name[Europe]'], iloc=slice(1,15))


.. image:: images/survival_regression_aaf.png


Regression is most interesting if we use it on data we have not yet
seen, i.e. prediction! We can use what we have learned to predict
individual hazard rates, survival functions, and median survival time.
The dataset we are using is aviable up until 2008, so let's use this data to
predict the (already partly seen) possible duration of Canadian
Prime Minister Stephen Harper.

.. code:: python

    ix = (data['ctryname'] == 'Canada') * (data['start_year'] == 2006)
    harper = X.loc[ix]
    print "Harper's unique data point", harper

.. parsed-literal::

    Harper's unique data point



.. parsed-literal::

    array([[    0.,     0.,     1.,     0.,     0.,     0.,     0.,     1.,
                0.,     0.,  2003.]])



.. code:: python

    ax = plt.subplot(2,1,1)

    aaf.predict_cumulative_hazard(harper).plot(ax=ax)
    ax = plt.subplot(2,1,2)

    aaf.predict_survival_function(harper).plot(ax=ax);


.. image:: images/survival_regression_harper.png

.. warning:: Because of the nature of the model, estimated survival functions of individuals can increase. This is an expected artifact of Aalen's additive model.


Cox's Proportional Hazard model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lifelines has an implementation of the Cox propotional hazards regression model (implemented in 
R under ``coxph``). It has a similar API to Aalen's additive model. Like R, it has a ``print_summary``
function that prints a tabular view of coefficients and related stats. 

This example data is from the paper `here <http://socserv.socsci.mcmaster.ca/jfox/Books/Companion/appendix/Appendix-Cox-Regression.pdf>`_.

.. code:: python

    from lifelines.datasets import load_rossi
    from lifelines import CoxPHFitter

    rossi_dataset = load_rossi()
    cph = CoxPHFitter()
    cph.fit(rossi_dataset, duration_col='week', event_col='arrest')

    cph.print_summary()  # access the results using cph.summary

    """
    n=432, number of events=114

            coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
    fin  -0.3790     0.6845    0.1914 -1.9806 0.0476     -0.7542     -0.0039   *
    age  -0.0572     0.9444    0.0220 -2.6042 0.0092     -0.1003     -0.0142  **
    race  0.3141     1.3691    0.3080  1.0198 0.3078     -0.2897      0.9180
    wexp -0.1511     0.8597    0.2121 -0.7124 0.4762     -0.5670      0.2647
    mar  -0.4328     0.6487    0.3818 -1.1335 0.2570     -1.1813      0.3157
    paro -0.0850     0.9185    0.1957 -0.4341 0.6642     -0.4687      0.2988
    prio  0.0911     1.0954    0.0286  3.1824 0.0015      0.0350      0.1472  **
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Concordance = 0.640
    """

To access the coefficients and the baseline hazard, you can use ``cph.hazards_`` and ``cph.baseline_hazard_`` respectively. After fitting, you can use use the suite of prediction methods (similar to Aalen's additve model above): ``.predict_partial_hazard``, ``.predict_survival_function``, etc.

.. code:: python
    
    cph.predict_partial_hazard(rossi_dataset.drop(["week", "arrest"], axis=1))


Plotting the coefficients
###########################################

With a fitted model, an altervative way to view the coefficients and their ranges is to use the ``plot`` method.

.. code:: python

    from lifelines.datasets import load_rossi
    from lifelines import CoxPHFitter

    rossi_dataset = load_rossi()
    cph = CoxPHFitter()
    cph.fit(rossi_dataset, duration_col='week', event_col='arrest')

    cph.plot()

.. image:: images/coxph_plot.png



Checking the proportional hazards assumption
#############################################

A quick and visual way to check the proportional hazards assumption of a variable is to plot the survival curves segmented by the values of the variable. If the survival curves are the same "shape", and differ only by constant factor, then the assumption holds. A more clear way to see this is to plot what's called the loglogs curve: the log(-log(survival curve)) vs log(time). If the curves are parallel (and hence do not cross each other), then it's likely the variable satisfies the assumption. If the curves do cross, likely you'll have to "stratify" the variable (see next section). In lifelines, the ``KaplanMeierFitter`` object has a ``.plot_loglogs`` function for this purpose. 

The following is the loglogs curves of two variables in our regime dataset. The first is the democracy type, which does have (close to) parallel lines, hence satisfies our assumption:

.. image:: images/lls_democracy.png


The second variable is the regime type, and this variable does not follow the proportional hazards assumption.

.. image:: images/lls_regime_type.png


Stratification
################

Sometimes a covariate may not obey the proportional hazard assumption. In this case, we can allow a factor to be adjusted for without estimating its effect. To specify categorical variables to be used in stratification, we specify them in the call to ``fit``:

.. code:: python

    cph.fit(rossi_dataset, 'week', event_col='arrest', strata=['race'])

    cph.print_summary()  # access the results using cph.summary

    """
    n=432, number of events=114

            coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
    fin  -0.3775     0.6856    0.1913 -1.9731 0.0485     -0.7525     -0.0024   *
    age  -0.0573     0.9443    0.0220 -2.6081 0.0091     -0.1004     -0.0142  **
    wexp -0.1435     0.8664    0.2127 -0.6746 0.4999     -0.5603      0.2734
    mar  -0.4419     0.6428    0.3820 -1.1570 0.2473     -1.1907      0.3068
    paro -0.0839     0.9196    0.1958 -0.4283 0.6684     -0.4677      0.3000
    prio  0.0919     1.0962    0.0287  3.1985 0.0014      0.0356      0.1482  **
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Concordance = 0.638
    """


Model Selection in Survival Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If censorship is present, it's not appropriate to use a loss function like mean-squared-error or 
mean-absolute-loss. Instead, one measure is the concordance-index, also known as the c-index. This measure
evaluates the accuracy of the ordering of predicted time. It is infact a generalization
of AUC, another common loss function, and is interpreted similarly: 

* 0.5 is the expected result from random predictions,
* 1.0 is perfect concordance and,
* 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

The measure is implemented in lifelines under `lifelines.utils.concordance_index` and accepts the actual times (along with any censorships) and the predicted times.

Cross Validation
######################################

Lifelines has an implementation of k-fold cross validation under `lifelines.utils.k_fold_cross_validation`. This function accepts an instance of a regression fitter (either ``CoxPHFitter`` of ``AalenAdditiveFitter``), a dataset, plus `k` (the number of folds to perform, default 5). On each fold, it splits the data 
into a training set and a testing set, fits itself on the training set, and evaluates itself on the testing set (using the concordance measure). 

.. code:: python
      
        from lifelines import CoxPHFitter
        from lifelines.datasets import load_regression_dataset
        from lifelines.utils import k_fold_cross_validation

        regression_dataset = load_regression_dataset()
        cph = CoxPHFitter()
        scores = k_fold_cross_validation(cph, regression_dataset, 'T', event_col='E', k=3)
        print scores
        print np.mean(scores)
        print np.std(scores)
        
        #[ 0.5896  0.5358  0.5028]
        # 0.542
        # 0.035
