.. image:: http://i.imgur.com/EOowdSD.png

-------------------------------------

Introduction to using lifelines
=====================================

In the previous :doc:`section</Survival Analysis intro>`,
we introduced how survival analysis is used, needed, and the
mathematical objects that it relies on. In this article, we will work
with real data and the *lifelines* library to estimate these mathematical objects.

Estimating the Survival function using Kaplan-Meier
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

For this example, we will be investigating the lifetimes of political
leaders around the world. A political leader in this case is defined by a single
individual's time in office who controls the ruling regime. This could be an
elected president, unelected dictator, monarch, etc. The birth event is
the start of the individual's tenure, and the death event is the retirement of the
individual. Censorship can occur if they are a) still in offices at the
time of dataset complilation (2008), or b) die while in office (this
includes assassinations).

For example, the Bush regime began in 2000 and officially ended in 2008
upon his retirement, thus this regime's lifespan was 8 years and the
"death" event was observed. On the other hand, the JFK regime lasted 2
years, from 1961 and 1963, and the regime's official death event *was
not* observed -- JFK died before his official retirement.

(This is an example that has gladly redefined the birth and death
events, and infact completely flips the idea upside down by using deaths
as the censorship event. This is also an example where the current time
is not the only cause of censorship -- there are alternative
events (eg: death in office) that can censor.)

To estimate the survival function, we use the `Kaplan-Meier
Estimate <http://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator>`__,
defined:

.. math:: \hat{S}(t) = \prod_{t_i \lt t} \frac{n_i - d_i}{n_i}

where :math:`d_i` are the number of death events at time :math:`t` and
:math:`n_i` is the number of subjects at risk of death just prior to time
:math:`t`.

Let's bring in our dataset. 

.. code:: python

    import pandas as pd
    import lifelines 

    data = lifelines.datasets.load_dd()

.. code:: python

    data.sample(10)
    #the boolean columns `observed` refers to whether the death (leaving office)
    #was observed or not.



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
          <th>164</th>
          <td>Bolivia</td>
          <td>145</td>
          <td>145.0</td>
          <td>South America</td>
          <td>Americas</td>
          <td>Rene Barrientos Ortuno</td>
          <td>Rene Barrientos Ortuno.Bolivia.1966.1968.Milit...</td>
          <td>Non-democracy</td>
          <td>Military Dict</td>
          <td>1966</td>
          <td>3</td>
          <td>0</td>
        </tr>
        <tr>
          <th>740</th>
          <td>India</td>
          <td>750</td>
          <td>750.0</td>
          <td>Southern Asia</td>
          <td>Asia</td>
          <td>Chandra Shekhar</td>
          <td>Chandra Shekhar.India.1990.1990.Parliamentary Dem</td>
          <td>Democracy</td>
          <td>Parliamentary Dem</td>
          <td>1990</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>220</th>
          <td>Bulgaria</td>
          <td>355</td>
          <td>355.0</td>
          <td>Eastern Europe</td>
          <td>Europe</td>
          <td>Todor Zhivkov</td>
          <td>Todor Zhivkov.Bulgaria.1954.1988.Civilian Dict</td>
          <td>Non-democracy</td>
          <td>Civilian Dict</td>
          <td>1954</td>
          <td>35</td>
          <td>1</td>
        </tr>
        <tr>
          <th>772</th>
          <td>Ireland</td>
          <td>205</td>
          <td>205.0</td>
          <td>Northern Europe</td>
          <td>Europe</td>
          <td>Charles Haughey</td>
          <td>Charles Haughey.Ireland.1979.1980.Mixed Dem</td>
          <td>Democracy</td>
          <td>Mixed Dem</td>
          <td>1979</td>
          <td>2</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1718</th>
          <td>United States of America</td>
          <td>2</td>
          <td>2.0</td>
          <td>Northern America</td>
          <td>Americas</td>
          <td>Gerald Ford</td>
          <td>Gerald Ford.United States of America.1974.1976...</td>
          <td>Democracy</td>
          <td>Presidential Dem</td>
          <td>1974</td>
          <td>3</td>
          <td>1</td>
        </tr>
        <tr>
          <th>712</th>
          <td>Iceland</td>
          <td>395</td>
          <td>395.0</td>
          <td>Northern Europe</td>
          <td>Europe</td>
          <td>Stefan Stefansson</td>
          <td>Stefan Stefansson.Iceland.1947.1948.Mixed Dem</td>
          <td>Democracy</td>
          <td>Mixed Dem</td>
          <td>1947</td>
          <td>2</td>
          <td>1</td>
        </tr>
        <tr>
          <th>804</th>
          <td>Italy</td>
          <td>325</td>
          <td>325.0</td>
          <td>Southern Europe</td>
          <td>Europe</td>
          <td>Mariano Rumor</td>
          <td>Mariano Rumor.Italy.1968.1969.Parliamentary Dem</td>
          <td>Democracy</td>
          <td>Parliamentary Dem</td>
          <td>1968</td>
          <td>2</td>
          <td>1</td>
        </tr>
        <tr>
          <th>746</th>
          <td>Indonesia</td>
          <td>850</td>
          <td>850.0</td>
          <td>South-Eastern Asia</td>
          <td>Asia</td>
          <td>Sukarno</td>
          <td>Sukarno.Indonesia.1949.1965.Civilian Dict</td>
          <td>Non-democracy</td>
          <td>Civilian Dict</td>
          <td>1949</td>
          <td>17</td>
          <td>1</td>
        </tr>
        <tr>
          <th>35</th>
          <td>Argentina</td>
          <td>160</td>
          <td>160.0</td>
          <td>South America</td>
          <td>Americas</td>
          <td>Pedro Eugenio Aramburu Cilveti</td>
          <td>Pedro Eugenio Aramburu Cilveti.Argentina.1955....</td>
          <td>Non-democracy</td>
          <td>Military Dict</td>
          <td>1955</td>
          <td>3</td>
          <td>1</td>
        </tr>
        <tr>
          <th>398</th>
          <td>Democratic Republic of the Congo (Zaire, Congo...</td>
          <td>490</td>
          <td>490.0</td>
          <td>Middle Africa</td>
          <td>Africa</td>
          <td>Joseph Kasabuvu</td>
          <td>Joseph Kasabuvu.Democratic Republic of the Con...</td>
          <td>Non-democracy</td>
          <td>Civilian Dict</td>
          <td>1960</td>
          <td>5</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    <p>10 rows × 12 columns</p>
    </div>



From the ``lifelines`` library, we'll need the
``KaplanMeierFitter`` for this exercise:

.. code:: python

    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()

For this estimation, we need the duration each leader was/has been in
office, and whether or not they were observed to have left office
(leaders who died in office or were in office in 2008, the latest date
this data was record at, do not have observed death events)

We next use the ``KaplanMeierFitter`` method ``fit`` to fit the model to
the data. (This is similar to, and inspired by, another popular
Python library `scikit-learn's <http://scikit-learn.org/stable/>`__
fit/predict API)

.. code:: 

  KaplanMeierFitter.fit(event_times, event_observed=None, 
                        timeline=None, label='KM-estimate', 
                        alpha=None)
  Parameters:
    event_times: an array, or pd.Series, of length n of times that
           the death event occured at
    event_observed: an array, or pd.Series, of length n -- True if 
          the death was observed, False if the event was lost 
          (right-censored). Defaults all True if event_observed==None
    timeline: set the index of the survival curve to this postively increasing array.
    label: a string to name the column of the estimate.
    alpha: the alpha value in the confidence intervals.
           Overrides the initializing alpha for this call to fit only.

  Returns:
    self, with new properties like 'survival_function_'.


Below we fit our data to the fitter: 


.. code:: python

    T = data["duration"] 
    C = data["observed"] 

    kmf.fit(T, event_observed=C )



.. parsed-literal::

   <lifelines.KaplanMeierFitter: fitted with 1808 observations, 340 censored>


After calling the ``fit`` method, the ``KaplanMeierFitter`` has a property
called ``survival_function_``. (Again, we follow the styling of
scikit-learn, and append an underscore to all properties that were computational estimated)
The property is a Pandas DataFrame, so we can call ``plot`` on it:

.. code:: python

    kmf.survival_function_.plot()
    plt.title('Survival function of political regimes');

.. image:: images/lifelines_intro_kmf_curve.png
   

How do we interpret this? The y-axis represents the probability a leader is still
around after :math:`t` years, where :math:`t` years is on the x-axis. We
see that very few leaders make it past 20 years in office. Of course,
like all good stats, we need to report how uncertain we are about these
point estimates, i.e. we need confidence intervals. They are computed on
the call to ``fit``, and are located under the ``confidence_interval_``
property.

Alternatively, we can call ``plot`` on the ``KaplanMeierFitter`` itself
to plot both the KM estimate and its confidence intervals:

.. code:: python

    kmf.plot()

.. image:: images/lifelines_intro_kmf_fitter.png

.. note::  Don't like the shaded area for confidence intervals? See below for examples on how to change this.


The median time in office, which defines the point in time where on
average 1/2 of the population has expired, is a property:

.. code:: python

    kmf.median_

    #   4
    #



Interesting that it is only 3 years. That means, around the world, when
a leader is elected there is a 50% chance he or she will be gone in 3
years!

Let's segment on democratic regimes vs non-democratic regimes. Calling
``plot`` on either the estimate itself or the fitter object will return
an ``axis`` object, that can be used for plotting further estimates:

.. code:: python

    ax = plt.subplot(111)
    
    dem = (data["democracy"] == "Democracy")
    kmf.fit(T[dem], event_observed=C[dem], label="Democratic Regimes")
    kmf.plot(ax=ax, ci_force_lines=True)
    kmf.fit(T[~dem], event_observed=C[~dem], label="Non-democratic Regimes")
    kmf.plot(ax=ax, ci_force_lines=True)
    
    plt.ylim(0,1);
    plt.title("Lifespans of different global regimes");


.. image:: images/lifelines_intro_multi_kmf_fitter.png


We might be interested in estimating the probabilities in between some
points. We can do that with the ``timeline`` argument. We specify the
times we are interested in, and are returned a DataFrame with the
probabilties of survival at those points:

.. code:: python

    ax = subplot(111)
    
    t = np.linspace(0,50,51)
    kmf.fit(T[dem], event_observed=C[dem], timeline=t, label="Democratic Regimes")
    ax = kmf.plot(ax=ax)
    print "Median survival time of democratic:", kmf.median_
    
    kmf.fit(T[~dem], event_observed=C[~dem], timeline=t, label="Non-democratic Regimes")
    ax = kmf.plot(ax=ax)
    print "Median survival time of non-democratic:", kmf.median_

    plt.ylim(0,1)
    plt.title("Lifespans of different global regimes");

.. parsed-literal::

    Median survival time of democratic: Democratic Regimes    3
    dtype: float64
    Median survival time of non-democratic: Non-democratic Regimes    6
    dtype: float64


.. image:: images/lifelines_intro_multi_kmf_fitter_2.png


It is incredible how much longer these non-democratic regimes exist for.
A democratic regime does have a natural bias towards death though: both
via elections and natural limits (the US imposes a strict 8 year limit).
The median of a non-democractic is only about twice as large as a
democratic regime, but the difference is really apparent in the tails:
if you're a non-democratic leader, and you've made it past the 10 year
mark, you probably have a long life ahead. Meanwhile, a democratic
leader rarely makes it past 10 years, and then have a very short
lifetime past that.

Here the difference between survival functions is very obvious, and
performing a statistical test seems pendantic. If the curves are more
similar, or we possess less data, we may be interested in performing a
statistical test. In this case, *lifelines* contains routines in
``lifelines.statistics`` to compare two survival curves. Below we
demonstrate this routine. The function ``logrank_test`` is a common
statistical test in survival analysis that compares two event series'
generators. If the value returned exceeds some prespecified value, then
we rule that the series have different generators.

.. code:: python

    from lifelines.statistics import logrank_test
    
    results = logrank_test(T[dem], T[~dem], C[dem], C[~dem], alpha=.99 )

    results.print_summary()

.. parsed-literal::

    Results
       df: 1
       alpha: 0.99
       t 0: -1
       test: logrank
       null distribution: chi squared
    
       __ p-value ___|__ test statistic __|____ test results ____|__ significant __
             0.00000 |            208.306 |      Reject Null     |     True


Lets compare the different *types* of regimes present in the dataset:

.. code:: python

    regime_types = data['regime'].unique()
    
    for i,regime_type in enumerate(regime_types):
        ax = plt.subplot(2,3,i+1)
        ix = data['regime'] == regime_type
        kmf.fit( T[ix], C[ix], label=regime_type )
        kmf.plot(ax=ax, legend=False)
        plt.title(regime_type)
        plt.xlim(0,50)
        if i==0:
            plt.ylabel('Frac. in power after $n$ years')
    plt.tight_layout()


.. image:: images/lifelines_intro_all_regimes.png


--------------

Getting data into the right format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*lifelines* data format is consistent across all estimator class and
functions: an array of individual durations, and the individuals
event observation (if any). These are often denoted ``T`` and ``C``
respectively. For example:

::

    T = [0,3,3,2,1,2]
    C = [1,1,0,0,1,1]
    kmf.fit(T, event_observed=C )

The raw data is not always available in this format -- *lifelines*
includes some helper functions to transform data formats to *lifelines*
format. These are located in the ``lifelines.utils`` sublibrary. For
example, the function ``datetimes_to_durations`` accepts an array or
Pandas object of start times/dates, and an array or Pandas objects of
end times/dates (or ``None`` if not observed):

.. code:: python

    from lifelines.utils import datetimes_to_durations
    
    start_date = ['2013-10-10 0:00:00', '2013-10-09', '2013-10-10']
    end_date = ['2013-10-13', '2013-10-10', None]
    T,C = datetimes_to_durations(start_date, end_date, fill_date='2013-10-15')
    print 'T (durations): ', T
    print 'C (event_observed): ',C

.. parsed-literal::

    T (durations):  [ 3.  1.  5.]
    C (event_observed):  [ True  True False]


The function ``datetimes_to_durations`` is very flexible, and has many
keywords to tinker with.

Estimating hazard rates using Nelson-Aalen
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The survival curve is a great way to summarize and visualize the
lifetime data, however it is not the only way. If we are curious about the hazard function :math:`\lambda(t)` of a
population, we unfortunately cannot transform the Kaplan Meier estimate
-- statistics doesn't work quite that well. Fortunately, there is a
proper estimator of the *cumulative* hazard function:

.. math::  \Lambda(t) =  \int_0^t \lambda(z) \;dz



The estimator for this quantity is called the Nelson Aalen estimator:



.. math:: \hat{\Lambda}(t) = \sum_{t_i \le t} \frac{d_i}{n_i} 

where :math:`d_i` is the number of deaths at time :math:`t_i` and
:math:`n_i` is the number of susceptible individuals.

In *lifelines*, this estimator is available as the ``NelsonAalenFitter``. Let's use the regime dataset from above:

.. code:: python

    T = data["duration"]
    C = data["observed"]

    from lifelines import NelsonAalenFitter
    naf = NelsonAalenFitter()

    naf.fit(T,event_observed=C)


After fitting, the class exposes the property ``cumulative_hazard_`` as
a DataFrame:

.. code:: python

    print naf.cumulative_hazard_.head()
    naf.plot()

.. parsed-literal::

       NA-estimate
    0     0.000000
    1     0.325912
    2     0.507356
    3     0.671251
    4     0.869867
    
    [5 rows x 1 columns]



.. image:: images/lifelines_intro_naf_fitter.png


The cumulative hazard has less immediate understanding than the survival
curve, but the hazard curve is the basis of more advanced techniques in
survival analysis. Recall that we are estimating *cumulative hazard
curve*, :math:`\Lambda(t)`. (Why? The sum of estimates is much more
stable than the point-wise estimates.) Thus we know the *rate of change*
of this curve is an estimate of the hazard function.

Looking at figure above, it looks like the hazard starts off high and
gets smaller (as seen by the decreasing rate of change). Let's break the
regimes down between democratic and non-democratic, during the first 20
years:

.. note::  We are using the ``ix`` argument in the call to ``plot`` here: it accepts a ``slice`` and plots only points within that slice.

.. code:: python

    naf.fit(T[dem], event_observed=C[dem], label="Democratic Regimes")
    ax = naf.plot(ix=slice(0,20))
    naf.fit(T[~dem], event_observed=C[~dem], label="Non-democratic Regimes")
    naf.plot(ax=ax, ix=slice(0,20))
    plt.title("Cumulative hazard function of different global regimes");


.. image:: images/lifelines_intro_naf_fitter_multi.png


Looking at the rates of change, I would say that both political
philosophies have a constant hazard, albeit democratic regimes have a
much *higher* constant hazard. So why did the combination of both
regimes have a *decreasing* hazard? This is the effect of *frailty*, a
topic we will discuss later.

Smoothing the hazard curve
~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpretation of the cumulative hazard function can be difficult -- it
is not how we usually interpret functions. (On the other hand, most
survival analysis is done using the cumulative hazard function, so understanding
it is recommended).

Alternatively, we can derive the more-interpretable hazard curve, but
there is a catch. The derivation involves a kernel smoother (to smooth
out the differences of the cumulative hazard curve) , and this requires
us to specify a bandwidth parameter that controls the amount of
smoothing. This functionality is provided in the ``smoothed_hazard_``
and ``hazard_confidence_intervals_`` methods. (Why methods? They require
an argument representing the bandwidth).

There is also a ``plot_hazard`` function (that also requires a
``bandwidth`` keyword) that will plot the estimate plus the confidence
intervals, similar to the traditional ``plot`` functionality.

.. code:: python

    b = 3.
    naf.fit(T[dem], event_observed=C[dem], label="Democratic Regimes")
    ax = naf.plot_hazard(bandwidth=b)
    naf.fit(T[~dem], event_observed=C[~dem], label="Non-democratic Regimes")
    naf.plot_hazard(ax=ax, bandwidth=b)
    plt.title("Hazard function of different global regimes | bandwidth=%.1f"%b);
    plt.ylim(0,0.4)
    plt.xlim(0,25);


.. image:: images/lifelines_intro_naf_smooth_multi.png


It is more clear here which group has the higher hazard, and like
hypothesized above, both hazard rates are close to being constant.

There is no obvious way to choose a bandwidth, and different
bandwidths can produce different inferences, so best to be very careful
here. (My advice: stick with the cumulative hazard function.)

.. code:: python

    b = 8.
    naf.fit(T[dem], event_observed=C[dem], label="Democratic Regimes")
    ax = naf.plot_hazard(bandwidth=b)
    naf.fit(T[~dem], event_observed=C[~dem], label="Non-democratic Regimes")
    naf.plot_hazard(ax=ax, bandwidth=b)
    plt.title("Hazard function of different global regimes | bandwidth=%.1f"%b);



.. image:: images/lifelines_intro_naf_smooth_multi_2.png



Other types of censorship
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Left Censored Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

We've mainly been focusing on *right-censorship*, which describes cases where we do not observe the death event.
This situation is the most common one. Alternatively, there are situations where we do not observe the *birth* event
occurring. Consider the case where a doctor sees a delayed onset of symptoms of an underlying disease. The doctor
is unsure *when* the disease was contracted (birth), but knows it was before the discovery. 

Another situation where we have left censored data is when measurements have only an upperbound, that is, the measurements
instruments could only detect the measurement was *less* than some upperbound.

*lifelines* has support for left-censored datasets in the ``KaplanMeierFitter`` class, by adding the keyword ``left_censorship=True`` (default ``False``) to the call to ``fit``. 

.. code:: python

    from lifelines.datasets import load_lcd
    lcd_dataset = load_lcd()

    ix = lcd_dataset['group'] == 'alluvial_fan'
    T = lcd_dataset[ix]['T']
    C = lcd_dataset[ix]['C'] #boolean array, True if observed.

    kmf = KaplanMeierFitter()
    kmf.fit(T,C, left_censorship=True)  

Instead of producing a survival function, left-censored data is more interested in the cumulative density function
of time to birth. This is available as the ``cumulative_density_`` property after fitting the data.

.. code:: python
    
    print kmf.cumulative_density_
    kmf.plot() #will plot the CDF


.. image:: images/lifelines_intro_lcd.png

Left Truncated Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Another form of bias that can be introduced into a dataset is called left-truncation. (Also a form of censorship). 
This occurs when individuals may die even before ever entering into the study. Both  ``KaplanMeierFitter`` and ``NelsonAalenFitter`` have an optional arugment for ``entry``, which is an array of equal size to the duration array.
It describes the offset from birth to entering the study. This is also useful when subjects enter the study at different
points in their lifetime. For example, if you are measuring time to death of prisoners in 
prison, the prisoners will enter the study at different ages. 

 .. note:: Nothing changes in the duration array: it still measures time from entry of study to time left study (either by death or censorship)

 .. note:: Other types of censorship, like interval-censorship, are not implemented in *lifelines* yet.
