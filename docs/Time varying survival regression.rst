
Time varying survival regression
=====================================

Cox's time varying proportional hazard model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often an individual will have a covariate change over time. An example of this is hospital patients who enter the study and, at some future time, may receive a heart transplant. We would like to know the effect of the transplant, but we must be careful if we condition on whether they received the transplant. Consider that if patients needed to wait at least 1 year before getting a transplant, then everyone who dies before that year is considered as a non-transplant patient, and hence this would overestimate the hazard of not receiving a transplant.

We can incorporate changes over time into our survival analysis by using a modification of the Cox model. The general mathematical description is:

.. math::  h(t | x) = \overbrace{b_0(t)}^{\text{baseline}}\underbrace{\exp \overbrace{\left(\sum_{i=1}^n \beta_i (x_i(t) - \overline{x_i}) \right)}^{\text{log-partial hazard}}}_ {\text{partial hazard}}

Note the time-varying :math:`x_i(t)` to denote that covariates can change over time. This model is implemented in *lifelines* as :class:`~lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter`. The dataset schema required is different than previous models, so we will spend some time describing it.

Dataset creation for time-varying regression
#############################################

*lifelines* requires that the dataset be in what is called the *long* format. This looks like one row per state change, including an ID, the left (exclusive) time point, and right (inclusive) time point. For example, the following dataset tracks three unique subjects.

.. table::

    +--+-----+----+-----+-+-----+
    |id|start|stop|group|z|event|
    +==+=====+====+=====+=+=====+
    | 1|    0|   8|    1|0|False|
    +--+-----+----+-----+-+-----+
    | 2|    0|   5|    0|0|False|
    +--+-----+----+-----+-+-----+
    | 2|    5|   8|    0|1|True |
    +--+-----+----+-----+-+-----+
    | 3|    0|   3|    1|0|False|
    +--+-----+----+-----+-+-----+
    | 3|    3|  12|    1|1|True |
    +--+-----+----+-----+-+-----+


In the above dataset, ``start`` and ``stop`` denote the boundaries, ``id`` is the unique identifier per subject, and ``event`` denotes if the subject died at the end of that period. For example, subject ID 2 had variable ``z=0`` up to and including the end of time period 5 (we can think that measurements happen at end of the time period), after which it was set to 1. Since ``event`` is 1 in that row, we conclude that the subject died at time 8,

This desired dataset can be built up from smaller datasets. To do this we can use some helper functions provided in *lifelines*. Typically, data will be in a format that looks like it comes out of a relational database. You may have a "base" table with ids, durations alive, and a censored flag, and possibly static covariates. Ex:

.. table::

    +--+--------+-----+----+
    |id|duration|event|var1|
    +==+========+=====+====+
    | 1|      10|True | 0.1|
    +--+--------+-----+----+
    | 2|      12|False| 0.5|
    +--+--------+-----+----+


We will perform a light transform to this dataset to modify it into the "long" format.

.. code:: python

      from lifelines.utils import to_long_format

      base_df = to_long_format(base_df, duration_col="duration")

The new dataset looks like:


.. table::

    +--+-----+----+----+-----+
    |id|start|stop|var1|event|
    +==+=====+====+====+=====+
    | 1|    0|  10| 0.1|True |
    +--+-----+----+----+-----+
    | 2|    0|  12| 0.5|False|
    +--+-----+----+----+-----+


You'll also have secondary dataset that references future measurements. This could come in two "types". The first is when you have a variable that changes over time (ex: administering varying medication over time, or taking a tempature over time). The second types is an event-based dataset: an event happens at some time in the future (ex: an organ transplant occurs, or an intervention). We will address this second type later. The first type of dataset may look something like:

Example:

.. table::

    +--+----+----+
    |id|time|var2|
    +==+====+====+
    | 1|   0| 1.4|
    +--+----+----+
    | 1|   4| 1.2|
    +--+----+----+
    | 1|   8| 1.5|
    +--+----+----+
    | 2|   0| 1.6|
    +--+----+----+

where ``time`` is the duration from the entry event. Here we see subject 1 had a change in their ``var2`` covariate at the end of time 4 and at the end of time 8. We can use :func:`lifelines.utils.add_covariate_to_timeline` to fold the covariate dataset into the original dataset.


.. code:: python

      from lifelines.utils import add_covariate_to_timeline

      df = add_covariate_to_timeline(base_df, cv, duration_col="time", id_col="id", event_col="event")


.. table::

    +--+-----+----+----+----+-----+
    |id|start|stop|var1|var2|event|
    +==+=====+====+====+====+=====+
    | 1|    0|   4| 0.1| 1.4|False|
    +--+-----+----+----+----+-----+
    | 1|    4|   8| 0.1| 1.2|False|
    +--+-----+----+----+----+-----+
    | 1|    8|  10| 0.1| 1.5|True |
    +--+-----+----+----+----+-----+
    | 2|    0|  12| 0.5| 1.6|False|
    +--+-----+----+----+----+-----+

From the above output, we can see that subject 1 changed state twice over the observation period, finally expiring at the end of time 10. Subject 2 was a censored case, and we lost track of them after time 12.

You may have multiple covariates you wish to add, so the above could be streamlined like so:

.. code:: python

      from lifelines.utils import add_covariate_to_timeline

      df = base_df.pipe(add_covariate_to_timeline, cv1, duration_col="time", id_col="id", event_col="event")\
                  .pipe(add_covariate_to_timeline, cv2, duration_col="time", id_col="id", event_col="event")\
                  .pipe(add_covariate_to_timeline, cv3, duration_col="time", id_col="id", event_col="event")


If your dataset is of the second type, that is, event-based, your dataset may look something like the following, where values in the matrix denote times since the subject's birth, and ``None`` or  ``NaN`` represent the event not happening (subjects can be excluded if the event never occurred as well) :

.. code-block:: python

    print(event_df)


        id    E1
    0   1     1.0
    1   2     NaN
    2   3     3.0
    ...

Initially, this can't be added to our baseline DataFrame. However, using :func:`lifelines.utils.covariates_from_event_matrix` we can convert a DataFrame like this into one that can be easily added.


.. code-block:: python

    from lifelines.utils import covariates_from_event_matrix

    cv = covariates_from_event_matrix(event_df, id_col="id")
    print(cv)


    event  id  duration  E1
    0       1       1.0   1
    1       3       3.0   1
    ...


    base_df = add_covariate_to_timeline(base_df, cv, duration_col="time", id_col="id", event_col="E")

For an example of pulling datasets like this from a SQL-store, and other helper functions, see :ref:`Example SQL queries and transformations to get time varying data`.

Cumulative sums
#############################################

One additional flag on :func:`~lifelines.utils.add_covariate_to_timeline` that is of interest is the ``cumulative_sum`` flag. By default it is False, but turning it to True will perform a cumulative sum on the covariate before joining. This is useful if the covariates describe an incremental change, instead of a state update. For example, we may have measurements of drugs administered to a patient, and we want the covariate to reflect how much we have administered since the start. Event columns do make sense to cumulative sum as well. In contrast, a covariate to measure the temperature of the patient is a state update, and should not be summed.  See :ref:`Example cumulative sums over time-varying covariates` to see an example of this.

Delaying time-varying covariates
#############################################

:func:`~lifelines.utils.add_covariate_to_timeline` also has an option for delaying, or shifting, a covariate so it changes later than originally observed. One may ask, why should one delay a time-varying covariate? Here's an example. Consider investigating the impact of smoking on mortality and available to us are time-varying observations of how many cigarettes are consumed each month. Unbeknown-st to us, when a subject reaches critical illness levels, they are admitted to the hospital and their cigarette consumption drops to zero. Some expire while in hospital. If we used this dataset naively, we would see that not smoking leads to sudden death, and conversely, smoking helps your health! This is a case of reverse causation: the upcoming death event actually influences the covariates.

To handle this, you can delay the observations by time periods:

.. code-block:: python

    from lifelines.utils import covariates_from_event_matrix


    base_df = add_covariate_to_timeline(base_df, cv, duration_col="time", id_col="id", event_col="E", delay=14)



Fitting the model
################################################

Once your dataset is in the correct orientation, we can use :class:`~lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter` to fit the model to your data. The method is similar to :class:`~lifelines.fitters.coxph_fitter.CoxPHFitter`, except we need to tell the :meth:`~lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter.fit` about the additional time columns.

Fitting the Cox model to the data involves an iterative gradient descent. *lifelines* takes extra effort to help with convergence, so please be attentive to any warnings that appear. Fixing any warnings will generally help convergence. For further help, see :ref:`Problems with convergence in the Cox Proportional Hazard Model`.


.. code:: python

    from lifelines import CoxTimeVaryingFitter

    ctv = CoxTimeVaryingFitter()
    ctv.fit(df, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=True)
    ctv.print_summary()
    ctv.plot()


Short note on prediction
################################################

Unlike the other regression models, prediction in a time-varying setting is not trivial. To predict, we would need to know the covariates values beyond the observed times, but if we knew that, we would also know if the subject was still alive or not! However, it is still possible to compute the hazard values of subjects at known observations, the baseline cumulative hazard rate, and baseline survival function. So while :class:`~lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter` exposes prediction methods, there are logical limitations to what these predictions mean.
