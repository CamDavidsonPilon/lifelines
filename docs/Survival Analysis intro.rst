Introduction to Survival Analysis
'''''''''''''''''''''''''''''''''

Applications
--------------------------------



Traditionally, survival analysis was developed to measure lifespans of
individuals. An actuary or health professional would ask questions like
"how long does this population live for, that is, how long between birth
and death?", and answer it using survival analysis. For example, the
population may be a nation's population (for actuaries), or a population
sticken by a disease (in the medical professional's case).
Traditionally, sort of a morbid subject.

The anaylsis can be further applied to not just traditional *births and
deaths*, but any duration. Medical professional might be interested in
the *time between childbirths*, where a birth in this case is the event
of having a child , and a death is becoming pregnant again! (obviously,
we are loose with our definitions of *birth and death*) Another example
is users subscribing to a service: a birth is a user who joins the
service, and a death is when the user leaves the service.

Censorship
--------------------------------


What makes measuring durations difficult is the time itself. At the time
you want to make inferences about durations, it is possible, likely
true, that not all the death events have occured yet. For example, a
medical professional will not wait 50 years for each individual in the
study to pass away before investigating -- he or she is interested in
the effectiveness of improving lifetimes after only a few years.

The individuals in a population who have not been subject to the death
event are labeled as *right-censored*: all the information we have on
these individuals are their current lifetime durations (which is
naturally *less* than their actual lifetimes).

.. note:: There is also left-censorship, where an individuals birth event is not seen. *lifelines* only has estimators for the right-censorship case.

A common mistake data analysts make is simply choosing to ignore the
right-censored individuals. We'll shall see why this is a mistake next:

Consider a case where the population is actually made up of two
subpopulations, :math:`A` and :math:`B`. Population :math:`A` has a very
low lifespan, say 2 time intervals on average, and population :math:`B`
enjoys a much large lifespan, say 12 time intervals on average. Of
course, we do not know this distinction before hand. At :math:`t=10`, we
wish to investigate the average lifespan. Below is an example of such a
situation.

.. code:: python

    
    from lifelines.plotting import plot_lifetimes
    from numpy.random import uniform, exponential
    
    N = 25
    current_time = 10
    actual_lifetimes = np.array([[exponential(12), exponential(2)][uniform()<0.5] for i in range(N)])
    observed_lifetimes = np.minimum(actual_lifetimes,current_time)
    observed= actual_lifetimes < current_time
    
    plt.xlim(0,25)
    plt.vlines(10,0,30,lw=2, linestyles="--")
    plt.xlabel('time')
    plt.title('Births and deaths of our population, at $t=10$')
    plot_lifetimes(observed_lifetimes, event_observed=observed)
    print "Observed lifetimes at time %d:\n"%(current_time), observed_lifetimes


.. image:: SurvivalAnalysisintro_files/SurvivalAnalysisintro_4_0.png


.. parsed-literal::

    Observed lifetimes at time 10:
    [ 10.     1.1    8.    10.     3.43   0.63   6.28   1.03   2.37   6.17  10.
       0.21   2.71   1.25  10.     3.4    0.62   1.94   0.22   7.43   6.16  10.
       9.41  10.    10.  ]


The red lines denote the lifespan of individuals where the death event
has been observed, and the blue lines denote the lifespan of the
right-censored individuals (deaths have not been observed). If we are
asked to estimate the average lifetime of our population, and we naivly
decided to *not* included the right-censored individuals, it is clear
that we would be serverly underestimating the true average lifespan.

Furthermore, if we instead simply took the mean of *all* observed
lifespans, including the current lifespans of right-censored instances,
we would *still* be underestimating the true average lifespan. Below we
plot the actual lifetimes of all instances (recall we do not see this
information at :math:`t=10`).

.. code:: python

    plt.xlim(0,25)
    plt.vlines(10,0,30,lw=2,linestyles="--")
    plot_lifetimes(actual_lifetimes, event_observed=observed)


.. image:: Survival Analysis intro_files/Survival Analysis intro_6_0.png


Survival analysis was originally developed to solve this type of
problem, that is, to deal with estimation when our data is
right-censored. But even in the case where all events have been
observed, i.e. no censorship, survival analysis is also a very useful
too to understand times and durations.

The observations need not always start at zero, either. This was done
only for understanding in the above example. In the service example,
where a customer joining is a birth, a customer can enter observation at
any time, and not necessarily at time zero. In survival analysis, all
times are relative: although individuals may start at different times,
we set them all to start at a single time and record durations from
there. (We actually only need to *duration* of the observation, and not
the start and end time.)

We next introduce the two fundamental objects in survival analysis, the
*survival function* and the *hazard function*.

--------------

Survival function
--------------------------------


Let :math:`T` be the (possibly infinite, but always positive) random
duration taken from the population under observation. For example, the
amount of time a couple is married. Or the time it takes a user to enter
a webpage (and infinite time if they never do). The survival function,
:math:`S(t)`, of a population is defined as

.. math::  S(t) = Pr( T > t) 

i.e., the probability the death event has not occured yet at time
:math:`t`, or equivalently, the probability of surviving until time
:math:`T`. Note the following properties of the survival function:

2. :math:`0 \le S(t) \le 1`
3. :math:`S(0) = 1`
4. :math:`F_T(t) = 1 - S(t)`, where :math:`F_T(t)` is the cumulative
   density function of :math:`T`, which implies
5. :math:`f_T(t) = -S'(t) `

Hazard curve
--------------------------------


We are also interested in the probability of dying in the next instant,
given we haven't expired yet. Mathematically, that is:

.. math::  \lim_{\delta t \rightarrow 0 } \; Pr( t \le T \le t + \delta t | T > t) 

This quantity goes to 0 as :math:`\delta t` shrinks, so we divide this
by the interval :math:`\delta t` (sorta like we do in calculus). This
defines the hazard function at time :math:`t`, :math:`\lambda(t)`:

.. math:: \lambda(t) =  \lim_{\delta t \rightarrow 0 } \; \frac{Pr( t \le T \le t + \delta t | T > t)}{\delta t} 

It can be shown with quite elementary probability that this is equal to:

.. math:: \lambda(t) = \frac{-S'(t)}{S(t)}

and solving this differential equation (yes, it is a differential
equation), we get:

.. math:: S(t) = \exp\left( -\int_0^t \lambda(z) dz \right)

What I love about the above equation is that it defines **all** survival
functions, and because the hazard function is arbitrary (i.e. there is
no parametric form), the entire function is non-parametric (this allows
for very flexible curves). Notice that we can now speak either about the
survival function, :math:`S(t)`, or the hazard function,
:math:`\lambda(t)`, and we can convert back and forth quite easily. It
also gives us another, albeit not very useful, expression for :math:`T`:
Upon differentiation and some algebra, we recover:

.. math:: f_T(t) = \lambda(t)\exp\left( \int_0^t \lambda(z) dz \right)

Of course, we do not observe the true survival curve of a population. We
must use the observed data to estimate it. We also want to continue to
be non-parametric, that is not assume anything more about how the
survival curve looks. The *best* method to recreate the survival
function non-parametrically from the data is known as the Kaplan-Meier
estimate, which brings us to :doc:`estimation using lifelines</Intro to lifelines>`.


.. code:: python

    
