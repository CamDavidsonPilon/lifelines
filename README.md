lifelines
=======
 
What is survival analysis and why should I learn it? Historically, survival analysis has been developed and applied most heavily by the actuarial and medical community. Generally, its purpose is to answer *why do events occur now versus later* under uncertainity and censoring (where *events* might be deaths, disease remission, etc.). This is great for researchers who are interested in measuring lifetimes: they can answer *what factors might influence deaths?*

There is another use of survival analysis: customers subscribing to services -- births users joining and deaths are users leaving. Telcom companies have understood this for years, but kept it in-house, but recently and specifically SaaS providers and app developers are understanding the benefits of survival analysis. 

####Dependencies:

The usual Python data stack: numpy, pandas, matplotlib (optional)


### Enough talk - just show me the examples!

    %pylab
    from lifelines.generate_datasets import *
    from lifelines.estimation import *

    n_ind = 4 # how many lifetimes do we observe
    n_dim = 5 # the number of covarites to generate. 
    t = np.linspace(0,40,400)

    hz, coefs, covart = generate_hazard_rates(n_ind, n_dim, t, model="aalen")
    # you're damn right these are dataframes

    hz.plot()

![Hazard Rates](http://i.imgur.com/O8Og76O.png)

    sv = construct_survival_curves(hz, t )
    sv.plot() #moar dataframes

![Survival Curves](http://i.imgur.com/jWu3CM9.png)

    #using the hazard curves, we can sample from survival times.
    rv = generate_random_lifetimes(hz, t, 50 )
    print rv
    array([[ 9.4235589 ,  3.60902256,  3.0075188 ,  0.60150376],
           [ 1.00250627,  3.20802005,  0.70175439,  0.30075188],
           [ 5.71428571,  8.02005013,  5.41353383,  0.30075188],
           ...,
           [ 3.70927318,  4.41102757,  3.30827068,  0.30075188],
           [ 1.80451128,  1.5037594 ,  0.30075188,  0.40100251],
           [ 1.40350877,  1.5037594 ,  0.80200501,  0.10025063]])

    survival_times = rv[:,0][:,None]  

    #estimation is clean and built to resemble scikit learn's api.
    kmf = KaplanMeierFitter()
    kmf.fit(survival_times)
    kmf.survival_function_.plot()

![KaplanMeier estimate](http://i.imgur.com/aztRkvl.png)

    naf = NelsonAalenFitter()
    naf.fit(survival_times)
    naf.cumulative_hazard_.plot()

![NelsonAalen](http://i.imgur.com/xA9OBFN.png)


### Censorship events
When there are right-censored events, the simplest case being there are still surviving individuals, we need to be more careful and factor these 
non-observed individuals in. The api for this is an obvious extension from above:


    t = np.linspace(0,40,1000)
    hz, coefs, covart = generate_hazard_rates(1, 2, t, model="aalen")

    #generate random lifetimes with uniform censoring. C is the boolean of censorship
    T, C = generate_random_lifetimes(hz, t, size=750, censor=True )

In the above line, `C` is a boolean array with `True` iff we observed the death event, otherwise, they individual was right-censored. `T` is the death event, or if censored, the most lifespan before censorship.  


    kmf = KaplanMeierFitter()
    kmf.fit(T,t,censorship=C) #add in the censorship here

    #plot it
    ax = kmf.survival_function_.plot()
    sv = construct_survival_curves(hz,t) 
    sv.plot(ax=ax) 

    ##what if we had ignored the censorship events?
    kmf.fit(T,t, column_name="KM-estimate without factoring censorship")
    kmf.survival_function_.plot(ax=ax)

    plt.show()

![SVest](http://i.imgur.com/jYm911Z.png)


### Plotting 

We can visualize the lifetimes too (really only good for data checking for small samples)

    from lifelines.plotting import plot_lifetimes
    N = 20
    current = 10
    birthtimes = current*np.random.uniform(size=(N,))
    T, C= generate_random_lifetimes(hz, t, size=N, censor=current - birthtimes )
    plot_lifetimes(T, censorship=C, birthtimes=birthtimes)

![lifetimes](http://i.imgur.com/JDt3t3X.png)


## Moar examples?

There are some IPython notebook files in the repo, and you can view them online here:

- [Divorce data](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/Lifelines%2520Examples.ipynb)
- [Gehan's survival dataset](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/The%2520Gehan%2520Survival%2520Data.ipynb)

