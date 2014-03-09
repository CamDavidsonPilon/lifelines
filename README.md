lifelines
===========
<img src="http://i.imgur.com/pwGRqiR.png" height=175 />

[![Latest Version](https://pypip.in/v/lifelines/badge.png)](https://pypi.python.org/pypi/lifelines/)

[What is survival analysis and why should I learn it?](http://lifelines.readthedocs.org/en/latest/Survival%20Analysis%20intro.html)
 Survival analysis was originally developed and applied heavily by the actuarial and medical community. Its purpose was to answer *why do events occur now versus later* under uncertainity (where *events* might refer to deaths, disease remission, etc.). This is great for researchers who are interested in measuring lifetimes: they can answer questions like *what factors might influence deaths?*

But outside of medicine and actuarial science, there are many other interesting and exciting applications of this 
lesser-known technique, for example:
- SaaS providers are interested in measuring customer lifetimes, or time to first behaviours.
- sociologists are interested in measure political parties lifetimes, or relationships, or marriages
- Businesses are interested in what variables affect lifetime value

*lifelines* is a pure Python implementation of the best parts of survival analysis. We'd love to hear if you are using *lifelines*, please ping me at [@cmrn_dp](https://twitter.com/Cmrn_DP) and let me know your 
thoughts on the library. 

## Installation:


####Dependencies:

The usual Python data stack: Numpy, Scipy, Pandas (a modern version please), Matplotlib

#### Installing

You can install *lifelines* using 
      
       pip install lifelines

Or getting the bleeding edge version with:

       pip install git+https://github.com/CamDavidsonPilon/lifelines.git

or upgrade with 

       pip install --upgrade git+https://github.com/CamDavidsonPilon/lifelines.git


from the command line. 


## Intro to *lifelines* and survival analysis
    
Situation: 500 random individuals are born at time 0, currently it is time 12, so we have possibly not observed all death events yet.

    # Create lifetimes, but censor all lifetimes after time 12
    censor_after = 12
    actual_lifetimes = np.random.exponential(10, size=500)
    observed_lifetimes = np.minimum( actual_lifetimes, censor_after*np.ones(500) )
    C = (actual_lifetimes < censor_after) #boolean array

Non-parametrically fit the *survival curve*:

    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(observed_lifetimes, censorship=C) 

    # fitter methods have an internal plotting method.
    # plot the curve with the confidence intervals
    kmf.plot()

![kmf](http://i.imgur.com/Bq73IfN.png)

It looks like 50% of all individuals are dead before time 7.

    print kmf.survival_function_.head()

    time            KM-estimate
    0.000000        1.000
    0.038912        0.998
    0.120667        0.996
    0.125719        0.994
    0.133778        0.992

Non-parametrically fit the *cumulative hazard curve*:

    from lifelines import NelsonAalenFitter

    naf = NelsonAalenFitter()
    naf.fit(observed_lifetimes, censorship=C) 

    #plot the curve with the confidence intervals
    naf.plot()

![naf](http://i.imgur.com/2L7arWX.png)

    print naf.cumulative_hazard_.head()

    time       NA-estimate
    0.000000     0.000000
    0.038912     0.002000
    0.120667     0.004004
    0.125719     0.006012
    0.133778     0.008024

Compare two populations using the logrank test:

    from lifelines.statistics import logrank_test
    other_lifetimes = np.random.exponential(3, size=500)

    summary, p_value, results = logrank_test(observed_lifetimes, other_lifetimes, alpha=0.95)
    print summary

    
    Results
       df: 1
       alpha: 0.95
       t 0: -1
       test: logrank
       null distribution: chi squared

       __ p-value ___|__ test statistic __|__ test results __
             0.00000 |              268.465 |     True    
    

## (Less Quick) Intro to *lifelines* and survival analysis

If you are new to survival analysis, wondering why it is useful, or are interested in *lifelines* examples and syntax,
please check out the [Documentation and Tutorials page](http://lifelines.readthedocs.org/en/latest/index.html)

Alternatively, you can **use the IPython notebooks tutorials**, located in the main directory of the repo:

1. [Introduction to survival analysis](http://nbviewer.ipython.org/github/CamDavidsonPilon/lifelines/blob/master/Survival%20Analysis%20intro.ipynb)
2. [Using lifelines on real data](http://nbviewer.ipython.org/github/CamDavidsonPilon/lifelines/blob/master/Intro%20to%20lifelines.ipynb) 


#### More examples

There are some IPython notebook files in the repo, and you can view them online here.

- [Divorce data](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/datasets/Divorces%2520Rates.ipynb)
- [Gehan's survival dataset](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/datasets/The%2520Gehan%2520Survival%2520Data.ipynb)


![lifelines](http://i.imgur.com/QXW71zA.png)


## License

The Feedback MIT License (FMIT) 

Copyright (c) 2013, Cameron Davidson-Pilon

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

2. Person obtaining a copy must return feedback to the authors.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


*lifelines* logo designed by Pulse designed by TNS from the Noun Project
