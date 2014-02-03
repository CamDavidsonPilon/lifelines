<img src="http://i.imgur.com/pwGRqiR.png" title="Hosted by imgur.com" height=225/>

lifelines
===========

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

    #fake data
    #T are the durations, C are boolean censorships
    T = np.random.exponential(1, size=500)
    C = np.random.binomial(1, 0.5, size=500)

Non-parametrically fit the survival curve:

    from lifelines.estimation import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(T, C) 
    kmf.plot() #plot the curve with the confidence intervals
    print kmf.survival_function_.head()
    print kmf.confidence_interval_.head()

Non-parametrically fit the cumulative hazard curve:

    from lifelines.estimation import  NelsonAalenFitter

    naf = NelsonAalenFitter()
    naf.fit(T, C) 
    naf.plot()
    print naf.cumulative_hazard_.head()

Compare two populations using the logrank test:

    from lifelines.statistics import logrank_test

    T2 = np.random.exponential(3, size=500)

    summary, p_value, results = logrank_test(T, T2, alpha=0.95)
    print s

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
