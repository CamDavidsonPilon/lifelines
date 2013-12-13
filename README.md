lifelines
=======

[![Build Status](https://travis-ci.org/CamDavidsonPilon/lifelines.png)](https://travis-ci.org/CamDavidsonPilon/lifelines)

 
[What is survival analysis and why should I learn it?](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/Tutorial%20and%20Examples.ipynb) Survival analysis was originally developed and applied heavily by the actuarial and medical community. Its purpose was to answer *why do events occur now versus later* under uncertainity (where *events* might refer to deaths, disease remission, etc.). This is great for researchers who are interested in measuring lifetimes: they can answer questions like *what factors might influence deaths?*

But outside of medicine and actuarial science, there are many other interesting and exicting applications of this 
lesser-known technique, for example:
- SaaS providers are interested in measuring customer lifetimes; 
- ecommerce shops are interested the time between first and second order (called *repeat purchase rate*).
- sociologists are interested in measure political parties lifetimes, or relationships, or marriages
- and many others 

*lifelines* is a pure Python implementation of the best parts of survival analysis. We'd love to hear if you use *lifelines*, please ping me at [@cmrn_dp](https://twitter.com/Cmrn_DP) and let me know your 
thoughts on the library. 


![liflines](http://i.imgur.com/QXW71zA.png)

####Dependencies:

The usual Python data stack: numpy, scipy, pandas (a modern version please), matplotlib (optional).

#### Installation:

You can install *lifelines* using 
      
       pip install lifelines

Or getting the bleeding edge version with:

       pip install git+https://github.com/CamDavidsonPilon/lifelines.git

from the command line. 


## (Quick) Intro to *lifelines* and survival analysis

If you are new to survival analysis, wondering why it is useful, or are interested in *lifelines* examples and use,
I recommend running the `Tutorial and Examples.ipynb` notebook in a IPython notebook session. Alternatively, you can [view it online here](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/Tutorial%20and%20Examples.ipynb).


## Documentation

*Work in progress (80%)*

I've added documentation to a notebook, `Documentation.ipynb`, that adds detail to 
the classes, methods and data types. You can use the IPython notebook to view it, or [view it online](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/Documentation.ipynb).

#### More examples

There are some IPython notebook files in the repo, and you can view them online here.

- [Divorce data](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/datasets/Divorces%2520Rates.ipynb)
- [Gehan's survival dataset](http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/lifelines/master/datasets/The%2520Gehan%2520Survival%2520Data.ipynb)




