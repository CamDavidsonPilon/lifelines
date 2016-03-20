![logo](http://i.imgur.com/EOowdSD.png)

[![PyPI version](https://badge.fury.io/py/lifelines.svg)](https://badge.fury.io/py/lifelines)
[![Build Status](https://travis-ci.org/CamDavidsonPilon/lifelines.svg?branch=master)](https://travis-ci.org/CamDavidsonPilon/lifelines)
[![Coverage Status](https://coveralls.io/repos/github/CamDavidsonPilon/lifelines/badge.svg?branch=master)](https://coveralls.io/github/CamDavidsonPilon/lifelines?branch=master)

[What is survival analysis and why should I learn it?](http://lifelines.readthedocs.org/en/latest/Survival%20Analysis%20intro.html)
 Survival analysis was originally developed and applied heavily by the actuarial and medical community. Its purpose was to answer *why do events occur now versus later* under uncertainty (where *events* might refer to deaths, disease remission, etc.). This is great for researchers who are interested in measuring lifetimes: they can answer questions like *what factors might influence deaths?*

But outside of medicine and actuarial science, there are many other interesting and exciting applications of this
lesser-known technique, for example:
- SaaS providers are interested in measuring customer lifetimes, or time to first behaviours
- sociologists are interested in measuring political parties' lifetimes, or relationships, or marriages
- analysing [Godwin's law](https://raw.githubusercontent.com/lukashalim/GODWIN/master/Kaplan-Meier-Godwin.png) in Reddit comments
- A/B tests to determine how long it takes different groups to perform an action.

*lifelines* is a pure Python implementation of the best parts of survival analysis. We'd love to hear if you are using *lifelines*, please leave an Issue and let us know your thoughts on the library.

### Installation:
#### Dependencies:

The usual Python data stack: NumPy, SciPy, Pandas (a modern version please). Matplotlib is optional (as of 0.6.0+). 

#### Installing

You can install *lifelines* using

       pip install lifelines


Or getting the bleeding edge version with:

       pip install --upgrade --no-deps git+https://github.com/CamDavidsonPilon/lifelines.git

from the command line.

##### Installation Issues?

See the common [problems/solutions for installing lifelines](https://github.com/CamDavidsonPilon/lifelines/issues?utf8=%E2%9C%93&q=label%3Ainstallation+).

#### Running the tests

You can optionally run the test suite after install with

    py.test


### *lifelines* Documentation and an Intro to Survival Analysis

If you are new to survival analysis, wondering why it is useful, or are interested in *lifelines* examples and syntax,
please check out the [Documentation and Tutorials page](http://lifelines.readthedocs.org/en/latest/index.html)


### Citing lifelines

 - Davidson-Pilon, C., Lifelines, (2016), Github repository, https://github.com/CamDavidsonPilon/lifelines 
 
or with bibTex here: 

```
@misc{Lifelines,
  author = {C., Davidson-Pilon},
  title = {Lifelines},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/camdavidsonpilon/lifelines}},
  commit = {latest_commit}
}
```
![lifelines](http://i.imgur.com/QXW71zA.png)
