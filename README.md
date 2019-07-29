![](http://i.imgur.com/EOowdSD.png)

[![PyPI version](https://badge.fury.io/py/lifelines.svg)](https://badge.fury.io/py/lifelines)
[![Build Status](https://travis-ci.org/CamDavidsonPilon/lifelines.svg?branch=master)](https://travis-ci.org/CamDavidsonPilon/lifelines)
[![Coverage Status](https://coveralls.io/repos/github/CamDavidsonPilon/lifelines/badge.svg?branch=master)](https://coveralls.io/github/CamDavidsonPilon/lifelines?branch=master)
[![Join the chat at https://gitter.im/python-lifelines/Lobby](https://badges.gitter.im/python-lifelines/Lobby.svg)](https://gitter.im/python-lifelines/Lobby)
[![DOI](https://zenodo.org/badge/12420595.svg)](https://zenodo.org/badge/latestdoi/12420595)


[What is survival analysis and why should I learn it?](http://lifelines.readthedocs.org/en/latest/Survival%20Analysis%20intro.html)
 Survival analysis was originally developed and applied heavily by the actuarial and medical community. Its purpose was to answer *why do events occur now versus later* under uncertainty (where *events* might refer to deaths, disease remission, etc.). This is great for researchers who are interested in measuring lifetimes: they can answer questions like *what factors might influence deaths?*

But outside of medicine and actuarial science, there are many other interesting and exciting applications of this survival analysis. For example:
- SaaS providers are interested in measuring subscriber lifetimes, or time to some first action
- inventory stock out is a censoring event for true "demand" of a good.
- sociologists are interested in measuring political parties' lifetimes, or relationships, or marriages
- analyzing [Godwin's law](https://raw.githubusercontent.com/lukashalim/GODWIN/master/Kaplan-Meier-Godwin.png) in Reddit comments
- A/B tests to determine how long it takes different groups to perform an action.

*lifelines* is a pure Python implementation of the best parts of survival analysis. We'd love to hear if you are using *lifelines*, please leave an Issue and let us know your thoughts on the library.

## Installation:

You can install *lifelines* using

       pip install lifelines

Or getting the bleeding edge version with:

       pip install --upgrade --no-deps git+https://github.com/CamDavidsonPilon/lifelines.git

from the command line.

### Installation Issues?

See the common [problems/solutions for installing lifelines](https://github.com/CamDavidsonPilon/lifelines/issues?utf8=%E2%9C%93&q=label%3Ainstallation+).


## *lifelines* documentation and an intro to survival analysis

If you are new to survival analysis, wondering why it is useful, or are interested in *lifelines* examples, API, and syntax, please check out the [Documentation and Tutorials page](http://lifelines.readthedocs.org/en/latest/index.html)

Example:
```python
from lifelines import KaplanMeierFitter

durations = [11, 74, 71, 76, 28, 92, 89, 48, 90, 39, 63, 36, 54, 64, 34, 73, 94, 37, 56, 76]
event_observed = [True, True, False, True, True, True, True, False, False, True, True,
                  True, True, True, True, True, False, True, False, True]

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)
kmf.plot()
```

<img src="https://imgur.com/d4Gi5J0.png" width="600">

## Contacting & troubleshooting
 - There is a [Gitter](https://gitter.im/python-lifelines/) channel available.
 - Some users have posted common questions at [stats.stackexchange.com](https://stats.stackexchange.com/search?tab=votes&q=%22lifelines%22%20is%3aquestion)
 - creating an issue in the [Github repository](https://github.com/camdavidsonpilon/lifelines).

## Roadmap
You can find the roadmap for lifelines [here](https://www.notion.so/camdp/6e2965207f564eb2a3e48b5937873c14?v=47edda47ab774ca2ac7532bb0c750559).

## Development

See our [Contributing](https://github.com/CamDavidsonPilon/lifelines/blob/master/CONTRIBUTING.md) guidelines.

-------------------------------------------------------------------------------

## Citing lifelines

You can use this badge below to generate a DOI and reference text for the latest related version of lifelines:

 [![DOI](https://zenodo.org/badge/12420595.svg)](https://zenodo.org/badge/latestdoi/12420595)
