lifelines
===========
<img src="http://i.imgur.com/pwGRqiR.png" height=175 />

[![Latest Version](https://pypip.in/v/lifelines/badge.png)](https://pypi.python.org/pypi/lifelines/)
[![Build Status](https://travis-ci.org/CamDavidsonPilon/lifelines.svg?branch=master)](https://travis-ci.org/CamDavidsonPilon/lifelines)

[What is survival analysis and why should I learn it?](http://lifelines.readthedocs.org/en/latest/Survival%20Analysis%20intro.html)
 Survival analysis was originally developed and applied heavily by the actuarial and medical community. Its purpose was to answer *why do events occur now versus later* under uncertainty (where *events* might refer to deaths, disease remission, etc.). This is great for researchers who are interested in measuring lifetimes: they can answer questions like *what factors might influence deaths?*

But outside of medicine and actuarial science, there are many other interesting and exciting applications of this
lesser-known technique, for example:
- SaaS providers are interested in measuring customer lifetimes, or time to first behaviours
- sociologists are interested in measuring political parties' lifetimes, or relationships, or marriages
- businesses are interested in what variables affect lifetime value

*lifelines* is a pure Python implementation of the best parts of survival analysis. We'd love to hear if you are using *lifelines*, please leave an Issue and let us know your thoughts on the library.

### Installation:


#### Dependencies:

The usual Python data stack: NumPy, SciPy, Pandas (a modern version please), Matplotlib

#### Installing

You can install *lifelines* using

       pip install lifelines


Or getting the bleeding edge version with:

       pip install --upgrade --no-deps git+https://github.com/CamDavidsonPilon/lifelines.git


from the command line.

#### Running the tests

You can optionally run the test suite after install with

    python -m lifelines.tests


### *lifelines* Documentation and an Intro to Survival Analysis

If you are new to survival analysis, wondering why it is useful, or are interested in *lifelines* examples and syntax,
please check out the [Documentation and Tutorials page](http://lifelines.readthedocs.org/en/latest/index.html)


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
