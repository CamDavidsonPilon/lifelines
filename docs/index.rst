.. lifelines documentation master file, created by
   sphinx-quickstart on Sun Feb  2 17:10:21 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lifelines
=====================================

The **lifelines** library provides a powerful tool for data analysts and scientists
looking for methods to solve a common problem:

    How do I predict lifetimes, or time to events?

The statistical tool that answers this question is *survival analysis*.
*lifelines* is a implementation of survival analysis in Python. What
benefits does *lifelines* offer over other survival analysis
implementations?

-  built ontop of Pandas
-  internal plotting methods
-  simple and intuitive API (*designed for humans*)
-  only does survival analysis (No unnecessary features or second-class
   implentations)


Contents:
------------------------------

.. toctree::
  :maxdepth: 2

  Quickstart
  Survival Analysis intro
  Intro to lifelines
  survival_regression
  examples


Installation
------------------------------

Dependencies are from the typical Python data-stack: Numpy, Pandas, Scipy, and Matplotlib. Install using::

    pip install lifelines


Source code and Issue Tracker
------------------------------

Available on Github, `CamDavidsonPilon/lifelines <https://github.com/CamDavidsonPilon/lifelines/>`_
Please report bugs, issues and feature extensions there. 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

