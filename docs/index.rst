.. lifelines documentation master file, created by
   sphinx-quickstart on Sun Feb  2 17:10:21 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: http://i.imgur.com/EOowdSD.png

-------------------------------------


Lifelines
=====================================

*lifelines* is a implementation of survival analysis in Python. What
benefits does *lifelines* offer over other survival analysis
implementations?

-  built on top of Pandas
-  internal plotting methods
-  simple and intuitive API (*designed for humans*)
-  only does survival analysis (No unnecessary features or second-class
   implementations)


Contents:
------------------------------

.. toctree::
  :maxdepth: 2

  Quickstart
  Survival Analysis intro
  Survival analysis with lifelines
  Survival Regression
  jupyter_notebooks/Proportional hazard assumption.ipynb
  jupyter_notebooks/Cox residuals.ipynb
  jupyter_notebooks/Piecewise Exponential Models and Creating Custom Models.ipynb
  jupyter_notebooks/Modelling time-lagged conversion rates.ipynb
  Examples


Installation
------------------------------

Dependencies are from the typical Python data-stack: Numpy, Pandas, Scipy, and optionally Matplotlib. Install using:

.. code-block:: console

    pip install lifelines


Source Code and Issue Tracker
------------------------------

Available on Github, `CamDavidsonPilon/lifelines <https://github.com/CamDavidsonPilon/lifelines/>`_.
Please report bugs, issues and feature extensions there. We also have `Gitter channel <https://gitter.im/python-lifelines/Lobby>`_ avaiable to discuss survival analysis and *lifelines*:

Citing lifelines
------------------------------

The following link will bring you to a page where you can find the latest citation for lifelines:

`Citation for lifelines <http://doi.org/10.5281/zenodo.1252342>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
