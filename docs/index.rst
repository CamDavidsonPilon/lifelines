.. lifelines documentation master file, created by
   sphinx-quickstart on Sun Feb  2 17:10:21 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: http://i.imgur.com/EOowdSD.png

-------------------------------------


lifelines
=====================================

*lifelines* is an implementation of survival analysis in Python. What
benefits does *lifelines* offer over other survival analysis
implementations?

-  built on top of Pandas
-  internal plotting methods
-  simple and intuitive API
-  only focus is survival analysis
-  handles right, left and interval censored data


Contents:
============

.. toctree::
  :maxdepth: 1
  :caption: Quickstart & Intro

  Quickstart
  Survival Analysis intro

.. toctree::
  :maxdepth: 1
  :caption: Univariate Models

  Survival analysis with lifelines
  jupyter_notebooks/Piecewise Exponential Models and Creating Custom Models.ipynb
  jupyter_notebooks/Modelling time-lagged conversion rates.ipynb

.. toctree::
  :maxdepth: 1
  :caption: Regression Models

  Survival Regression
  jupyter_notebooks/Custom Regression Models.ipynb
  Compatibility with scikit-learn
  Time varying survival regression
  jupyter_notebooks/Proportional hazard assumption.ipynb

.. toctree::
  :maxdepth: 1
  :caption: Additional documentation

  Examples
  References

.. toctree::
  :maxdepth: 1
  :caption: About lifelines

  Changelog
  Development blog <https://dataorigami.net/blogs/napkin-folding/tagged/lifelines>
  Citing lifelines <https://doi.org/10.5281/zenodo.805993>

.. toctree::
  :maxdepth: 1
  :caption: Questions? Suggestions?

  Gitter channel <https://gitter.im/python-lifelines/Lobby>
  Create a GitHub issue <https://github.com/camdavidsonpilon/lifelines/issues>

.. toctree::
  :maxdepth: 1
  :caption: Developer Documentation

  Contributing


Installation
------------------------------


.. code-block:: console

    pip install lifelines

or

.. code-block:: console

    conda install -c conda-forge lifelines



Source code and issue tracker
------------------------------

Available on Github, `CamDavidsonPilon/lifelines <https://github.com/CamDavidsonPilon/lifelines/>`_.
Please report bugs, issues and feature extensions there. We also have `Gitter channel <https://gitter.im/python-lifelines/Lobby>`_ available to discuss survival analysis and *lifelines*:

Citing *lifelines*
------------------------------

The following link will bring you to a page where you can find the latest citation for *lifelines*:

`Citation for lifelines <https://doi.org/10.5281/zenodo.805993>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
