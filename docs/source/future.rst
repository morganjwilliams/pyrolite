Future
========

This page details some of the under-development and planned features for
:code:`pyrolite`. Note that while no schedules are attached, features under development
are likely to be completed with weeks to months, while those 'On The Horizon' may be
significantly further away (or in some cases may not make it to release).

Under Development
-------------------

These features are under development and should be released in the near future.

:code:`pyrolite.mineral`

  There are a few components which will make better use of mineral chemistry data,
  and facilitate better comparison of whole-rock and mineral data
  (`Issue #5 <https://github.com/morganjwilliams/pyrolite/issues/5>`__):

    * Normative mineral calculations
    * Mineral formulae recalculation, site partitioning, endmember calculations

:code:`pyrolite.util.spatial`

  Expansion of current minor utilities to a broader suite.
  Spatial in the 'from here to there' sense, but also the geometric sense.
  Questions along the lines of 'how similar or different are these things', central to
  many applications of geochemistry, fall into this spatial category.
  A few key components and applications include:

    * Angular distance (spherical geometry) for incorporating lat-long distances,
      including for (distance-) weighted bootstrap resampling
    * Compositional distances
    * Probability density distribution/histogram comparison

  .. note:: This project isn't intended as a geospatial framework; for that there are
            many great offerings already! As such you won't see much in the way of
            geospatial or geostatistical functionality here.


On the Horizon, Potential Future Updates
----------------------------------------

These are a number of features which are in various stages of development, which are
planned be integrated over the longer term.

:code:`pyrolite.comp.impute`

  Expansion of compositional data imputation algorithms beyond EMCOMP
  (`Issue #6 <https://github.com/morganjwilliams/pyrolite/issues/6>`__).

:code:`pyrolite.geochem.isotope`

  * Stable isotope calculations
  * Simple radiogenic isotope system calculations and plots

:code:`pyrolite.plot`

  A few improvements to plotting functionality are on the horizon, including native
  integration of a ternary projection for :code:`matplotlib`.

:code:`pyrolite.geochem.magma`

  Utilities for simple melting and fractionation models.

:code:`pyrolite.geochem.quality`

  Utilities for:
    * assessing data quality
    * identifying potential analytical artefacts
    * assessing uncertainties

:code:`pyrolite.util.melts`

  * Wrapper for the :code:`alphaMELTS` executable
  * Links to *under-development* python-MELTS.

  .. note:: There are some great things happening on the MELTS-for-scripting front;
            these utilities will be intended to link your data to these tools.
