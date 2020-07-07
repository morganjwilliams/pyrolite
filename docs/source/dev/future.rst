Future
========

This page details some of the under-development and planned features for
:mod:`pyrolite`. Note that while no schedules are attached, features under development
are likely to be completed with weeks to months, while those 'On The Horizon' may be
significantly further away (or in some cases may not make it to release).

Under Development
-------------------

These features are either under development or planned to be implemented
and should be released in the near future.

:mod:`pyrolite.mineral`

  There are a few components which will make better use of mineral chemistry data,
  and facilitate better comparison of whole-rock and mineral data
  (`Issue #5 <https://github.com/morganjwilliams/pyrolite/issues/5>`__):

    * Normative mineral calculations
    * Mineral formulae recalculation, site partitioning, endmember calculations

:mod:`pyrolite.geochem.isotope`

  * Stable isotope calculations
  * Simple radiogenic isotope system calculations and plots

:mod:`pyrolite.comp.impute`

  Expansion of compositional data imputation algorithms beyond EMCOMP
  (`Issue #6 <https://github.com/morganjwilliams/pyrolite/issues/6>`__).

:mod:`pyrolite.util.spatial`

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

:mod:`pyrolite.geochem.magma`

  Utilities for simple melting and fractionation models.

:code:`pyrolite.geochem.quality`

  Utilities for:
    * assessing data quality
    * identifying potential analytical artefacts
    * assessing uncertainties

* **Interactive Plotting Backend Options**: :mod:`pyrolite` visualisation is currently
  based entirely on static plot generation via :mod:`matplotlib`. While this works
  well for publication-style figures, it may be possible to leverage :mod:`pandas`-based
  frameworks to provide options for alternative backends, some of which are more
  interactive and amendable to data exploration (e.g. :mod:`hvplot`). We'll look into
  the feasibility of this in the near future. See the :mod:`pandas` extension docs for
  one option for implementing this
  (`plotting-backends <https://pandas.pydata.org/pandas-docs/stable/development/extending.html#plotting-backends>`__).


Governance and Documentation
------------------------------

* Depending on how the community grows, and whether :mod:`pyrolite` brings with it
  a series of related tools, the project and related tools may be migrated to an
  umbrella organization on GitHub (e.g. pyrolite/pyrolite) so they can be
  collectively managed by a community.

* **Internationalization**: While the pyrolite source is documented in English,
  it would be good to be able to provide translated versions of the documentation
  to minimise hurdles to getting started.

* **Teaching Resources**: :mod:`pyrolite` is well placed to provide solutions
  and resources for use in under/post-graduate education. While we have documentation
  sections dedicated to examples and tutorials, perhaps we could develop explicit
  sections for educational resources and exercises.
