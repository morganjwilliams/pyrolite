.. raw:: latex

   \chapter{Introduction}

pyrolite
==========

  pyrolite is a set of tools for making the most of your geochemical data.

The python package includes functions to work with compositional data, to transform
geochemical variables (e.g. elements to oxides), functions for common plotting
tasks (e.g. spiderplots, ternary diagrams, bivariate and ternary density diagrams),
and numerous auxiliary utilities.

- On this site you can browse the  `API <./api/API.html>`__, or look
  through some of the `usage examples <./examples/index.html>`__.

- There's also a quick `installation guide <./installation.html>`__, a list of
  `recent changes <./dev/changelog.html>`__ and some notes on
  where the project is heading in the near `future <./dev/future.html>`__.

- If you're interested in `contributing to the project <./dev/contributing.html>`__, there are
  many potential avenues, whether you're experienced with python or not.

.. note:: pyrolite has recently been
          `published in the Journal of Open Source Software <https://joss.theoj.org/papers/10.21105/joss.02314>`__!

Why *pyrolite*?
..................

The name *pyrolite* is an opportunistic repurposing of a term used to describe an early
model mantle composition proposed by Ringwood [Ringwood1962]_, comprised principally
of **pyr**-oxene & **ol**-ivine. While the model certainly hasn't stood the test of time,
the approach optimises the aphorism "All models are wrong, but some are useful"
[Box1976]_. It is with this mindset that pyrolite is built, to better enable you to
make use of your geochemical data to build and test geological models.

.. [Ringwood1962] Ringwood, A.E. (1962). A model for the upper mantle.
    Journal of Geophysical Research (1896-1977) 67, 857–867.
    `doi: 10.1029/JZ067i002p00857 <https://doi.org/10.1029/JZ067i002p00857>`__

.. [Box1976] Box, G.E.P. (1976). Science and Statistics.
    Journal of the American Statistical Association 71, 791–799.
    `doi: 10.1080/01621459.1976.10480949 <https://doi.org/10.1080/01621459.1976.10480949>`__


.. raw:: latex

   \chapter{Getting Started}

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    installation
    gettingstarted
    examples/index
    tutorials/index
    cite

.. raw:: latex

    \chapter{Development}

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Development

    dev/development
    dev/changelog
    dev/future
    dev/conduct
    dev/contributing
    dev/contributors

.. raw:: latex

    \chapter{Reference}

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api/API
   data/index
   ext/extensions
