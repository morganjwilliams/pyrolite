Changelog
==============

All notable changes to this project will be documented here.

`Development`_
--------------

.. note:: Changes noted in this subsection are to be released in the next version.
        If you're keen to check something out before its released, you can use a
        `development install <installation.html#development-installation>`__.


`0.1.21`_
--------------

New
~~~~~~~~

* Added parallel coordinate plots: :func:`pyrolite.plot.pyroplot.parallel`

Updates
~~~~~~~~

* Updated :func:`~pyrolite.plot.pyroplot.scatter` and
  :func:`~pyrolite.plot.tern.ternary` to better deal with colormaps
* Updated :mod:`pyrolite.ext.alphamelts` interface:

    * Docs
    * Updated to default to tables with percentages (Wt%, Vol%)
    * Updated :mod:`~pyrolite.ext.alphamelts.plottemplates` y-labels
    * Fixed :mod:`~pyrolite.ext.alphamelts.automation` grid bug

`0.1.20`_
--------------

New
~~~~~~~~

* Stub for DataFrame.pyrochem accessor (yet to be developed)
* Added :func:`pyrolite.util.skl.vis.plot_mapping` for manifold dimensional reduction
* Added :func:`pyrolite.util.skl.vis.alphas_from_multiclass_prob` for visualising
  multi-class classification probabilities in scatter plots

Updates
~~~~~~~~

* Convert reference compositions and normalisation to use a JSON database
* Updated default y-aspect for ternary plots and axes patches
* Updated :mod:`pyrolite.ext.alphamelts.automation`,
  :mod:`pyrolite.ext.alphamelts.meltsfile`, :mod:`pyrolite.ext.alphamelts.tables`
* Updated docs to use :class:`pyrolite.ext.alphamelts.automation.MeltsBatch` with a parameter grid
* Added :mod:`pyrolite.plot.biplot` to API docs

`0.1.19`_
--------------

`0.1.18`_
--------------

`0.1.17`_
--------------

`0.1.16`_
--------------

`0.1.15`_
--------------

`0.1.14`_
--------------

`0.1.13`_
--------------

`0.1.12`_
--------------

`0.1.11`_
--------------

`0.1.10`_
--------------

`0.1.9`_
--------------

`0.1.8`_
--------------

`0.1.7`_
--------------

`0.1.6`_
--------------

`0.1.5`_
--------------

`0.1.4`_
--------------

`0.1.2`_
--------------

`0.1.1`_
--------------

`0.1.0`_
--------------


.. _Development: https://github.com/morganjwilliams/pyrolite/compare/0.1.21...develop
.. _0.1.21: https://github.com/morganjwilliams/pyrolite/compare/0.1.20...0.1.21
.. _0.1.20: https://github.com/morganjwilliams/pyrolite/compare/0.1.19...0.1.20
.. _0.1.19: https://github.com/morganjwilliams/pyrolite/compare/0.1.18...0.1.19
.. _0.1.18: https://github.com/morganjwilliams/pyrolite/compare/0.1.17...0.1.18
.. _0.1.17: https://github.com/morganjwilliams/pyrolite/compare/0.1.16...0.1.17
.. _0.1.16: https://github.com/morganjwilliams/pyrolite/compare/0.1.15...0.1.16
.. _0.1.15: https://github.com/morganjwilliams/pyrolite/compare/0.1.14...0.1.15
.. _0.1.14: https://github.com/morganjwilliams/pyrolite/compare/0.1.13...0.1.14
.. _0.1.13: https://github.com/morganjwilliams/pyrolite/compare/0.1.12...0.1.13
.. _0.1.12: https://github.com/morganjwilliams/pyrolite/compare/0.1.11...0.1.12
.. _0.1.11: https://github.com/morganjwilliams/pyrolite/compare/0.1.10...0.1.11
.. _0.1.10: https://github.com/morganjwilliams/pyrolite/compare/0.1.9...0.1.10
.. _0.1.9: https://github.com/morganjwilliams/pyrolite/compare/0.1.8...0.1.9
.. _0.1.8: https://github.com/morganjwilliams/pyrolite/compare/0.1.7...0.1.8
.. _0.1.7: https://github.com/morganjwilliams/pyrolite/compare/0.1.6...0.1.7
.. _0.1.6: https://github.com/morganjwilliams/pyrolite/compare/0.1.5...0.1.6
.. _0.1.5: https://github.com/morganjwilliams/pyrolite/compare/0.1.4...0.1.5
.. _0.1.4: https://github.com/morganjwilliams/pyrolite/compare/0.1.2...0.1.4
.. _0.1.2: https://github.com/morganjwilliams/pyrolite/compare/0.1.1...0.1.2
.. _0.1.1: https://github.com/morganjwilliams/pyrolite/compare/0.1.0...0.1.1
.. _0.1.0: https://github.com/morganjwilliams/pyrolite/compare/0.0.17...0.1.0
