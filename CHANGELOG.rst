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

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~

* Added parallel coordinate plots: :func:`pyrolite.plot.pyroplot.parallel`
* Updated :func:`~pyrolite.plot.pyroplot.scatter` and
  :func:`~pyrolite.plot.tern.ternary` to better deal with colormaps

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated :mod:`pyrolite.ext.alphamelts` interface:

    * Docs
    * Updated to default to tables with percentages (Wt%, Vol%)
    * Updated :mod:`~pyrolite.ext.alphamelts.plottemplates` y-labels
    * Fixed :mod:`~pyrolite.ext.alphamelts.automation` grid bug

`0.1.20`_
--------------

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~

* Stub for :class:`pyrolite.geochem.pyrochem` accessor (yet to be fully developed)
* Convert reference compositions within of :mod:`pyrolite.geochem.norm` to use a JSON database

:mod:`pyrolite.util.skl`
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.util.skl.vis.plot_mapping` for manifold dimensional reduction
* Added :func:`pyrolite.util.skl.vis.alphas_from_multiclass_prob` for visualising
  multi-class classification probabilities in scatter plots

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.plot.biplot` to API docs
* Updated default y-aspect for ternary plots and axes patches

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated :mod:`pyrolite.ext.alphamelts.automation`,
  :mod:`pyrolite.ext.alphamelts.meltsfile`, :mod:`pyrolite.ext.alphamelts.tables`
* Updated docs to use :class:`pyrolite.ext.alphamelts.automation.MeltsBatch` with a parameter grid


`0.1.19`_
--------------

* Added this changelog
* Require :mod:`pandas` >= v0.23 for DataFrame accessors

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Moved normalisation into :mod:`pyrolite.geochem`
* Improved support for molecular-based calculations in :mod:`pyrolite.geochem`
* Added :mod:`pyrolite.geochem` section to API docs
* Added the :func:`~pyrolite.geochem.convert_chemistry` docs example

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Improvements for :mod:`pyrolite.ext.alphamelts.download`
* Completed :mod:`pyrolite.ext.alphamelts.automation.MeltsBatch`
* Added the :mod:`pyrolite.ext.alphamelts.web` docs example
* Added :mod:`pyrolite.ext.alphamelts.plottemplates` to API docs
* Added :func:`pyrolite.ext.alphamelts.tables.write_summary_phaselist`
* Added :func:`pyrolite.ext.alphamelts.automation.exp_name` for automated alphaMELTS
  experiment within batches

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~
* Added :class:`pyrolite.util.meta.ToLogger` output stream for logging
* Added :func:`pyrolite.util.multip.combine_choices` for generating parameter
  combination grids

`0.1.18`_
--------------

* Require :mod:`scipy` >= 1.2

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~

* Automatic import of dataframe accessor `df.pyroplot` removed;
  import :mod:`pyrolite.plot` to use :class:`pyrolite.plot.pyroplot` dataframe accessor
* Updated label locations for :mod:`pyrolite.plot.biplot`
* Default location of the y-axis updated for :func:`pyrolite.plot.stem.stem`

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added stub for :mod:`pyroilte.geochem.qualilty`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~

* Moved `pyrolite.classification` to :mod:`pyrolite.util.classification`
* Added :func:`pyrolite.util.plot.marker_cycle`

`0.1.17`_
--------------

* Update status to Beta

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~

* Added database for geochemical components (`geochemdb.json`) for faster import
  via :func:`~pyrolite.geochem.ind.common_elements` and
  :func:`~pyrolite.geochem.ind.common_oxides`
* Added stub for :mod:`pyrolite.geochem.isotope`
* Update to using :func:`pyrolite.util.transform.aggregate_element` rather
  than `aggregate_cation`

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~

* Expanded use of :mod:`pyrolite.plot.pyroplot` dataframe accessor
* Added :func:`pyrolite.plot.pyrochem.cooccurence`
* Added :mod:`pyrolite.plot.biplot`
* Added support for conditional density spiderplots
  within :func:`~pyrolite.plot.spider.spider` and :func:`~pyrolite.plot.spider.REE_v_radii`
* Updated keyword argument parsing for :func:`~pyrolite.plot.spider.spider`

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Removed automatic import of mineral structures to reduce delay
* Updated :func:`pyrolite.mineral.lattice.strain_coefficient`
* Added stub for :func:`pyrolite.mineral.normative`
* Updated :class:`pyrolite.mineral.sites.Site`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~
* Added functions for interpolating paths and patches (e.g. contours) and exporting
  these:
  :func:`~util.plot.interpolate_path`, :func:`~util.plot.interpolated_patch_path`,
  :func:`~util.plot.get_contour_paths`, :func:`~util.plot.path_to_csv`
* Added :func:`util.plot._mpl_sp_kw_split`
* Added :func:`util.text.remove_suffix`
* Added :func:`util.text.int_to_alpha`

:mod:`pyrolite.ext`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated alphaMELTS interface location to external package interface rather than
  utility  (from :mod:`pyrolite.util` to :mod:`pyrolite.ext`)
* Added :mod:`pyrolite.ext.datarepo` stub

`0.1.16`_
--------------

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.mineral.lattice` example
* Added :func:`pyrolite.mineral.lattice.youngs_modulus_approximation`

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.ext.alphamelts` Monte Carlo uncertainty estimation example
* Added :func:`pyrolite.ext.alphamelts.automation.MeltsExperiment.callstring` to
  facilitate manual reproducibility of pyrolite calls to alphaMELTS.
* Improved alphaMELTS interface termination
* Added :func:`pyrolite.ext.alphamelts.plottemplates.phase_linestyle` to for auto-differentiated
  linestyles in plots generated from alphaMELTS output tables
* Added :func:`pyrolite.ext.alphamelts.plottemplates.table_by_phase` to generate axes
  per phase from a specific output table

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added MORB compositions from Gale et al. (2013) to Reference Compositions
* Updated `pyrolite.geochem.ind.get_radii` to :func:`pyrolite.geochem.ind.get_ionic_radii`
* :code:`dropPm` parameter added to :func:`pyrolite.geochem.ind.REE`

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~

* Updated `pyrolite.plot.spider.REE_radii_plot` to :func:`pyrolite.plot.spider.REE_v_radii`
* Updated :func:`pyrolite.util.meta.steam_log` to take into account active logging
  handlers

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.util.pd.drop_where_all_empty`
* Added :func:`pyrolite.util.pd.read_table` for simple :code:`.csv` and :code:`.xlsx`/:code:`.xls` imports
* Added :func:`pyrolite.util.plot.rect_from_centre`
* Added :func:`pyrolite.util.text.slugify` for removing spaces and non-alphanumeric characters

`0.1.15`_
--------------

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Bugfixes for :mod:`~pyrolite.ext.alphamelts.automation` and :mod:`~pyrolite.ext.alphamelts.download`
* Add a :code:`permissions` keyword argument to :func:`pyrolite.util.general.copy_file`

`0.1.14`_
--------------

* Added Contributor Covenant Code of Conduct

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.plot.stem.stem` example
* Added :mod:`pyrolite.plot.stem`
* Added :mod:`pyrolite.plot.stem` to API docs
* Added :mod:`pyrolite.plot.stem` example

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.mineral.lattice` for lattice strain calculations
* Added :mod:`pyrolite.mineral` to API docs

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~

* Improved :mod:`pyrolite.ext.alphamelts.automation` workflows, process tracking and
  termination
* Incorporated :class:`~pyrolite.ext.alphamelts..automation.MeltsProcess` into
  :class:`~pyrolite.ext.alphamelts.automation.MeltsExperiment`
* Added :class:`~pyrolite.ext.alphamelts.automation.MeltsBatch` stub
* Added :func:`~pyrolite.ext.alphamelts.meltsfile.read_meltsfile` and
  :func:`~pyrolite.ext.alphamelts.meltsfile.read_envfile`
* Added :mod:`pyrolite.ext.alphamelts.plottemplates`
* Added :func:`pyrolite.ext.alphamelts.tables.get_experiments_summary` for aggregating
  alphaMELTS experiment results across folders

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~

* Added manifold uncertainty example with :func:`pyrolite.util.skl.vis.plot_mapping`
* Updated :mod:`pyrolite.util.ditributions.norm_to_lognorm`
* Added :func:`pyrolite.util.general.get_process_tree` to extract related processes
* Added :func:`pyrolite.util.pd.zero_to_nan`


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
