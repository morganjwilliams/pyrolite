Changelog
=============

All notable changes to this project will be documented here.

Todo
------

* **Feature**: Docs page for reference compositions
  (`#38 <https://github.com/morganjwilliams/pyrolite/issues/38>`__).
  Work in progress on a
  `feature branch <https://github.com/morganjwilliams/pyrolite/tree/feature/docs-pyrolite.data>`__.
* **Feature**: REY function
  (`#35 <https://github.com/morganjwilliams/pyrolite/issues/35>`__).
* **Feature**: Updates to include more lithogeochemical plot templates
  (`#26 <https://github.com/morganjwilliams/pyrolite/issues/26>`__)
* **Bug**: Conditional density spider plots should have bins centred on the element indexes
  (currently this is an edge)
* **Feature**: Index memory for :func:`~pyrolite.plot.spider.spider`
  (`#27 <https://github.com/morganjwilliams/pyrolite/issues/27>`__)

`Development`_
--------------

.. note:: Changes noted in this subsection are to be released in the next version.
        If you're keen to check something out before its released, you can use a
        `development install <development.html#development-installation>`__.


`0.2.6`_
--------------

* **New Contributors**: `Kaarel Mand <https://github.com/kaarelmand>`__ and
  `Laura Miller <https://github.com/Lauraanme>`__
* **PR Merged**: `Louise Schoneveld <https://github.com/lavender22>`__ submitted
  a pull request to fill out the newly-added
  `Formatting and Cleaning Up Plots tutorial <https://pyrolite.readthedocs.io/en/develop/tutorials/plot_formatting.html>`__.
  This tutorial aims to provide some basic guidance for common figure and axis
  formatting tasks as relevant to :mod:`pyrolite`.
* Added `codacy` for code quality checking, and implemented numerous clean-ups
  and a few new tests across the package.
* Performance upgrades, largely for the documentation page.
  The docs page should build and load faster, and have less memory hang-ups -
  due to smaller default image sizes/DPI.
* Removed dependency on :mod:`fancyimpute`, instead using functions from
  :mod:`scikit-learn`

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~

* **Bugfix**: pyrolite lambdas differ slightly from [ONeill2016]_
  (`#39 <https://github.com/morganjwilliams/pyrolite/issues/39>`__).
  Differences between the lambda coefficients of the original and pyrolite
  implementations of the lambdas calculation were identified (thanks to
  `Laura Miller <https://github.com/Lauraanme>`__ for this one).
  With further investigation, it's likely the cost function passed to
  :func:`scipy.optimize.least_squares` contained an error.
  This has been remedied, and the relevant pyrolite functions now
  by default should give values comparable to [ONeill2016]_. As part of this,
  the reference composition `ChondriteREE_ON` was added to the reference database
  with the REE abundances presented in [ONeill2016]_.
* **Bugfix**: Upgrades for :func:`~pyrolite.geochem.transform.convert_chemistry`
  to improve performance
  (`#29 <https://github.com/morganjwilliams/pyrolite/issues/29>`__).
  This bug appears to have resulted from caching the function calls to
  :func:`pyrolite.geochem.ind.simple_oxides`, which is addressed with
  `18fede0 <https://github.com/morganjwilliams/pyrolite/commit/18fede01d54d06edd3fe1451409880d889e7ee62>`__.
* **Feature**: Added the [WhittakerMuntus1970]_ ionic radii for use in silicate
  geochemistry (
  `#41 <https://github.com/morganjwilliams/pyrolite/issues/41>`__),
  which can optionally be used with :func:`pyrolite.geochem.ind.get_ionic_radii`
  using the `source` keyword argument (:code:`source='Whittaker'`). Thanks to
  `Charles Le Losq <https://github.com/charlesll>`__ for the suggestion!
* **Bugfix**: Removed an erroneous zero from the GLOSS reference composition
  (`GLOSS_P2014` value for Pr).
* Updated :func:`~pyrolite.geochem.ind.REE` to default to :code:`dropPm=True`
* Moved :mod:`pyrolite.mineral.ions` to :mod:`pyrolite.geochem.ions`

.. [ONeill2016] O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth
    Element Patterns in Basalts. J Petrology 57, 1463–1508.
    `doi: 10.1093/petrology/egw047 <https://doi.org/10.1093/petrology/egw047>`__.

.. [WhittakerMuntus1970] Whittaker, E.J.W., Muntus, R., 1970.
    Ionic radii for use in geochemistry.
    Geochimica et Cosmochimica Acta 34, 945–956.
    `doi: 10.1016/0016-7037(70)90077-3 <https://doi.org/10.1016/0016-7037(70)90077-3>`__.

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~

* **Bugfix**: Added the mineral database to `MANIFEST.in` to allow this to be installed
  with :mod:`pyrolite` (fixing a bug where this isn't present, identified by
  `Kaarel Mand <https://github.com/kaarelmand>`__).

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~

* **Bugfix**: Updated :mod:`pyrolite.plot` to use :func:`pandas.DataFrame.reindex` over
  :func:`pandas.DataFrame.loc` where indexes could include missing values to deal with
  `#31 <https://github.com/morganjwilliams/pyrolite/issues/31>`__.
* Updated :func:`~pyrolite.plot.spider.spider` to accept :code:`logy` keyword argument,
  defaulting to :code:`True`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~

* Broke down :mod:`pyrolite.util.plot` into submodules, and updated relevant imports.
  This will result in minimal changes to API usage where functions are
  imported explicitly.
* Split out :mod:`pyrolite.util.lambdas` from :mod:`pyrolite.util.math`
* Added a minimum figure dimension to :func:`~pyrolite.util.plot.axes.init_axes`
  to avoid having null-dimensions during automatic figure generation from empty
  datasets.
* Added :func:`~pyrolite.util.synthetic.example_spider_data` to generate
  an example dataset for demonstrating spider diagrams and associated functions.
  This allowed detailed synthetic data generation for
  :func:`~pyrolite.plot.pyroplot.spider` and :func:`pyrolite.plot.pyroplot.REE`
  plotting examples to be cut down significantly.
* Removed unused submodule :mod:`pyrolite.util.wfs`

`0.2.5`_
--------------

* **PR Merged**: `@lavender22 <https://github.com/lavender22>`__ updated the spider
  diagram example to add a link to the normalisation example (which lists
  different reservoirs you can normalise to).
* Added an 'Importing Data' section to the docs
  `Getting Started page <../gettingstarted.html#importing-data>`__.
* Disabled automatic extension loading (e.g. for :mod:`pyrolite_meltsutil`) to
  avoid bugs during version mismatches.

:mod:`pyrolite.comp`
~~~~~~~~~~~~~~~~~~~~~~~

* Updated the :class:`pyrolite.comp.pyrocomp` dataframe accessor API to include
  reference to compositional data log transform functions within
  :mod:`pyrolite.comp.codata`

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added support for spider plot index ordering added with the keyword
  :code:`index_order` (`#30 <https://github.com/morganjwilliams/pyrolite/issues/30>`__)
* Added support for color indexing in :mod:`~pyrolite.plot.color` using
  :class:`pandas.Series`, and also for list-like arrays of categories
* Added a workaround for referring to axes positions where the projection is changed
  to a ternary projection (displacing the original axis), but the reference to the
  original axes object (now booted from :code:`fig.axes`/:code:`fig.orderedaxes`) is
  subsequently used.
* Updated :func:`~pyrolite.plot.color.process_color` processing of auxillary
  color keyword arguments (fixing a bug for color arguments in
  :func:`~pyrolite.plot.stem`)
* Added support for a :code:`color_mappings` keyword argument for mapping
  categorical variables to specific colors.
* Updated the effect of :code:`relim` keyword argument of
  :func:`~pyrolite.plot.density.density` to remove the scaling (it will no longer
  log-scale the axes, just the grid/histogram bins).
* Updated :class:`~pyrolite.plot.ternary.grid.Grid` to accept an x-y tuple to specify
  numbers of bins in each direction within a grid (e.g. :code:`bins=(20, 40)`)
* Updated the grids used in some of the :func:`~pyrolite.plot.density.density`
  methods to be edges, lining up the arrays such that shading parameters
  will work as expected (e.g. :code:`shading='gouraud'`)

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~
* Added sorting function :code:`~pyrolite.geochem.ind.by_incompatibility`
  for incompatible element sorting (based on BCC/PM relative abundances).

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~~
* Minor bugfix for :func:`~pyrolite.mineral.mindb.update_database`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~
* Moved :func:`~pyrolite.util.general.check_perl` out of :mod:`pyrolite` into
  :mod:`pyrolite_meltsutil`

`0.2.4`_
--------------

* Removed Python 3.5 support, added Python 3.8 support.
* Updated ternary plots to use :mod:`mpltern`
  (`#28 <https://github.com/morganjwilliams/pyrolite/issues/28>`__)
* Added a
  `ternary heatmap tutorial <https://pyrolite.readthedocs.io/en/develop/tutorials/ternary_density.html>`__

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :meth:`pyrolite.plot.pyroplot.plot` method
* Removed :meth:`pyrolite.plot.pyroplot.ternary` method (ternary plots now served
  through the same interface as bivariate plots using
  :meth:`pyrolite.plot.pyroplot.scatter`, :meth:`pyrolite.plot.pyroplot.plot`,
  and :meth:`pyrolite.plot.pyroplot.plot`)
* Added :mod:`pyrolite.plot.color` for processing color arguments.
* Moved :mod:`pyrolite.plot.density` to its own sub-submodule, including
  :mod:`pyrolite.plot.density.ternary` and :mod:`pyrolite.plot.density.grid`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated :mod:`~pyrolite.util.time` to include official colors.
* Added :mod:`pyrolite.util.time`
  `example <https://pyrolite.readthedocs.io/en/develop/examples/util/timescale.html>`__
* Updated :func:`~pyrolite.util.meta.stream_log` to deal with logger
  duplication issues.
* Various updates to :mod:`pyrolite.util.plot`, noted below:
* Added universal axes initiation for bivariate/ternary diagrams using
  :func:`~pyrolite.util.plot.init_axes` and axes labelling with
  :func:`~pyrolite.util.plot.label_axes`,
* Added keyword argument processing functions :func:`~pyrolite.util.plot.scatterkwargs`,
  :func:`~pyrolite.util.plot.linekwargs`, and
  :func:`~pyrolite.util.plot.patchkwargs`
* Added functions for replacing non-projected axes with ternary axes, including
  :func:`~pyrolite.util.plot.replace_with_ternary_axis`,
  :func:`~pyrolite.util.plot.axes_to_ternary` (and
  :func:`~pyrolite.util.plot.get_axes_index` to maintain ordering of new axes)
* Added :func:`~pyrolite.util.plot.get_axis_density_methods` to access the relevant
  histogram/density methods for bivariate and ternary axes
* Renamed private attributes for default colormaps to
  :data:`~pyrolite.util.plot.DEFAULT_DISC_COLORMAP` and
  :data:`~pyrolite.util.plot.DEFAULT_CONT_COLORMAP`
* Updated :func:`~pyrolite.util.plot.add_colorbar` to better handle colorbars
  for ternary diagrams

`0.2.3`_
--------------

* Added `Getting Started page <../gettingstarted.html>`__

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated database for :mod:`pyrolite.mineral.mindb` to include epidotes,
  garnets, micas

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Minor updates for :mod:`pyrolite.plot.templates`, added functionality to
  :func:`pyrolite.plot.templates.TAS` stub.
* Fixed a bug for :code:`vmin` in :mod:`pyrolite.plot.spider` density modes

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~

* :mod:`pyrolite.geochem.parse` now also includes functions which were previously
  included in :mod:`pyrolite.geochem.validate`
* Fixed some typos in reference compositions from Gale et al. (2013)

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.util.plot.set_ternary_labels` for setting and positioning
  ternary plot labels

`0.2.2`_
--------------

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`~pyrolite.geochem.magma.SCSS` for modelling sulfur content at
  sulfate/sulfide saturation.

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added `mineral database <../examples/geochem/mineral_mindb.html>`__ and
  and `mineral endmember decomposition <../examples/geochem/mineral_endmembers.html>`__
  examples


`0.2.1`_
--------------

* Updated and refactored documentation

  * Added `Development <development.html>`__, Debugging section,
    `Extensions <../ext/extensions.html>`__
  * Added :mod:`sphinx_gallery` with binder links for examples
  * Removed duplicated examples
  * Amended `citation guidelines <../cite.html>`__

* Removed extensions from pyrolite (:code:`pyrolite.ext.datarepo`,
  :code:`pyrolite.ext.alphamelts`). These will soon be available as separate extension
  packages. This enabled faster build and test times, and removed extraneous dependencies
  for the core :mod:`pyrolite` package.
* Added :code:`stats_require` as optional requirements in :code:`setup.py`

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`~pyrolite.geochem.transform.get_ratio` and
  :meth:`pyrolite.geochem.pyrochem.get_ratio`
* Added :meth:`pyrolite.geochem.pyrochem.compositional` selector

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`~pyrolite.plot.parallel.parallel` now better handles :mod:`~matplotlib.pyplot`
  figure and subplot arguments
* :func:`~pyrolite.plot.tern.ternary` and related functions now handle label offsets
  and label fontsizes
* Minor bugfixes for :mod:`~pyrolite.plot.density`
* Added :code:`unity_line` argument to :func:`~pyrolite.plot.spider.spider`
  to be consistent with :func:`~pyrolite.plot.spider.REE_v_radii`

:mod:`pyrolite.mineral`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added a simple :mod:`pyrolite.mineral.mindb` database
* Added :mod:`pyrolite.mineral.transform` to house mineral transformation functions
* Expanded :mod:`pyrolite.mineral.normative` to include
  :func:`~pyrolite.mineral.normative.unmix` and
  :func:`pyrolite.mineral.normative.endmember_decompose` for composition-based
  mineral endmember decomposition

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.util.plot.mappable_from_values` to enable generating
  :class:`~matplotlib.cm.ScalarMappable` objects from an array of values, for use
  in generating colorbars

`0.2.0`_
--------------

* Added alt-text to documentation example images
* Updated contributing guidelines
* Added Python 3.8-dev to Travis config (not yet available)
* Removed :mod:`pandas-flavor` decorators from :mod:`pyrolite.geochem` and
  :mod:`pyrolite.comp`, eliminating the dependency on :mod:`pandas-flavor`

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Expanded :class:`pyrolite.geochem.pyrochem` DataFrame accessor and constituent
  methods
* Updates and bugfixes for :mod:`pyrolite.geochem.transform` and
  :mod:`pyrolite.geochem.norm`
* Updated the `normalization example <../examples/geochem/normalization.html>`__

:mod:`pyrolite.comp`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :class:`pyrolite.comp.pyrocomp` DataFrame accessor with the
  :func:`pyrolite.comp.codata.renormalise` method.
* Removed unused imputation and aggregation functions.

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :meth:`~pyrolite.plot.pyroplot.heatscatter` and `example <../examples/plotting/heatscatter.html>`__.
* Updates and bugfixes for :func:`pyrolite.plot.spider.REE_v_radii`, including updating
  spacing to reflect relative ionic radii

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.util.plot.get_twins`


`0.1.21`_
--------------

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~

* Added parallel coordinate plots: :meth:`pyrolite.plot.pyroplot.parallel`
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

* Moved normalization into :mod:`pyrolite.geochem`
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
* Added :meth:`pyrolite.plot.pyrochem.cooccurence`
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated :class:`pyrolite.ext.alphamelts.automation.MeltsProcess` workflow
* Updated :class:`pyrolite.ext.alphamelts.download` local installation
* Added :mod:`pyrolite.ext.alphamelts.install` example
* Added :mod:`pyrolite.ext.alphamelts.tables` example
* Added :mod:`pyrolite.ext.alphamelts.automation` example
* Added :mod:`pyrolite.ext.alphamelts.env` example

`0.1.12`_
--------------

:mod:`pyrolite.util.pd`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Bugfix for :func:`pyrolite.util.pd.to_frame`

`0.1.11`_
--------------

* Added `citation <cite.html>`__ page to docs
* Added `contributors <contributors.html>`__ page to docs
* Updated docs `future <future.html>`__ page
* Updated docs config and logo

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added stub for :mod:`pyrolite.geochem.isotope`, :mod:`pyrolite.geochem.isotope.count`

:mod:`pyrolite.comp`
~~~~~~~~~~~~~~~~~~~~~~~

* Added compositional data example
* Added :func:`pyrolite.comp.codata.logratiomean`
* Added :mod:`pyrolite.data.Aitchison` and assocaited data files

:mod:`pyroilite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.ext.alphamelts` to API docs
* Added :mod:`pyrolite.ext.alphamelts.automation`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Expanded :mod:`pyrolite.util` API docs
* Added :mod:`pyrolite.util.distributions`
* Moved `pyrolite_datafolder` from :mod:`pyrolite.util.general` to
  :func:`pyrolite.util.meta.pyrolite_datafolder`
* Added :func:`~pyrolite.util.plot.share_axes`,
  :func:`~pyrolite.util.plot.ternary_patch`,
  :func:`~pyrolite.util.plot.subaxes`
* Added :mod:`pyrolite.util.units`, moved
  `pyrolite.geochem.norm.scale_multiplier` to :func:`pyrolite.util.units.scale`
* Updated :func:`pyrolite.util.synthetic.random_cov_matrix` to optionally take a
  :code:`sigmas` keyword argument

`0.1.10`_
--------------

* Updated `installation <installation.html>`__ docs

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.util.types`
* Added :mod:`pyrolite.util.web`
* Added manifold uncertainty example with :func:`pyrolite.util.skl.vis.plot_mapping`
* Moved `stream log` to :func:`pyrolite.util.meta.stream_log`
* Added :func:`pyrolite.util.meta.take_me_to_the_docs()`
* Updated :mod:`pyrolite.util.skl.vis`

:mod:`pyrolite.ext.datarepo`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Updated :mod:`pyrolite.ext.datarepo.georoc` (then `pyrolite.util.repositories.georoc`)

`0.1.9`_
--------------

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.plot.templates`, and related API docs
* Added Pearce templates under :mod:`pyrolite.plot.templates.pearce`
* Update default color schemes in scatter plots within :mod:`pyrolite.plot` to
  fall-back to :mod:`matplotlib.pyplot` cycling

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~~

* Added conditional import for :class:`~sklearn.decomposition.PCA` and :mod:`statsmodels`
  within :mod:`pyrolite.util.plot`
* Refactored :mod:`sklearn` utilities to submodule :mod:`pyrolite.util.skl`
* Added :func:`pyrolite.util.meta.sphinx_doi_link`
* Updated :func:`pyrolite.util.meta.inargs`
* Updated :func:`pyrolite.util.meta.stream_log` (then `pyrolite.util.general.stream_log`)
* Added conditional import for :mod:`imblearn` under :mod:`pyrolite.util.skl.pipeline`

:mod:`pyrolite.ext.alphamelts`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.ext.alphamelts` (then `pyrolite.util.alphamelts`)
* Bugfix for Python 3.5 style strings in :mod:`pyrolite.ext.alphamelts.parse`

`0.1.8`_
--------------

* Bugfixes for :mod:`pyrolite.plot.spider` and :mod:`pyrolite.util.plot.conditional_prob_density`

`0.1.7`_
--------------

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`~pyrolite.plot.pyroplot.cooccurence` method to :class:`pyrolite.plot.pyroplot`
  DataFrame accessor

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~

* Added :func:`pyrolite.util.missing.cooccurence_pattern`
* Moved `pyrolite.util.skl.plot_cooccurence` to :func:`pyrolite.util.plot.plot_cooccurence`
* Updated :func:`pyrolite.util.plot.conditional_prob_density`,
  :func:`pyrolite.util.plot.bin_edges_to_centres` and
  :func:`pyrolite.util.plot.bin_centres_to_edges`

`0.1.6`_
--------------

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~
* Update :func:`~pyrolite.plot.spider.spider` to use :code:`contours` keyword argument,
  and pass these to :func:`pyrolite.util.plot.plot_Z_percentiles`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~

* Bugfixes for invalid steps in :func:`pyrolite.util.math.linspc_`,
  :func:`pyrolite.util.math.logspc_`

`0.1.5`_
--------------

* Updated docs `future <future.html>`__ page

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~

* Bugfix for iron redox recalcuation in
  :func:`pyrolite.geochem.transform.convert_chemistry`

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~

* Added :code:`mode` keyword argument to :func:`pyrolite.plot.spider.spider`
  to enable density-based visualisation of spider plots.
* Update :func:`pyrolite.plot.pyroplot.spider` to accept :code:`mode` keyword argument
* Update :func:`pyrolite.plot.pyroplot.REE` to use a :code:`index` keyword arguument
  in the place of the previous :code:`mode`; :code:`mode` is now used to switch between
  line and density base methods of visualising spider plots consistent with
  :func:`~pyrolite.plot.spider.spider`
* Added :func:`~pyrolite.plot.spider.spider`
  `examples for conditional density plots <../examples/plotting/conditionaldensity.html>`__
  using :func:`~pyrolite.util.plot.conditional_prob_density`
* Bugfix for :code:`set_under` in :func:`~pyrolite.plot.density.density`
* Updated `logo example <../tutorials/logo.html>`__

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~

* Updated :mod:`pyrolite.util.meta`
* Added :func:`pyrolite.util.plot.conditional_prob_density`;
  added conditional :mod:`statsmodels` import within :mod:`pyrolite.util.plot`
  to access :class:`~statsmodels.nonparametric.kernel_density.KDEMultivariateConditional`
* Added keyword argument :code:`logy` to :func:`pyrolite.util.math.interpolate_line`
* Added :func:`pyrolite.util.math.grid_from_ranges` and
  :func:`pyrolite.util.math.flattengrid`
* Added support for differential x-y padding in :func:`pyrolite.util.plot.get_full_extent`
  and :func:`pyrolite.util.plot.save_axes`
* Added :func:`pyrolite.util.skl.pipeline.fit_save_classifier`
  (then `pyrolite.util.skl.fit_save_classifier`)

`0.1.4`_
--------------

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~

* Updated relevant docs and references for :mod:`pyrolite.plot` and the
  :class:`pyrolite.plot.pyroplot` DataFrame accessor

:mod:`pyrolite.comp`
~~~~~~~~~~~~~~~~~~~~~~

* Expanded :mod:`pyrolite.comp.impute` and improved :func:`pyrolite.comp.impute.EMCOMP`
* Added `EMCOMP example <../examples/comp/EMCOMP.html>`__

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~

* Updated :mod:`pyrolite.util.meta` with docstring utilities
  :func:`~pyrolite.util.meta.numpydoc_str_param_list` and
  :func:`~pyrolite.util.meta.get_additional_params`

`0.1.2`_
--------------

* Fixed logo naming issue in docs

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~

* Bugfixes for :func:`pyrolite.plot.density.density` (then `pyrolite.plot.density`)
  and :func:`pyrolite.plot.util.ternary_heatmap`

`0.1.1`_
--------------


:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~

* Added `logo example <../tutorials/logo.html>`__
* Refactored :mod:`pyrolite.plot` to use the :class:`pyrolite.plot.pyroplot` DataFrame
  accessor:

  * Renamed `pyrolite.plot.spiderplot` to
    :func:`pyrolite.plot.spider.spider`
  * Renamed `pyrolite.plot.spider.REE_radii_plot` to
    :func:`pyrolite.plot.spider.REE_v_radii`
  * Renamed `pyrolite.plot.ternaryplot` to
    :func:`pyrolite.plot.tern.ternary`
  * Renamed `pyrolite.plot.densityplot` to
    :func:`pyrolite.plot.density.density`

* Updated :func:`pyrolite.plot.density.density` and :func:`pyrolite.plot.tern.ternary`

:mod:`pyrolite.comp`
~~~~~~~~~~~~~~~~~~~~~~

* Bugfixes and improvements for :mod:`pyrolite.comp.impute`

:mod:`pyrolite.geochem`
~~~~~~~~~~~~~~~~~~~~~~~~

* Updated :func:`~pyrolite.geochem.transform.oxide_conversion` and
  :func:`~pyrolite.geochem.transform.convert_chemistry`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~~~~

* Added :func:`~pyrolite.util.plot.plot_stdev_ellipses` and
  :func:`~pyrolite.util.plot.plot_pca_vectors`
* Updated :func:`pyrolite.util.plot.plot_Z_percentiles`
* Updated :func:`pyrolite.util.plot.ternary_heatmap`
* Updated :func:`pyrolite.util.plot.vector_to_line`

`0.1.0`_
--------------

:mod:`pyrolite.plot`
~~~~~~~~~~~~~~~~~~~~~~~

* Updates to :func:`pyrolite.plot.density.density` to better deal with linear/log
  spaced and a ternary heatmap

:mod:`pyrolite.comp`
~~~~~~~~~~~~~~~~~~~~

* Added :func:`~pyrolite.comp.impute.EMCOMP` to :mod:`pyrolite.comp.impute`
* Renamed `inv_alr`, `inv_clr`, `inv_ilr` and `inv_boxcox` to
  :func:`~pyrolite.comp.codata.inverse_alr`,
  :func:`~pyrolite.comp.codata.inverse_clr`,
  :func:`~pyrolite.comp.codata.inverse_ilr` and
  :func:`~pyrolite.comp.codata.inverse_boxcox`

:mod:`pyrolite.util`
~~~~~~~~~~~~~~~~~~~~~

* Added :mod:`pyrolite.util.synthetic`
* Moved `pyrolite.util.pd.test_df` and `pyrolite.util.pd.test_ser`
  to :func:`pyrolite.util.synthetic.test_df` and
  :func:`pyrolite.util.synthetic.test_ser`
* Added :mod:`pyrolite.util.missing` and :func:`pyrolite.util.missing.md_pattern`
* Added :func:`pyrolite.util.math.eigsorted`,
  :func:`pyrolite.util.math.augmented_covariance_matrix`,
  :func:`pyrolite.util.math.interpolate_line`


.. note:: Releases before 0.1.0 are available via
    `GitHub <https://github.com/morganjwilliams/pyrolite/releases>`__ for reference,
    but were :code:`alpha` versions which were never considered stable.

.. _Development: https://github.com/morganjwilliams/pyrolite/compare/0.2.6...develop
.. _0.2.6: https://github.com/morganjwilliams/pyrolite/compare/0.2.5...0.2.6
.. _0.2.5: https://github.com/morganjwilliams/pyrolite/compare/0.2.4...0.2.5
.. _0.2.4: https://github.com/morganjwilliams/pyrolite/compare/0.2.3...0.2.4
.. _0.2.3: https://github.com/morganjwilliams/pyrolite/compare/0.2.2...0.2.3
.. _0.2.2: https://github.com/morganjwilliams/pyrolite/compare/0.2.1...0.2.2
.. _0.2.1: https://github.com/morganjwilliams/pyrolite/compare/0.2.0...0.2.1
.. _0.2.0: https://github.com/morganjwilliams/pyrolite/compare/0.1.21...0.2.0
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
