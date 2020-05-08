.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_plotting_spider.py:


Spiderplots & Density Spiderplots
==================================


.. code-block:: default

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt









Here we'll set up an example which uses EMORB as a starting point. Typically we'll
normalise trace element compositions to a reference composition
to be able to link the diagram to 'relative enrichement' occuring during geological
processes, so here we're normalising to a Primitive Mantle composition first.
We're here taking this normalised composition and adding some noise in log-space to
generate multiple compositions about this mean (i.e. a compositional distribution).
For simplicility, this is handlded by
:func:`~pyrolite.util.synthetic.example_spider_data`:



.. code-block:: default

    from pyrolite.util.synthetic import example_spider_data

    normdf = example_spider_data(start="EMORB_SM89", norm_to="PM_PON")








.. seealso:: `Normalisation <../geochem/normalization.html>`__


Basic spider plots are straightforward to produce:



.. code-block:: default

    import pyrolite.plot

    ax = normdf.pyroplot.spider(color="0.5", alpha=0.5, unity_line=True, figsize=(10, 4))
    ax.set_ylabel("X / $X_{Primitive Mantle}$")
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_001.png
    :class: sphx-glr-single-img





The default ordering here follows that of the dataframe columns, but we typically
want to reorder these based on some physical ordering. A :code:`index_order` keyword
argument can be used to supply a function which will reorder the elements before
plotting. Here we order the elements by relative incompatiblity using
:func:`pyrolite.geochem.ind.order_incompatibility`:


.. code-block:: default

    from pyrolite.geochem.ind import by_incompatibility

    ax = normdf.pyroplot.spider(
        color="k",
        alpha=0.1,
        unity_line=True,
        index_order=by_incompatibility,
        figsize=(10, 4),
    )
    ax.set_ylabel("X / $X_{Primitive Mantle}$")
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_002.png
    :class: sphx-glr-single-img





The spiderplot can be extended to provide visualisations of ranges and density via the
various modes. We could now plot the range of compositions as a filled range:



.. code-block:: default

    ax = normdf.pyroplot.spider(
        mode="fill",
        color="green",
        alpha=0.5,
        unity_line=True,
        index_order=by_incompatibility,
        figsize=(10, 4),
    )
    ax.set_ylabel("X / $X_{Primitive Mantle}$")
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_003.png
    :class: sphx-glr-single-img





Alternatively, we can plot a conditional density spider plot:



.. code-block:: default

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))
    normdf.pyroplot.spider(
        ax=ax[0], color="k", alpha=0.05, unity_line=True, index_order=by_incompatibility
    )
    normdf.pyroplot.spider(
        ax=ax[1],
        mode="binkde",
        vmin=0.05,  # 95th percentile,
        resolution=10,
        unity_line=True,
        index_order=by_incompatibility,
    )
    [a.set_ylabel("X / $X_{Primitive Mantle}$") for a in ax]
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_004.png
    :class: sphx-glr-single-img





We can now assemble a more complete comparison of some of the conditional density
modes for spider plots:



.. code-block:: default

    modes = [
        ("plot", "plot", [], dict(color="k", alpha=0.01)),
        ("fill", "fill", [], dict(color="k", alpha=0.5)),
        ("binkde", "binkde", [], dict(resolution=5)),
        (
            "binkde",
            "binkde contours specified",
            [],
            dict(contours=[0.95], resolution=5),  # 95th percentile contour
        ),
        ("histogram", "histogram", [], dict(resolution=5, ybins=30)),
    ]








.. code-block:: default

    down, across = len(modes), 1
    fig, ax = plt.subplots(
        down, across, sharey=True, sharex=True, figsize=(across * 8, 2 * down)
    )
    [a.set_ylabel("X / $X_{Primitive Mantle}$") for a in ax]
    for a, (m, name, args, kwargs) in zip(ax, modes):
        a.annotate(  # label the axes rows
            "Mode: {}".format(name),
            xy=(0.1, 1.05),
            xycoords=a.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax = ax.flat
    for mix, (m, name, args, kwargs) in enumerate(modes):
        normdf.pyroplot.spider(
            mode=m,
            ax=ax[mix],
            vmin=0.05,  # minimum percentile
            fontsize=8,
            unity_line=True,
            index_order=by_incompatibility,
            *args,
            **kwargs
        )

    plt.tight_layout()



.. image:: /examples/plotting/images/sphx_glr_spider_005.png
    :class: sphx-glr-single-img





.. seealso:: `Heatscatter Plots <heatscatter.html>`__,
             `Density Diagrams <density.html>`__


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  7.640 seconds)


.. _sphx_glr_download_examples_plotting_spider.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/plotting/spider.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: spider.py <spider.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: spider.ipynb <spider.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
