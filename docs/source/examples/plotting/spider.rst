.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_plotting_spider.py:


Spiderplots & Density Spiderplots
==================================


.. code-block:: default

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt









Here we'll set up an example which uses EMORB as a starting point:



.. code-block:: default

    from pyrolite.geochem.norm import get_reference_composition

    ref = get_reference_composition("EMORB_SM89")  # EMORB composition as a starting point
    ref.set_units("ppm")
    df = ref.comp.pyrochem.compositional







Basic spider plots are straightforward to produce:


.. code-block:: default

    import pyrolite.plot

    df.pyroplot.spider(color="k")
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0.0, 0.0, 0.0, 1)




Typically we'll normalise trace element compositions to a reference composition
to be able to link the diagram to 'relative enrichement' occuring during geological
processes:



.. code-block:: default

    normdf = df.pyrochem.normalize_to("PM_PON", units="ppm")
    normdf.pyroplot.spider(color="k", unity_line=True)
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0.0, 0.0, 0.0, 1)




The spiderplot can be extended to provide visualisations of ranges and density via the
various modes. First let's take this composition and add some noise in log-space to
generate multiple compositions about this mean (i.e. a compositional distribution):



.. code-block:: default

    start = normdf.applymap(np.log)
    nindex, nobs = normdf.columns.size, 120

    noise_level = 0.5  # sigma for noise
    x = np.arange(nindex)
    y = np.tile(start.values, nobs).reshape(nobs, nindex)
    y += np.random.normal(0, noise_level / 2.0, size=(nobs, nindex))  # noise
    y += np.random.normal(0, noise_level, size=(1, nobs)).T  # random pattern offset

    distdf = pd.DataFrame(y, columns=normdf.columns)
    distdf["Eu"] += 1.0  # significant offset for Eu anomaly
    distdf = distdf.applymap(np.exp)







We could now plot the range of compositions as a filled range:



.. code-block:: default

    distdf.pyroplot.spider(mode="fill", color="green", alpha=0.5, unity_line=True)
    plt.show()



.. image:: /examples/plotting/images/sphx_glr_spider_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0.0, 0.5019607843137255, 0.0, 0.5)




Alternatively, we can plot a conditional density spider plot:



.. code-block:: default

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))
    _ = distdf.pyroplot.spider(ax=ax[0], color="k", alpha=0.05, unity_line=True)
    _ = distdf.pyroplot.spider(
        ax=ax[1],
        mode="binkde",
        cmap="viridis",
        vmin=0.05,  # minimum percentile,
        resolution=10,
        unity_line=True
    )



.. image:: /examples/plotting/images/sphx_glr_spider_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0.0, 0.0, 0.0, 0.95)




We can now assemble a more complete comparison of some of the conditional density
modes for spider plots:



.. code-block:: default

    modes = [
        ("plot", "plot", [], dict(color="k", alpha=0.01)),
        ("fill", "fill", [], dict(color="k", alpha=0.5)),
        ("binkde", "binkde", [], dict(resolution=10)),
        (
            "binkde",
            "binkde contours specified",
            [],
            dict(contours=[0.95], resolution=10),  # 95th percentile contour
        ),
        ("histogram", "histogram", [], dict(resolution=5, ybins=30)),
    ]








.. code-block:: default

    down, across = len(modes), 1
    fig, ax = plt.subplots(
        down, across, sharey=True, sharex=True, figsize=(across * 8, 2 * down)
    )

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
        distdf.pyroplot.spider(
            mode=m,
            ax=ax[mix],
            cmap="viridis",
            vmin=0.05,  # minimum percentile
            fontsize=8,
            unity_line=True,
            *args,
            **kwargs
        )

    plt.tight_layout()



.. image:: /examples/plotting/images/sphx_glr_spider_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (0.0, 0.0, 0.0, 0.99)
    (0.0, 0.0, 0.0, 0.5)




.. seealso:: `Heatscatter Plots <heatscatter.html>`__,
             `Density Diagrams <density.html>`__


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  19.015 seconds)


.. _sphx_glr_download_examples_plotting_spider.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/plotting/spider.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: spider.py <spider.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: spider.ipynb <spider.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
