.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_geochem_normalization.py:


Normalization
==============

A selection of reference compositions are included in pyrolite, and can be easily
accessed with :func:`pyrolite.geochem.norm.get_reference_composition` (see the list
at the bottom of the page for a complete list):


.. code-block:: default

    import pandas as pd
    import matplotlib.pyplot as plt
    import pyrolite.plot
    from pyrolite.geochem.ind import REE
    from pyrolite.geochem.norm import get_reference_composition, all_reference_compositions









.. code-block:: default

    chondrite = get_reference_composition("Chondrite_PON")







To use the compositions with a specific set of units, you can change them with
:func:`~pyrolite.geochem.norm.Composition.set_units`:



.. code-block:: default

    CI = chondrite.set_units("ppm")







The :func:`~pyrolite.geochem.pyrochem.normalize_to` method can be used to
normalise DataFrames to a given reference (e.g. for spiderplots):



.. code-block:: default

    fig, ax = plt.subplots(1)

    for name, ref in list(all_reference_compositions().items())[::2]:
        if name != "Chondrite_PON":
            ref.set_units("ppm")
            df = ref.comp.pyrochem.REE.pyrochem.normalize_to(CI, units="ppm")
            df.pyroplot.REE(unity_line=True, ax=ax, label=name)

    ax.set_ylabel("X/X$_{Chondrite}$")
    ax.legend(frameon=False, facecolor=None, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.show()



.. image:: /examples/geochem/images/sphx_glr_normalization_001.png
    :class: sphx-glr-single-img





.. seealso::

  Examples:
    `lambdas: Parameterising REE Profiles <lambdas.html>`__,
    `REE Radii Plot <../plotting/REE_radii_plot.html>`__

Currently available models include:

|refcomps|


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.141 seconds)


.. _sphx_glr_download_examples_geochem_normalization.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/geochem/normalization.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: normalization.py <normalization.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: normalization.ipynb <normalization.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
