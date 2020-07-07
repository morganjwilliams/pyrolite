.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_geochem_ionic_radii.py:


Ionic Radii
=============

:mod:`pyrolite` incldues a few sets of reference tables for ionic radii in aangstroms
(Å) from [Shannon1976]_ and [WhittakerMuntus1970]_, each with tables indexed
by element, ionic charge and coordination. The easiset way to access these is via
the :func:`~pyrolite.geochem.ind.get_ionic_radii` function. The function can be used
to get radii for individual elements:


.. code-block:: default

    from pyrolite.geochem.ind import get_ionic_radii, REE

    Cu_radii = get_ionic_radii("Cu")
    print(Cu_radii)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    index
    Cu2+IV      0.57
    Cu2+IVSQ    0.57
    Cu2+V       0.65
    Cu2+VI      0.73
    Name: ionicradius, dtype: float64




Note that this function returned a series of the possible radii, given specific
charges and coordinations of the Cu ion. If we completely specify these, we'll get
a single number back:



.. code-block:: default

    Cu2plus6fold_radii = get_ionic_radii("Cu", coordination=6, charge=2)
    print(Cu2plus6fold_radii)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.73




You can also pass lists to the function. For example, if you wanted to get the Shannon
ionic radii of Rare Earth Elements (REE) in eight-fold coordination with a valence of
+3, you should use the following:



.. code-block:: default

    shannon_ionic_radii = get_ionic_radii(REE(), coordination=8, charge=3)
    print(shannon_ionic_radii)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1.16, 1.143, 1.126, 1.109, 1.079, 1.0659999999999998, 1.053, 1.04, 1.0270000000000001, 1.015, 1.004, 0.9940000000000001, 0.985, 0.977]




The function defaults to using the Shannon ionic radii consistent with [Pauling1960]_,
but you can adjust to use the set you like with the `pauling` boolean argument
(:code:`pauling=False` to use Shannon's 'Crystal Radii') or the `source` argument
(:code:`source='Whittaker'` to use the [WhittakerMuntus1970]_ dataset):



.. code-block:: default

    shannon_crystal_radii = get_ionic_radii(REE(), coordination=8, charge=3, pauling=False)
    whittaker_ionic_radii = get_ionic_radii(
        REE(), coordination=8, charge=3, source="Whittaker"
    )







We can see what the differences between these look like across the REE:



.. code-block:: default

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)

    ax.plot(shannon_ionic_radii, marker="D", label="Shannon Ionic Radii")
    ax.plot(shannon_crystal_radii, marker="D", label="Shannon Crystal Radii")
    ax.plot(whittaker_ionic_radii, marker="D", label="Whittaker & Muntus\nIonic Radii")
    {a: b for (a, b) in zip(REE(), whittaker_ionic_radii)}
    ax.set_xticks(range(len(REE())))
    ax.set_xticklabels(REE())
    ax.set_ylabel("Ionic Radius ($\AA$)")
    ax.set_title("Rare Earth Element Ionic Radii")
    ax.legend(facecolor=None, frameon=False, bbox_to_anchor=(1, 1))




.. image:: /examples/geochem/images/sphx_glr_ionic_radii_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.legend.Legend object at 0x000001670E2EA9C8>



.. seealso::

  Examples:
   `lambdas: Parameterising REE Profiles <lambdas.html>`__,
   `REE Radii Plot <../plotting/REE_radii_plot.html>`__

  Functions:
    :func:`~pyrolite.geochem.ind.get_ionic_radii`,
    :func:`pyrolite.geochem.ind.REE`,
    :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`,


References
----------
.. [Shannon1976] Shannon RD (1976). Revised effective ionic radii and systematic
        studies of interatomic distances in halides and chalcogenides.
        Acta Crystallographica Section A 32:751–767.
        `doi: 10.1107/S0567739476001551 <https://doi.org/10.1107/S0567739476001551>`__.
.. [WhittakerMuntus1970] Whittaker, E.J.W., Muntus, R., 1970.
       Ionic radii for use in geochemistry.
       Geochimica et Cosmochimica Acta 34, 945–956.
       `doi: 10.1016/0016-7037(70)90077-3 <https://doi.org/10.1016/0016-7037(70)90077-3>`__.
.. [Pauling1960] Pauling, L., 1960. The Nature of the Chemical Bond.
        Cornell University Press, Ithaca, NY.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.269 seconds)


.. _sphx_glr_download_examples_geochem_ionic_radii.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/geochem/ionic_radii.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: ionic_radii.py <ionic_radii.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: ionic_radii.ipynb <ionic_radii.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
