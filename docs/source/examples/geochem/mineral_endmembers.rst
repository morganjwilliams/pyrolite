.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_geochem_mineral_endmembers.py:


Mineral Endmember Decomposition
=================================

A common task when working with mineral chemistry data is to take measured compositions
and decompose these into relative proportions of mineral endmember compositions.
pyrolite includes some utilities to achieve this and a limited mineral database
for looking up endmember compositions. This part of the package is being actively
developed, so expect expansions and improvements soon.


.. code-block:: default

    import pandas as pd
    import numpy as np
    from pyrolite.mineral.mindb import get_mineral
    from pyrolite.mineral.normative import endmember_decompose








First we'll start with a composition of an unknown olivine:



.. code-block:: default

    comp = pd.Series({"MgO": 42.06, "SiO2": 39.19, "FeO": 18.75})







We can break this down into olivine endmebmers using the
:func:`~pyrolite.mineral.transform.endmember_decompose` function:



.. code-block:: default

    ed = endmember_decompose(
        pd.DataFrame(comp).T, endmembers="olivine", ord=1, molecular=True
    )
    ed





.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th>name</th>
              <th>forsterite</th>
              <th>fayalite</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>79.994</td>
              <td>20.006</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Equally, if you knew the likely endmembers beforehand, you could specify a list of
endmembers:



.. code-block:: default

    ed = endmember_decompose(
        pd.DataFrame(comp).T, endmembers=["forsterite", "fayalite"], ord=1, molecular=True
    )
    ed





.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>forsterite</th>
              <th>fayalite</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>79.994</td>
              <td>20.006</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

We can check this by recombining the components with these proportions. We can first
lookup the compositions for our endmembers:



.. code-block:: default

    em = pd.DataFrame([get_mineral("forsterite"), get_mineral("fayalite")])
    em.loc[:, ~(em == 0).all(axis=0)]  # columns not full of zeros





.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>name</th>
              <th>group</th>
              <th>formula</th>
              <th>Mg</th>
              <th>Si</th>
              <th>O</th>
              <th>Fe</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>forsterite</td>
              <td>olivine</td>
              <td>Mg2SiO4</td>
              <td>0.346</td>
              <td>0.200</td>
              <td>0.455</td>
              <td>0.000</td>
            </tr>
            <tr>
              <td>1</td>
              <td>fayalite</td>
              <td>olivine</td>
              <td>Fe2SiO4</td>
              <td>0.000</td>
              <td>0.138</td>
              <td>0.314</td>
              <td>0.548</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

First we have to convert these element-based compositions to oxide-based compositions:



.. code-block:: default

    emvalues = (
        em.loc[:, ["Mg", "Si", "Fe"]]
        .pyrochem.to_molecular()
        .fillna(0)
        .pyrochem.convert_chemistry(to=["MgO", "SiO2", "FeO"], molecular=True)
        .fillna(0)
        .pyrocomp.renormalise(scale=1)
    )
    emvalues





.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>MgO</th>
              <th>SiO2</th>
              <th>FeO</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>0.667</td>
              <td>0.333</td>
              <td>0.000</td>
            </tr>
            <tr>
              <td>1</td>
              <td>0.000</td>
              <td>0.333</td>
              <td>0.667</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

These can now be used with our endmember proportions to regenerate a composition:



.. code-block:: default

    recombined = pd.DataFrame(ed.values.flatten() @ emvalues).T.pyrochem.to_weight()
    recombined





.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>MgO</th>
              <th>SiO2</th>
              <th>FeO</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>42.059</td>
              <td>39.191</td>
              <td>18.75</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

To make sure these compositions are within 0.01 percent:



.. code-block:: default

    assert np.allclose(recombined.values, comp.values, rtol=10 ** -4)








.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.294 seconds)


.. _sphx_glr_download_examples_geochem_mineral_endmembers.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/geochem/mineral_endmembers.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: mineral_endmembers.py <mineral_endmembers.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: mineral_endmembers.ipynb <mineral_endmembers.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
