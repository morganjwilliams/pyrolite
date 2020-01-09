.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_geochem_convert_chemistry.py:


Element-Oxide Transformation
============================

One of pyrolite's strengths is converting mixed elemental and oxide data to a new
form. The simplest way to perform this is by using the
:func:`~pyrolite.geochem.transform.convert_chemistry` function. Note that by default
pyrolite assumes that data are in the same units.


.. code-block:: default

    import pyrolite.geochem
    import pandas as pd

    pd.set_option("precision", 3)  # smaller outputs







Here we create some synthetic data to work with, which has some variables in Wt% and
some in ppm. Notably some elements are present in more than one column (Ca, Na):



.. code-block:: default

    from pyrolite.util.synthetic import test_df

    df = test_df(cols=["MgO", "SiO2", "FeO", "CaO", "Na2O", "Te", "K", "Na"]) * 100
    df.pyrochem.elements *= 100 # elements in ppm








.. code-block:: default

    df.head(2)





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
              <th>CaO</th>
              <th>Na2O</th>
              <th>Te</th>
              <th>K</th>
              <th>Na</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>18.445</td>
              <td>3.872</td>
              <td>8.731</td>
              <td>2.101</td>
              <td>7.025</td>
              <td>420.179</td>
              <td>356.628</td>
              <td>5205.854</td>
            </tr>
            <tr>
              <th>1</th>
              <td>19.749</td>
              <td>3.756</td>
              <td>8.903</td>
              <td>2.329</td>
              <td>7.570</td>
              <td>418.555</td>
              <td>352.688</td>
              <td>4998.054</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

As the units are heterogeneous, we'll need to convert the data frame to a single set of
units (here we use Wt%):



.. code-block:: default

    df.pyrochem.elements = df.pyrochem.elements.pyrochem.scale('ppm', 'wt%') # ppm to wt%







We can transform this chemical data to a new set of compositional variables.
Here we i) convert CaO to Ca, ii) aggregate Na2O and Na to Na and iii) calculate
mass ratios for Na/Te and MgO/SiO2.
Note that you can also use this function to calculate mass ratios:



.. code-block:: default

    df.pyrochem.convert_chemistry(
        to=["MgO", "SiO2", "FeO", "Ca", "Te", "Na", "Na/Te", "MgO/SiO2"]
    ).head(2)





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
              <th>Ca</th>
              <th>Te</th>
              <th>Na</th>
              <th>Na/Te</th>
              <th>MgO/SiO2</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>18.445</td>
              <td>3.872</td>
              <td>8.731</td>
              <td>1.501</td>
              <td>0.042</td>
              <td>5.732</td>
              <td>136.415</td>
              <td>4.763</td>
            </tr>
            <tr>
              <th>1</th>
              <td>19.749</td>
              <td>3.756</td>
              <td>8.903</td>
              <td>1.665</td>
              <td>0.042</td>
              <td>6.116</td>
              <td>146.115</td>
              <td>5.258</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

You can also specify molar ratios for iron redox, which will result in multiple iron
species within the single dataframe:



.. code-block:: default

    df.pyrochem.convert_chemistry(to=[{"FeO": 0.9, "Fe2O3": 0.1}]).head(2)





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
              <th>FeO</th>
              <th>Fe2O3</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>7.858</td>
              <td>0.970</td>
            </tr>
            <tr>
              <th>1</th>
              <td>8.013</td>
              <td>0.989</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.393 seconds)


.. _sphx_glr_download_examples_geochem_convert_chemistry.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/geochem/convert_chemistry.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: convert_chemistry.py <convert_chemistry.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: convert_chemistry.ipynb <convert_chemistry.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
