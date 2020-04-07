.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_geochem_indexes_selectors.py:


Geochemical Indexes and Selectors
==================================


.. code-block:: default

    import pyrolite.geochem
    import pandas as pd

    pd.set_option("precision", 3)  # smaller outputs








.. code-block:: default

    from pyrolite.util.synthetic import test_df

    df = test_df(cols=["CaO", "MgO", "SiO2", "FeO", "Mn", "Ti", "La", "Lu", "Mg/Fe"])








.. code-block:: default


    df.head(2).pyrochem.oxides






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
              <th>CaO</th>
              <th>MgO</th>
              <th>SiO2</th>
              <th>FeO</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>0.006</td>
              <td>0.143</td>
              <td>0.088</td>
              <td>0.141</td>
            </tr>
            <tr>
              <td>1</td>
              <td>0.006</td>
              <td>0.155</td>
              <td>0.114</td>
              <td>0.158</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />


.. code-block:: default


    df.head(2).pyrochem.elements






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
              <th>Mn</th>
              <th>Ti</th>
              <th>La</th>
              <th>Lu</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>0.030</td>
              <td>0.026</td>
              <td>0.048</td>
              <td>0.091</td>
            </tr>
            <tr>
              <td>1</td>
              <td>0.025</td>
              <td>0.027</td>
              <td>0.049</td>
              <td>0.081</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />


.. code-block:: default


    df.head(2).pyrochem.REE






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
              <th>La</th>
              <th>Lu</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>0.048</td>
              <td>0.091</td>
            </tr>
            <tr>
              <td>1</td>
              <td>0.049</td>
              <td>0.081</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />


.. code-block:: default


    df.head(2).pyrochem.compositional






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
              <th>CaO</th>
              <th>MgO</th>
              <th>SiO2</th>
              <th>FeO</th>
              <th>Mn</th>
              <th>Ti</th>
              <th>La</th>
              <th>Lu</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>0.006</td>
              <td>0.143</td>
              <td>0.088</td>
              <td>0.141</td>
              <td>0.030</td>
              <td>0.026</td>
              <td>0.048</td>
              <td>0.091</td>
            </tr>
            <tr>
              <td>1</td>
              <td>0.006</td>
              <td>0.155</td>
              <td>0.114</td>
              <td>0.158</td>
              <td>0.025</td>
              <td>0.027</td>
              <td>0.049</td>
              <td>0.081</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />


.. code-block:: default


    df.pyrochem.list_oxides





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['CaO', 'MgO', 'SiO2', 'FeO']




.. code-block:: default


    df.pyrochem.list_elements





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['Mn', 'Ti', 'La', 'Lu']




.. code-block:: default


    df.pyrochem.list_REE





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['La', 'Lu']




.. code-block:: default


    df.pyrochem.list_compositional





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['CaO', 'MgO', 'SiO2', 'FeO', 'Mn', 'Ti', 'La', 'Lu']



All elements (up to U):



.. code-block:: default

    from pyrolite.geochem.ind import common_elements, common_oxides, REE

    common_elements()  # string return





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']



All elements, returned as a list of `~periodictable.core.Formula`:



.. code-block:: default

    common_elements(output="formula")  # periodictable.core.Formula return





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U]



Oxides for elements with positive charges (up to U):



.. code-block:: default

    common_oxides()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['Th2O', 'ThO', 'Th2O3', 'ThO2', 'Se2O', 'SeO', 'Se2O3', 'SeO2', 'Se2O5', 'SeO3', 'Sb2O', 'SbO', 'Sb2O3', 'SbO2', 'Sb2O5', 'Ta2O', 'TaO', 'Ta2O3', 'TaO2', 'Ta2O5', 'N2O', 'NO', 'N2O3', 'NO2', 'N2O5', 'Sr2O', 'SrO', 'Co2O', 'CoO', 'Co2O3', 'CoO2', 'Co2O5', 'Cu2O', 'CuO', 'Cu2O3', 'CuO2', 'Tc2O', 'TcO', 'Tc2O3', 'TcO2', 'Tc2O5', 'TcO3', 'Tc2O7', 'Al2O', 'AlO', 'Al2O3', 'Zn2O', 'ZnO', 'V2O', 'VO', 'V2O3', 'VO2', 'V2O5', 'Ba2O', 'BaO', 'NdO', 'Nd2O3', 'NdO2', 'Hg2O', 'HgO', 'HgO2', 'TmO', 'Tm2O3', 'DyO', 'Dy2O3', 'DyO2', 'Tl2O', 'TlO', 'Tl2O3', 'As2O', 'AsO', 'As2O3', 'AsO2', 'As2O5', 'S2O', 'SO', 'S2O3', 'SO2', 'S2O5', 'SO3', 'CeO', 'Ce2O3', 'CeO2', 'Be2O', 'BeO', 'Sn2O', 'SnO', 'Sn2O3', 'SnO2', 'Br2O', 'Br2O3', 'BrO2', 'Br2O5', 'Br2O7', 'HoO', 'Ho2O3', 'Mg2O', 'MgO', 'Cs2O', 'Gd2O', 'GdO', 'Gd2O3', 'YbO', 'Yb2O3', 'Si2O', 'SiO', 'Si2O3', 'SiO2', 'Ru2O', 'RuO', 'Ru2O3', 'RuO2', 'Ru2O5', 'RuO3', 'Ru2O7', 'RuO4', 'RaO', 'Sc2O', 'ScO', 'Sc2O3', 'SmO', 'Sm2O3', 'Mn2O', 'MnO', 'Mn2O3', 'MnO2', 'Mn2O5', 'MnO3', 'Mn2O7', 'H2O', 'Li2O', 'In2O', 'InO', 'In2O3', 'C2O', 'CO', 'C2O3', 'CO2', 'K2O', 'Ge2O', 'GeO', 'Ge2O3', 'GeO2', 'W2O', 'WO', 'W2O3', 'WO2', 'W2O5', 'WO3', 'Rb2O', 'Nb2O', 'NbO', 'Nb2O3', 'NbO2', 'Nb2O5', 'Fe2O', 'FeO', 'Fe2O3', 'FeO2', 'Fe2O5', 'FeO3', 'Fe2O7', 'Ni2O', 'NiO', 'Ni2O3', 'NiO2', 'Pa2O3', 'PaO2', 'Pa2O5', 'Ac2O3', 'B2O', 'BO', 'B2O3', 'Fr2O', 'ErO', 'Er2O3', 'Rh2O', 'RhO', 'Rh2O3', 'RhO2', 'Rh2O5', 'RhO3', 'Pd2O', 'PdO', 'Pd2O3', 'PdO2', 'Pd2O5', 'PdO3', 'Pb2O', 'PbO', 'Pb2O3', 'PbO2', 'Re2O', 'ReO', 'Re2O3', 'ReO2', 'Re2O5', 'ReO3', 'Re2O7', 'La2O', 'LaO', 'La2O3', 'Bi2O', 'BiO', 'Bi2O3', 'BiO2', 'Bi2O5', 'Pt2O', 'PtO', 'Pt2O3', 'PtO2', 'Pt2O5', 'PtO3', 'Zr2O', 'ZrO', 'Zr2O3', 'ZrO2', 'RnO', 'RnO3', 'Tb2O', 'TbO', 'Tb2O3', 'TbO2', 'At2O', 'At2O3', 'At2O5', 'At2O7', 'PoO', 'PoO2', 'Po2O5', 'PoO3', 'P2O', 'PO', 'P2O3', 'PO2', 'P2O5', 'EuO', 'Eu2O3', 'Hf2O', 'HfO', 'Hf2O3', 'HfO2', 'Cl2O', 'ClO', 'Cl2O3', 'ClO2', 'Cl2O5', 'ClO3', 'Cl2O7', 'Cd2O', 'CdO', 'U2O', 'UO', 'U2O3', 'UO2', 'U2O5', 'UO3', 'Te2O', 'TeO', 'Te2O3', 'TeO2', 'Te2O5', 'TeO3', 'Mo2O', 'MoO', 'Mo2O3', 'MoO2', 'Mo2O5', 'MoO3', 'Ag2O', 'AgO', 'Ag2O3', 'AgO2', 'PrO', 'Pr2O3', 'PrO2', 'Pr2O5', 'Ca2O', 'CaO', 'LuO', 'Lu2O3', 'Os2O', 'OsO', 'Os2O3', 'OsO2', 'Os2O5', 'OsO3', 'Os2O7', 'OsO4', 'Au2O', 'AuO', 'Au2O3', 'Au2O5', 'Ga2O', 'GaO', 'Ga2O3', 'Na2O', 'I2O', 'I2O3', 'IO2', 'I2O5', 'IO3', 'I2O7', 'Y2O', 'YO', 'Y2O3', 'PmO', 'Pm2O3', 'Ir2O', 'IrO', 'Ir2O3', 'IrO2', 'Ir2O5', 'IrO3', 'Ir2O7', 'IrO4', 'Ir2O9', 'Cr2O', 'CrO', 'Cr2O3', 'CrO2', 'Cr2O5', 'CrO3', 'Ti2O', 'TiO', 'Ti2O3', 'TiO2', 'FeOT', 'Fe2O3T', 'LOI']




.. code-block:: default


    REE()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.361 seconds)


.. _sphx_glr_download_examples_geochem_indexes_selectors.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/geochem/indexes_selectors.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: indexes_selectors.py <indexes_selectors.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: indexes_selectors.ipynb <indexes_selectors.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
