.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_geochem_mineral_mindb.py:


Mineral Database
====================

pyrolite includes a limited mineral database which is useful for
for looking up endmember compositions. This part of the package is being actively
developed, so expect expansions and improvements soon.


.. code-block:: default

    import pandas as pd
    from pyrolite.mineral.mindb import (
        list_groups,
        list_minerals,
        list_formulae,
        get_mineral,
        get_mineral_group,
    )

    pd.set_option("precision", 3)  # smaller outputs







From the database, you can get the list of its contents using a few utility
functions:


.. code-block:: default

    list_groups()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['pyroxene', 'feldspar', 'spinel', 'epidote', 'garnet', 'amphibole', 'olivine', 'mica']




.. code-block:: default

    list_minerals()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['taramite', 'piemontite', 'ferrorichterite', 'gedrite', 'phlogopite', 'enstatite', 'jadeite', 'eckermanite', 'celadonite', 'katophorite', 'magnetite', 'magnesiohastingsite', 'margarite', 'namansilite', 'glaucophane', 'spinel', 'epidote', 'johannsenite', 'microcline', 'magnesiohornblende', 'kosmochlor', 'annite', 'polylithionite', 'paragonite', 'ferroeckermanite', 'barroisite', 'aluminoceladonite', 'tschermakite', 'anorthite', 'phengite', 'eastonite', 'hastingsite', 'albite', 'clinozoisite', 'muscovite', 'pyrope', 'siderophyllite', 'andradite', 'hercynite', 'tephroite', 'ferropargasite', 'ferrokaersutite', 'ferrosilite', 'uvarovite', 'chromoceladonite', 'kaersutite', 'morimotoite', 'grossular', 'anthopyllite', 'magnesioarfvedsonite', 'ferrotschermakite', 'ferrokatophorite', 'diopside', 'riebeckite', 'almandine', 'magnesiochromite', 'tremolite', 'winchite', 'pargasite', 'chromite', 'richterite', 'trilithionite', 'magnesioreibeckite', 'clintonite', 'forsterite', 'allanite', 'aegirine', 'ferroaluminoceladonite', 'ferroceladonite', 'majorite', 'magnesioferrite', 'ferrohornblende', 'arvedsonite', 'chromphyllite', 'esseneite', 'spodumene', 'edenite', 'ferroedenite', 'manganiceladonite', 'spessartine', 'liebenbergite', 'hedenbergite', 'fayalite']




.. code-block:: default

    list_formulae()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ['Ni1.5Mg0.5SiO4', 'CaCe{3+}Al2Fe{2+}(Si2O7)(SiO4)O(OH)', 'NaFe{3+}Si2O6', 'Mn2SiO4', 'Fe{2+}Al2O4', 'K2(Fe{3+}2Fe{2+}2)(Si8)O20(OH)4', 'K2(Mn{3+}2Mg2)(Si8)O20(OH)4', 'NaAlSi2O6', 'Ca3(TiFe{2+})(SiO4)3', 'Na(NaCa)(Fe5)(Si8)O22(OH)2', 'Ca3Fe{3+}2(SiO4)3', 'LiAlSi2O6', 'NaCrSi2O6', 'Ca2Al3(Si2O7)(SiO4)O(OH)', 'Na(Ca2)(Mg4Al)(Si6Al2)O22(OH)2', '(Na2)(Fe3Fe{3+}2)(Si8)O22(OH)2', 'CaAlFe{3+}SiO6', 'Na(Na2)(Fe{2+}4Fe{3+})(Si8)O22(OH)2', 'Na(NaCa)(Mg3Al2)(Si6Al2)O22(OH)2', 'Na(Ca2)(Fe4Al)(Si6Al2)O22(OH)2', 'Na2(Al4)(Si6Al2)O20(OH)4', 'K2(Mg2Al2)(Si8)O20(OH)4', '(Ca2)(Mg4Al)(Si7Al)O22(OH)2', 'Na(NaCa)(Mg4Al)(Si7Al)O22(OH)2', 'Mg3Al2(SiO4)3', 'K2(Fe{2+}6)(Si6Al2)O20(OH)4', 'Na(Ca2)(Fe{2+}4Fe{3+})(Si6Al2)O22(OH)2', 'CaAl2Si2O8', 'Ca2(Al4)(Si4Al4)O20(OH)4', 'Na(Ca2)(Mg4Ti)(Si6Al2)O22(OH)2', 'Ca2Al2Mn{3+}(Si2O7)(SiO4)O(OH)', 'K2(Al4)(Si6Al2)O20(OH)4', 'Na(Na2)(Fe4Al)(Si8)O22(OH)2', 'Na(Ca2)(Fe4Ti)(Si6Al2)O22(OH)2', 'K2(Cr{3+}4)(Si6Al2)O20(OH)4', 'K2(Fe{3+}2Mg2)(Si8)O20(OH)4', 'Mg2SiO4', 'K2(Al3Mg)(Si7Al)O20(OH)4', '(NaCa)(Mg3Al2)(Si7Al)O22(OH)2', '(Mg2)(Mg5)(Si8)O22(OH)2', 'Na(NaCa)(Fe4Al)(Si7Al)O22(OH)2', 'Na(Ca2)(Mg4Fe{3+})(Si6Al2)O22(OH)2', 'Mg3(MgSi)(SiO4)3', 'K2(Al3Li3)(Si6Al2)O20(OH)4', 'K2(Fe{2+}2Al2)(Si8)O20(OH)4', 'K2(Mg6)(Si6Al2)O20(OH)4', 'K2(Fe{2+}4)(Si4Al6)O20(OH)4', 'MgFe{3+}2O4', 'Ca3Cr2(SiO4)3', '(Ca2)(Mg5)(Si8)O22(OH)2', 'Ca3Al2(SiO4)3', 'Na(Na2)(Mg4Fe{3+})(Si8)O22(OH)2', 'Mn3Al2(SiO4)3', 'K2(Mg4)(Si4Al6)O20(OH)4', 'Fe2Si2O6', 'Ca2Al2Fe{3+}(Si2O7)(SiO4)O(OH)', 'CaFeSi2O6', 'Fe{2+}3Al2(SiO4)3', 'Mg2Si2O6', 'K2(Mg2Cr{3+}2)(Si8)O20(OH)4', 'NaMn{3+}Si2O6', '(Ca2)(Fe3Al2)(Si6Al2)O22(OH)2', 'MgCr{3+}2O4', '(Na2)(Mg3Al2)(Si8)O22(OH)2', 'Fe2SiO4', '(Ca2)(Mg3Al2)(Si6Al2)O22(OH)2', 'Fe{2+}Fe{3+}2O4', 'MgAl2O4', 'Na(Ca2)(Mg5)(Si7Al)O22(OH)2', 'KAlSi3O8', '(Mg2)(Mg3Al2)(Si6Al2)O22(OH)2', 'Na(Na2)(Mg4Al)(Si8)O22(OH)2', 'Na(Ca2)(Fe5)(Si7Al)O22(OH)2', 'Fe{2+}Cr{3+}2O4', 'K2(Al2Li2)(Si8)O20(OH)4', 'Na(NaCa)(Mg5)(Si8)O22(OH)2', '(Ca2)(Fe4Al)(Si7Al)O22(OH)2', 'CaMnSi2O6', '(Na2)(Mg3Fe{3+}2)(Si8)O22(OH)2', 'Ca2(Mg4Al2)(Si2Al6)O20(OH)4', '(NaCa)(Mg4Al)(Si8)O22(OH)2', 'CaMgSi2O6', 'NaAlSi3O8']



You can also directly get the composition of specific minerals by name:



.. code-block:: default

    get_mineral("forsterite")




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    name       forsterite
    group         olivine
    formula       Mg2SiO4
    Mg              0.346
    Si                0.2
    O               0.455
    Fe                  0
    Mn                  0
    Ni                  0
    Ca                  0
    Al                  0
    Fe{3+}              0
    Na                  0
    Mn{3+}              0
    Cr                  0
    Li                  0
    Cr{3+}              0
    Fe{2+}              0
    K                   0
    H                   0
    Ti                  0
    Ce{3+}              0
    dtype: object



If you want to get compositions for all minerals within a specific group, you can
use :func:`~pyrolite.mineral.mindb.get_mineral_group`:


.. code-block:: default

    get_mineral_group("olivine")





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
              <th>formula</th>
              <th>Mg</th>
              <th>Si</th>
              <th>O</th>
              <th>Fe</th>
              <th>Mn</th>
              <th>Ni</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>forsterite</td>
              <td>Mg2SiO4</td>
              <td>0.346</td>
              <td>0.200</td>
              <td>0.455</td>
              <td>0.000</td>
              <td>0.000</td>
              <td>0.000</td>
            </tr>
            <tr>
              <td>1</td>
              <td>fayalite</td>
              <td>Fe2SiO4</td>
              <td>0.000</td>
              <td>0.138</td>
              <td>0.314</td>
              <td>0.548</td>
              <td>0.000</td>
              <td>0.000</td>
            </tr>
            <tr>
              <td>2</td>
              <td>tephroite</td>
              <td>Mn2SiO4</td>
              <td>0.000</td>
              <td>0.139</td>
              <td>0.317</td>
              <td>0.000</td>
              <td>0.544</td>
              <td>0.000</td>
            </tr>
            <tr>
              <td>3</td>
              <td>liebenbergite</td>
              <td>Ni1.5Mg0.5SiO4</td>
              <td>0.063</td>
              <td>0.146</td>
              <td>0.333</td>
              <td>0.000</td>
              <td>0.000</td>
              <td>0.458</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.047 seconds)


.. _sphx_glr_download_examples_geochem_mineral_mindb.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/geochem/mineral_mindb.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: mineral_mindb.py <mineral_mindb.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: mineral_mindb.ipynb <mineral_mindb.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
