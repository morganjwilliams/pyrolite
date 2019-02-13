Elements and Oxides
=====================

All Elements up to U

.. code-block:: python

  >>> import pyrolite.geochem.ind.common_elements as ce
  >>> ce()  # string return
  ['H', 'He', 'Li', 'Be', ...,  'Th', 'Pa', 'U']
  >>> ce(output='formula')  # periodictable.core.Element return
  [H, He, Li, Be, ...,  Th, Pa, U]

Oxides for Elements with Positive Charges (up to U)

.. code-block:: python

  >>> import pyrolite.geochem.ind.common_oxides as co
  >>> co()  # string return
  ['H2O', 'He2O', 'HeO', 'Li2O', 'Be2O', 'BeO', 'B2O', 'BO', 'B2O3', ...,
  'U2O', 'UO', 'U2O3', 'UO2', 'U2O5', 'UO3']
  >>> co(output='formula')  # periodictable.formulas.Formula return
  [H, He, Li, Be, ...,  Th, Pa, U]

REE Elements

.. code-block:: python

  >>> from pyrolite.geochem.ind import REE
  >>> REE()
  ['La', 'Ce', 'Pr', 'Nd', 'Pm', ..., 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
