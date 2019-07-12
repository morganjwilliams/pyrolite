Element-Oxide Transformation
-----------------------------

One of pyrolite's strengths is converting mixed elemental and oxide data to a new
form. The simplest way to perform this is by using the
:func:`~pyrolite.geochem.transform.convert_chemistry` function. Note that by default
pyrolite assumes that data are in the same units.

.. literalinclude:: ../../../../examples/geochem/convert_chemistry.py
  :language: python
  :end-before: # %% Random data

Here we create some synthetic data to work with, which has some variables in Wt% and
some in ppm. Notably some elements are present in more than one column (Ca, Na):

.. literalinclude:: ../../../../examples/geochem/convert_chemistry.py
  :language: python
  :start-after:  # %% Random data
  :end-before: # %% Unit Conversion

As the units are heterogeneous, we'll need to convert the data frame to a single set of
units (here we use Wt%):

.. literalinclude:: ../../../../examples/geochem/convert_chemistry.py
  :language: python
  :start-after:  # %% Unit Conversion
  :end-before: # %% Conversion

We can transform this chemical data to a new set of compositional variables.
Note that you can also use this function to calculate mass ratios:

.. literalinclude:: ../../../../examples/geochem/convert_chemistry.py
  :language: python
  :start-after: # %% Conversion
  :end-before: # %% Iron Conversion

.. code-block:: python

  >>> new_df.columns

  Index(['MgO', 'SiO2', 'FeO', 'CaO', 'Te', 'Na', 'Na/Te', 'MgO/SiO2'], dtype='object')

You can also specify molar ratios for iron redox, which will result in multiple iron
species within the single dataframe:

.. literalinclude:: ../../../../examples/geochem/convert_chemistry.py
  :language: python
  :start-after: # %% Iron Conversion

.. code-block:: python

  >>> new_df.columns

  Index(['MgO', 'SiO2', 'CaO', 'FeO', 'Fe2O3'], dtype='object')
