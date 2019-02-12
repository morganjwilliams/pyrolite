Data Cleaning
==============

Some simple utilities for cleaning up data tables are included.
Assuming you're importing data into `pd.DataFrame`

.. code-block:: python

  import pandas as pd
  df = pd.DataFrame({'label':'basalt', 'ID': 19076,
                     'mgo':20.0, 'SIO2':30.0, 'cs':5.0, 'TiO2':2.0},
                    index=[0])
  >>> df.columns
  Index(['label', 'ID', 'mgo', 'SIO2', 'cs', 'TiO2'], dtype='object')

.. code-block:: python

  from pyrolite.util.text import titlecase
  from pyrolite.geochem.parse import tochem

  >>> df.columns = [titlecase(h, abbrv=['ID']) for h in df.columns]
  Index(['Label', 'ID', 'Mgo', 'Sio2', 'Cs', 'Tio2'], dtype='object')
  >>> df.columns = tochem(df.columns)
  Index(['Label', 'ID', 'MgO', 'SiO2', 'Cs', 'TiO2'], dtype='object')
