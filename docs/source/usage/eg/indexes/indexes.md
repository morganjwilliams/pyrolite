# Elements and Oxides


All Elements up to U
```python
>>> import pyrolite.geochem.common_elements as ce
>>> ce()  # string return
['H', 'He', 'Li', 'Be', ...,  'Th', 'Pa', 'U']
>>> ce(output='formula')  # periodictable.core.Element return
[H, He, Li, Be, ...,  Th, Pa, U]
```
Oxides for Elements with Positive Charges (up to U)
```python
>>> import pyrolite.geochem.common_oxides as co
>>> co()  # string return
['H2O', 'He2O', 'HeO', 'Li2O', 'Be2O', 'BeO', 'B2O', 'BO', 'B2O3', ...,
'U2O', 'UO', 'U2O3', 'UO2', 'U2O5', 'UO3']
>>> co(output='formula')  # periodictable.formulas.Formula return
[H, He, Li, Be, ...,  Th, Pa, U]
```
REE Elements
```python
>>> from pyrolite.geochem import REE
>>> REE()
['La', 'Ce', 'Pr', 'Nd', 'Pm', ..., 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
```

#### Data Cleaning

Some simple utilities for cleaning up data tables are included.
Assuming you're importing data into `pd.DataFrame`:
```python
import pandas as pd
df = pd.DataFrame({'label':'basalt', 'ID': 19076,
                   'mgo':20.0, 'SIO2':30.0, 'cs':5.0, 'TiO2':2.0},
                  index=[0])
>>> df.columns
Index(['label', 'ID', 'mgo', 'SIO2', 'cs', 'TiO2'], dtype='object')
```
```python
from pyrolite.util.text import titlecase
from pyrolite.geochem import tochem

>>> df.columns = [titlecase(h, abbrv=['ID']) for h in df.columns]
Index(['Label', 'ID', 'Mgo', 'Sio2', 'Cs', 'Tio2'], dtype='object')
>>> df.columns = tochem(df.columns)
Index(['Label', 'ID', 'MgO', 'SiO2', 'Cs', 'TiO2'], dtype='object')
```
