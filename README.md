# `pyrolite`

## Install

```bash
pip install pyrolite
```

## Build Status
[![PyPI](https://img.shields.io/pypi/v/pyrolite.svg)](https://pypi.python.org/pypi/pyrolite/)

**License**:  [CSIRO Modified BSD/MIT License](https://raw.githubusercontent.com/morganjwilliams/pyrolite/master/LICENSE)

**Maintainer**: Morgan Williams (morgan.williams _at_ csiro.au)

| **master** | **develop** |
|:----------:|:-----------:|
| [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=master)](https://travis-ci.org/morganjwilliams/pyrolite) | [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=develop)](https://travis-ci.org/morganjwilliams/pyrolite) |
| [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=master)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=master) | [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=develop)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=develop) |

## Usage Examples

Note: Examples for compositional data yet to come.

### Elements and Oxides

#### Index Generators

All Elements up to U
```python
>>> import pyrolite.common_elements as ce
>>> ce()  # periodictable.core.Element return
[H, He, Li, Be, ...,  Th, Pa, U]
>>> ce(output='str')  # string return
['H', 'He', 'Li', 'Be', ...,  'Th', 'Pa', 'U']
```
Oxides for Elements with Positive Charges (up to U)
```python
>>> import pyrolite.common_oxides as co
>>> co()  # periodictable.formulas.Formula return
[H, He, Li, Be, ...,  Th, Pa, U]
>>> co(output='str')  # string return
['H2O', 'He2O', 'HeO', 'Li2O', 'Be2O', 'BeO', 'B2O', 'BO', 'B2O3', ...,
'U2O', 'UO', 'U2O3', 'UO2', 'U2O5', 'UO3']
```
REE Elements
```python
>>> from pyrolite.geochem import REE
>>> REE(output='str')
['La', 'Ce', 'Pr', 'Nd', 'Pm', ..., 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
```

#### Data Cleaning

Some simple utilities for cleaning up data tables are included. Assuming you're importing data into `pd.DataFrame`:
```python
import pandas as pd
df = pd.DataFrame({'label':'basalt', 'ID': 19076,
                   'mgo':20.0, 'SIO2':30.0, 'cs':5.0, 'TiO2':2.0},
                  index=[0])
>>> df.columns
Index(['label', 'ID', 'mgo', 'SIO2', 'cs', 'TiO2'], dtype='object')
```
```python
from pyrolite.textutil import titlecase, tochem

>>> df.columns = [titlecase(h, abbrv=['ID']) for h in df.columns]
Index(['Label', 'ID', 'Mgo', 'Sio2', 'Cs', 'Tio2'], dtype='object')
>>> df.columns = tochem(df.columns)
Index(['Label', 'ID', 'MgO', 'SiO2', 'Cs', 'TiO2'], dtype='object')
```

### Normalisation

A selection of reference compositions are included:

```python
>>> from pyrolite.normalisation import ReferenceCompositions
>>> refcomp = ReferenceCompositions()
{
'Chondrite_PON': Model of Chondrite (Palme2014),
'D-DMM_WH': Model of DepletedDepletedMORBMantle (Workman2005),
'DMM_WH': Model of DepletedMORBMantle (Workman2005),
'DM_SS': Model of DepletedMantle (Salters2004),
'E-DMM_WH': Model of EnrichedDepletedMORBMantle (Workman2005),
'PM_PON': Model of PrimitiveMantle (Palme2014)
}
```

```python
>>> CH = refcomp['Chondrite_PON']
>>> PM = refcomp['PM_PON']
>>> reels = ree(output='str')
>>> CH[reels]
      value  unc_2sigma units
var                           
La    0.2414    0.014484   ppm
Ce    0.6194    0.037164   ppm
...
Tm   0.02609    0.001565   ppm
Yb    0.1687    0.010122   ppm
Lu   0.02503    0.001502   ppm
```

The `normalize` method can be used to normalise dataframes to a given reference (e.g. for spiderplots):
```python
>>> from pyrolite.plot import spiderplot
>>> refcomp = ReferenceCompositions()
>>> CH = refcomp['Chondrite_PON']
>>> DMM = refcomp['DMM_WH']
>>>
>>> reels = ree(output='str')
>>> df = DMM.data.loc[reels, ['value']]
>>> spiderplot(CH.normalize(df), label=f'{DMM.Reference}')
```

### Classification

Some simple discrimination methods are implemented, including the Total Alkali-Silica (TAS) classification:

```python
>>> from pyrolite.classification import Geochemistry
>>>
>>> cm = Geochemistry.TAS()
>>> df.TotalAlkali = df.Na2O + df.K2O
>>> df['TAS'] = cm.classify(df)
```
This classifier can be quickly added to a bivariate plot, assuming you have data in a pandas DataFrame:
```python
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>>
>>> fig, ax = plt.subplots(1, figsize=(6, 4))
>>> cm.add_to_axes(ax, facecolor='0.9', edgecolor='k',
>>>                linewidth=0.5, zorder=-1)
>>> classnames = cm.clsf.fclasses + ['none']
>>> df['TAScolors'] = df['TAS'].map(lambda x: classnames.index(x))
>>> ax.scatter(df.SiO2, df.TotalAlkali, c=df.TAScolors,
>>>            alpha=0.5, marker='D', s=8, cmap='tab20c')
```
