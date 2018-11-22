# `pyrolite`

<p align="left">
  <a href="https://pypi.python.org/pypi/pyrolite/">
    <img src="https://img.shields.io/pypi/v/pyrolite.svg" alt="PyPI"></a>
  <a href="https://pyrolite.readthedocs.io/">
     <img src="https://readthedocs.org/projects/pyrolite/badge/?version=latest" alt="Docs"/></a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg"
         alt="Code Style: Black"></a>
  <a href="https://pypi.python.org/pypi/pyrolite/">
    <img src="https://img.shields.io/pypi/pyversions/pyrolite.svg"
         alt="Compatible Versions"></a>
  <a href="https://github.com/morganjwilliams/pyrolite/blob/master/LICENSE" >
    <img src="https://img.shields.io/badge/License-CSIRO_BSD/MIT_License-blue.svg"
         alt="License: CSIRO Modified BSD/MIT License"></a>
  <a href="https://saythanks.io/to/morganjwilliams">
    <img src="https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg"
         alt="Say Thanks"></a>
</p>

## Install

```bash
pip install pyrolite
```

## Build Status


| **master** | **develop** |
|:----------:|:-----------:|
| [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=master)](https://travis-ci.org/morganjwilliams/pyrolite) | [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=develop)](https://travis-ci.org/morganjwilliams/pyrolite) |
| [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=master)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=master) | [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=develop)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=develop) |

**Maintainer**: Morgan Williams (morgan.williams _at_ csiro.au)

## Usage Examples

Note: Examples for compositional data yet to come.

### Elements and Oxides

#### Index Generators

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
>>> co()  # periodictable.formulas.Formula return
[H, He, Li, Be, ...,  Th, Pa, U]
```
REE Elements
```python
>>> from pyrolite.geochem import REE
>>> REE()
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
from pyrolite.util.text import titlecase
from pyrolite.geochem import tochem

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
>>> CH[REE()]
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
>>> df = DMM.data.loc[REE(), ['value']]
>>> spiderplot(CH.normalize(df), label=f'{DMM.Reference}')
```

<img src="https://raw.githubusercontent.com/morganjwilliams/pyrolite/develop/docs/resources/SpiderplotExample.png" alt="SpiderPlotExample" height="250px"/>

More reference compositions will soon be included (e.g. Sun and McDonough, 1989).

### Data Density Plots

Log-spaced data density plots can be useful to visualise geochemical data density:
```python
>>> from pyrolite.plot import densityplot
>>> # with a dataframe <df> containing columns Nb/Yb and Th/Yb
>>> densityplot(df, components=['Nb/Yb', 'Th/Yb'], bins=100, logspace=True)
```
Below is an example of ocean island basalt data
([GEOROC](http://georoc.mpch-mainz.gwdg.de/georoc/) compilation), plotted in a
'Pearce' discrimination diagram:

<img src="https://raw.githubusercontent.com/morganjwilliams/pyrolite/develop/docs/resources/OIB_PearcePlot.png" alt="Ocean Island Basalt Nb/Yb vs Th/Yb" height="250px"/>

More on these discrimination diagrams: [Pearce, J.A., 2008. Geochemical fingerprinting of oceanic basalts with applications to ophiolite classification and the search for Archean oceanic crust. Lithos 100, 14–48.](https://doi.org/10.1016/j.lithos.2007.06.016)


### Dimensional Reduction using Orthagonal Polynomials ('Lambdas')

Derivation of weight values for deconstructing a smooth function into orthagonal
polynomial components (e.g. for the REE):
```python
>>> from pyrolite.geochem import lambda_lnREE
>>> refc = 'Chondrite_PON'
>>> # with a dataframe <df> containing REE data in columns La, ..., Lu
>>> lambdas = lambda_lnREE(df, exclude=['Pm'], norm_to=refc)
```

![Orthagonal Polynomial Example](https://raw.githubusercontent.com/morganjwilliams/pyrolite/develop /docs/resources/LambdaExample.png)

For more on using orthagonal polynomials to describe geochemical pattern data, see: [O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth Element Patterns in Basalts. J Petrology 57, 1463–1508.](https://doi.org/10.1093/petrology/egw047)


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
>>> df['TotalAlkali'] = df['Na2O'] + df['K2O']
>>>
>>> fig, ax = plt.subplots(1, figsize=(6, 4))
>>> cm.add_to_axes(ax, facecolor='0.9', edgecolor='k',
>>>                linewidth=0.5, zorder=-1)
>>> classnames = cm.clsf.fclasses + ['none']
>>> df['TAScolors'] = df['TAS'].map(lambda x: classnames.index(x))
>>> ax.scatter(df.SiO2, df.TotalAlkali, c=df.TAScolors,
>>>            alpha=0.5, marker='D', s=8, cmap='tab20c')
```
