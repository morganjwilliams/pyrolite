# `pyrolite`

Maintainer: Morgan Williams (morgan.williams _at_ csiro.au)

## Build Status
[![PyPI](https://img.shields.io/pypi/v/pyrolite.svg)](https://pypi.python.org/pypi/pyrolite/)

| **master** | **develop** |
|:----------:|:-----------:|
| [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=master)](https://travis-ci.org/morganjwilliams/pyrolite) | [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=develop)](https://travis-ci.org/morganjwilliams/pyrolite) |
| [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=master)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=master) | [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=develop)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=develop) |

## License

[CSIRO Modified BSD/MIT License](https://raw.githubusercontent.com/morganjwilliams/pyrolite/master/LICENSE)

## Usage Examples

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
>>> from pyrolite.geochem import REE as ree
>>> ree(output='str')
['La', 'Ce', 'Pr', 'Nd', 'Pm', ..., 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
```

### Compositional Data


### Normalisation


### Classification
