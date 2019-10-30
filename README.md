# `pyrolite`

<img src="https://raw.githubusercontent.com/morganjwilliams/pyrolite/develop/docs/source/_static/icon.jpg" alt="pyrolite Logo" width="30%" align="right">

<p align="left">
<a href="https://pypi.python.org/pypi/pyrolite/">
  <img src="https://img.shields.io/pypi/v/pyrolite.svg" alt="PyPI"></a>
<a href="https://zenodo.org/badge/latestdoi/137172322">
  <img src="https://zenodo.org/badge/137172322.svg" alt="DOI"></a>
<a href="https://github.com/morganjwilliams/pyrolite/blob/master/LICENSE" >
  <img src="https://img.shields.io/badge/License-CSIRO_BSD/MIT_License-blue.svg"
       alt="License: CSIRO Modified BSD/MIT License"></a>
</p>

<p align="left">
  <a href="https://pypi.python.org/pypi/pyrolite/">
    <img src="https://img.shields.io/pypi/pyversions/pyrolite.svg"
         alt="Compatible Versions"></a>
  <a href="https://github.com/python/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg"
         alt="Code Style: Black"></a>
 <a href="https://pypistats.org/packages/pyrolite" >
   <img src="https://img.shields.io/pypi/dm/pyrolite.svg?style=flat"
         alt="pyrolite downloads" ></a>
</p>

<p>
<a href="https://pyrolite.readthedocs.io/">
   <img src="https://readthedocs.org/projects/pyrolite/badge/?version=develop" alt="Docs"/></a>
<a href="https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs%2Fnotebooks%2F" ><img src="https://img.shields.io/badge/TryMeOutWith-Binder-F5A252.svg?logo=data:image/png;base64,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" alt="Try pyrolite on Binder"></a>
</p>


### Install

```bash
pip install pyrolite
```

### Build Status


| **master** | **develop** |
|:----------:|:-----------:|
| [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=master)](https://travis-ci.org/morganjwilliams/pyrolite) | [![Build Status](https://travis-ci.org/morganjwilliams/pyrolite.svg?branch=develop)](https://travis-ci.org/morganjwilliams/pyrolite) |
| [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=master)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=master) | [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=develop)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=develop) |

**Maintainer**: Morgan Williams (morgan.williams _at_ csiro.au)

### Contributing

The long-term aim of this project is to be designed, built and supported by (and for)
the geochemistry community. For more information, see the [documentation](https://pyrolite.readthedocs.io/), particularly the [
Contributing page](https://pyrolite.readthedocs.io/en/latest/contributing.html) and [Code of Conduct](https://pyrolite.readthedocs.io/en/latest/conduct.html).
