# `pyrolite`

<img src="https://raw.githubusercontent.com/morganjwilliams/pyrolite/develop/docs/source/_static/icon.jpg" alt="pyrolite Logo" width="30%" align="right">

[![pyrolite Documentation](https://readthedocs.org/projects/pyrolite/badge/?version=develop)](https://pyrolite.readthedocs.io/)
[![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/20)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02314/status.svg)](https://doi.org/10.21105/joss.02314)
[![License: CSIRO Modified BSD/MIT License](https://img.shields.io/badge/License-CSIRO_BSD/MIT_License-blue.svg?style=flat)](https://github.com/morganjwilliams/pyrolite/blob/master/LICENSE)
[![Try pyrolite on Binder](https://img.shields.io/badge/TryItOutWith-Binder-F5A252.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs%2Fnotebooks%2F)
[![Chat on Gitter](https://img.shields.io/gitter/room/pyrolite/community.svg)](https://gitter.im/pyrolite/community/)

pyrolite is a set of tools for making the most of your geochemical data.

The python package includes functions to work with compositional data, to transform
geochemical variables (e.g. elements to oxides), functions for common plotting
tasks (e.g. spiderplots, ternary diagrams, bivariate and ternary density diagrams),
and numerous auxiliary utilities.

pyrolite's is principally developed for use in geochemical research, but is also
well suited to being incorporated into university-level geochemistry and petrology
classes which wish to include a little Python. The documentation is continually
evolving, and more examples and tutorials will gradually be added (feel free to
request features or examples; see [Contributing](#contributing) below).

## Install

[![PyPI](https://img.shields.io/pypi/v/pyrolite.svg?style=flat)](https://pypi.python.org/pypi/pyrolite)
[![Compatible Python Versions](https://img.shields.io/pypi/pyversions/pyrolite.svg?style=flat)](https://pypi.python.org/pypi/pyrolite/)
[![pyrolite downloads](https://img.shields.io/pypi/dm/pyrolite.svg?style=flat)](https://pypistats.org/packages/pyrolite)

```bash
pip install pyrolite
```

If you want the most up to date *development* version, you can instead install directly from the GitHub repo. Note that breaking changes occur on this branch, and is not guaranteed to remain stable (check the [Development and Build Status](#development--build-status) below). If you still want to try out the most recent bugfixes and yet-to-be-released features, you can install this version with:

```bash
pip install git+git://github.com/morganjwilliams/pyrolite.git@develop#egg=pyrolite
```

For more information, see the documentation's [installation page](https://pyrolite.readthedocs.io/en/master/installation.html), and the [Getting Started Guide](https://pyrolite.readthedocs.io/en/master/gettingstarted.html).

## Examples

Check out the documentation for galleries of [examples](https://pyrolite.readthedocs.io/en/master/examples/index.html) and [tutorials](https://pyrolite.readthedocs.io/en/master/tutorials/index.html). If you'd rather flip through notebooks here on GitHub, these same examples can be found in the folders [`docs/source/examples`](./docs/source/examples/) and [`docs/source/tutorials`](./docs/source/examples/).

## Contributing

The long-term aim of this project is to be designed, built and supported by (and for) the geochemistry community. The project welcomes feature requests, bug reports and contributions to the code base, documentation and test suite. We're happy to help onboard new contributors and walk you through the process. Check out the [Issues Board](https://github.com/morganjwilliams/pyrolite/issues) to get an idea of some of the some of the currently identified bugs and things we're looking to work on. For more information, see the [documentation](https://pyrolite.readthedocs.io/), particularly the [
Contributing page](https://pyrolite.readthedocs.io/en/develop/contributing.html) and [Code of Conduct](https://pyrolite.readthedocs.io/en/develop/conduct.html).

For a list of people who have helped build and improve pyrolite, check out the [Contributors page](https://pyrolite.readthedocs.io/en/develop/dev/contributors.html).

If you'd like an idea of where the project might be heading in the near future, have a look at [the current roadmap](https://pyrolite.readthedocs.io/en/develop/dev/future.html).

## Citation
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02314/status.svg)](https://doi.org/10.21105/joss.02314)
[![Archive](https://zenodo.org/badge/137172322.svg?style=flat)](https://zenodo.org/badge/latestdoi/137172322)

If you use pyrolite extensively for your research, citation of the software would be particularly appreciated. It helps quantify the impact of the project (assisting those contributing through paid and volunteer work), and is one way to get the message out and help build the pyrolite community. For information on citing pyrolite, [see the relevant docs page](https://pyrolite.readthedocs.io/en/develop/cite.html).

## Development & Build Status

[![Formatted with Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Code Quality](https://api.codacy.com/project/badge/Grade/fd9912a3faae43bf84a47e3da685d84c)](https://www.codacy.com/manual/morganjwilliams/pyrolite?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=morganjwilliams/pyrolite&amp;utm_campaign=Badge_Grade)

|                                                                                  **master**                                                                                  |                                                                                  **develop**                                                                                   |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                               [![Unit Tests on Master](https://github.com/morganjwilliams/pyrolite/workflows/Unittest/badge.svg?branch=master)](https://github.com/morganjwilliams/pyrolite/actions?query=workflow:Unittest+branch:master)                                |                               [![Unit Tests on Develop](https://github.com/morganjwilliams/pyrolite/workflows/Unittest/badge.svg?branch=develop)](https://github.com/morganjwilliams/pyrolite/actions?query=workflow:Unittest+branch:develop)                                |
| [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=master)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=master) | [![Coverage Status](https://coveralls.io/repos/github/morganjwilliams/pyrolite/badge.svg?branch=develop)](https://coveralls.io/github/morganjwilliams/pyrolite?branch=develop) |

**Maintainer**: Morgan Williams (morgan.williams _at_ csiro.au)
