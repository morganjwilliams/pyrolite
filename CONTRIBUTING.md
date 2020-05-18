# Contributing

The long-term aim of this project is to be designed, built and supported by (and for) the geochemistry community. In the present, the majority of the work involves
incorporating geological knowledge and frameworks into a practically useful core set of tools which can be later be expanded. As such, requests for features and bug reports are particularly valuable contributions, in addition to code and expanding the documentation. All individuals contributing to the project are expected to follow the [Code of Conduct](https://pyrolite.readthedocs.io/en/develop/dev/conduct.html), which outlines community expectations and
responsibilities.

Also, be sure to add your name or GitHub username to the
[contributors list](https://pyrolite.readthedocs.io/en/develop/dev/contributors.html).

**Note**: This project is currently in `beta`, and as such there's much work to be done.

## Feature Requests

If you're new to Python, and want to implement a specific process, plot or framework as part of `pyrolite`, you can submit a [Feature Request](https://github.com/morganjwilliams/pyrolite/issues/new?assignees=morganjwilliams&labels=enhancement&template=feature-request.md).
Perhaps also check the [Issues Board](https://github.com/morganjwilliams/pyrolite/issues) first to see if someone else has suggested something similar (or if something is in development), and comment there.

## Bug Reports

If you've tried to do something with `pyrolite`, but it didn't work, and googling
error messages didn't help (or, if the error messages are full of
`pyrolite.XX.xx`), you can submit a [Bug Report](https://github.com/morganjwilliams/pyrolite/issues/new?assignees=morganjwilliams&labels=bug&template=bug-report.md).
Perhaps also check the [Issues Board](https://github.com/morganjwilliams/pyrolite/issues) first to see if someone else is having the same issue, and comment there.

## Contributing to Documentation

The [documentation and examples](https://pyrolite.readthedocs.io) for `pyrolite`
are gradually being developed, and any contributions or corrections would be greatly appreciated. Currently the examples are patchy, and any 'getting started' guides would be a helpful addition.

These pages serve multiple purposes:

* A human-readable reference of the source code (compiled from docstrings).
* A set of simple examples to demonstrate use and utility.
* A place for developing extended examples

  * Note: these examples could easily be distributed as educational resources showcasing the utility of programmatic approaches to geochemistry

## Contributing Code

Code contributions are always welcome, whether it be small modifications or entire
features. As the project gains momentum, check the [Issues Board](https://github.com/morganjwilliams/pyrolite/issues) for outstanding issues, features under development. If you'd like to contribute, but you're not so
experienced with Python, look for `good first issue` tags or email the maintainer
for suggestions.

To contribute code, the place to start will be forking the source for `pyrolite`
from [GitHub](https://github.com/morganjwilliams/pyrolite/tree/develop). Once forked, clone a local copy and from the repository directory you can install a development (editable) copy via `python setup.py develop`. To incorporate suggested
changes back to into the project, push your changes to your remote fork, and then submit a pull request onto [pyrolite/develop](https://github.com/morganjwilliams/pyrolite/tree/develop) or a relevant feature branch.

Notes:

* See the [Installation page](https://pyrolite.readthedocs.io/en/develop/installation.html) for directions for installing extra dependencies for development, and the [Development page](https://pyrolite.readthedocs.io/en/develop/dev/development.html) for information on development environments and tests.

* `pyrolite` development roughly follows a [`gitflow` workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
  `pyrolite/master` is only used for releases, and large separable features
  should be build on `feature` branches off `develop`.

* Contributions introducing new functions, classes or entire features should
  also include appropriate tests where possible (see [Writing Tests](#writing-tests), below).

* `pyrolite` uses [Black](https://github.com/python/black/) for code formatting, and submissions which have passed through `Black` are appreciated, although not critical.


Writing Tests
--------------

There is currently a broad unit test suite for `pyrolite`, which guards
against breaking changes and assures baseline functionality. `pyrolite` uses continuous integration via [Travis](https://travis-ci.org/morganjwilliams/pyrolite), where the full suite of tests are run for each commit and pull request, and test coverage output to [Coveralls](https://coveralls.io/github/morganjwilliams/pyrolite).

Adding or expanding tests is a helpful way to ensure `pyrolite` does what is meant to, and does it reproducibly. The unit test suite one critical component of the package, and necessary to enable sufficient trust to use `pyrolite` for scientific purposes.
