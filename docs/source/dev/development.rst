Development
=============

Development History and Planning
--------------------------------

* `Changelog <changelog.html>`__
* `Roadmap <roadmap.html>`__


Contributing
--------------

* `Contributing <contributing.html>`__
* `Contributors <contributors.html>`__
* `Code of Conduct <conduct.html>`__


Development Installation
----------------------------

To access and use the development version, you can 
`clone the repository <https://github.com/morganjwilliams/pyrolite>`__ and 
set up the environment:

.. code-block:: bash

  git clone https://github.com/morganjwilliams/pyrolite.git
  git checkout develop
  uv sync --extra dev


Tests
---------

If you clone the source repository, unit tests can be run using pytest from the root
directory after installation with development dependencies
(:code:`pip install -e .[dev]`):

.. code-block:: bash

   uv run pytest


If instead you only want to test a subset, you can call :mod:`pytest` directly from
within the pyrolite repository:

.. code-block:: bash

   uv run pytest ./test/<path to test or test folder>
