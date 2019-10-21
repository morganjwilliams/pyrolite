Development
=============

Development History and Planning
--------------------------------

* `Changelog <changelog.html>`__
* `Future <future.html>`__


Contributing
--------------

* `Contributing <contributing.html>`__
* `Contributors <contributors.html>`__
* `Code of Conduct <conduct.html>`__


Development Installation
----------------------------

To access and use the development version, you can either
`clone the repository <https://github.com/morganjwilliams/pyrolite>`__ or install
via pip directly from GitHub:

.. code-block:: bash

  pip install git+git://github.com/morganjwilliams/pyrolite.git@develop#egg=pyrolite


Tests
---------

If you clone the source repository, unit tests can be run using pytest from the root
directory after installation with development dependencies
(:code:`pip install -e .[dev]`):

.. code-block:: bash

   python setup.py test


If instead you only want to test a subset, you can call :mod:`pytest` directly from
within the pyrolite repository:

.. code-block:: bash

   pytest ./test/<path to test or test folder>
