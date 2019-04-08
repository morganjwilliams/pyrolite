Tests
========

.. seealso:: `Contributing <contributing.html>`__

If you clone the source repository, unit tests can be run using pytest from the root
directory after installation:

.. code-block:: bash

   python setup.py test


If instead you only want to test a subset, you can call :mod:`pytest` directly from
within the pyrolite repository:

.. code-block:: bash

   pytest ./test/<path to test or test folder>
