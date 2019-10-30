Automating alphaMELTS runs
=============================

pyrolite includes some utilities to help you run alphaMELTS with a little less hassle,
especially for established workflows or repetitive calculations. Here we run multiple
experiments at different conditions for a single MORB composition. Once we have the
data in a :class:`~pandas.DataFrame`, we configure the default alphaMELTS environment
before running the batch of experiments.

.. literalinclude:: automation.py
  :language: python
  :end-before: # %% Data

.. literalinclude:: automation.py
  :language: python
  :start-after: # %% Data
  :end-before: # %% Environment

.. literalinclude:: automation.py
  :language: python
  :start-after: # %% Environment
  :end-before: # %% Batch

.. literalinclude:: automation.py
  :language: python
  :start-after: # %% Batch

.. code-block:: bash

  __main__ - INFO - Starting 5 Calculations for 1 Compositions.
  __main__ - INFO - Estimated Time: 0:00:30
  __main__ - INFO - 0%|          | 0/5 [00:00<?, ?it/s]
  __main__ - INFO - 20%|##        | 1/5 [00:04<00:18,  4.56s/it]
  __main__ - INFO - 40%|####      | 2/5 [00:09<00:13,  4.56s/it]
  __main__ - INFO - 60%|######    | 3/5 [00:14<00:09,  4.71s/it]
  __main__ - INFO - 80%|########  | 4/5 [00:18<00:04,  4.68s/it]
  __main__ - INFO - 100%|##########| 5/5 [00:23<00:00,  4.80s/it]
  __main__ - INFO - Calculations Complete after 0:00:23.873137


.. seealso::

  Examples:
    `alphaMELTS Environment Configuration <environment.html>`__,
    `pyrolite-hosted alphaMELTS Installation <localinstall.html>`__,
    `Handling Outputs from Melts: Tables <tables.html>`__,
    `Compositional Uncertainty Propagation for alphaMELTS Experiments <montecarlo.html>`__
