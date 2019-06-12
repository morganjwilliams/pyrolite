Automating alphaMELTS runs
=============================

pyrolite includes some utilities to help you run alphaMELTS with a little less hassle,
especially for established workflows or repetitive calculations.

.. literalinclude:: ../../../../examples/alphamelts/automation.py
  :language: python
  :end-before: # %% testdf

First we can configure an environment, which in this case is written to file:

.. literalinclude:: ../../../../examples/alphamelts/automation.py
  :language: python
  :start-after: # %% setup environment
  :end-before: # %% setup dataframe

We can then add individual experiment parameters to the dataframe:

.. literalinclude:: ../../../../examples/alphamelts/automation.py
  :language: python
  :start-after: # %% setup dataframe
  :end-before: # %% autorun

And finally, we can run an experiment for each composition in the dataframe:

.. literalinclude:: ../../../../examples/alphamelts/automation.py
  :language: python
  :start-after: # %% autorun


.. seealso::

  Examples:
    `alphaMELTS Environment Configuration <environment.html>`__,
    `pyrolite-hosted alphaMELTS Installation <localinstall.html>`__,
    `Handling Outputs from Melts: Tables <tables.html>`__,
    `Compositional Uncertainty Propagation for alphaMELTS Experiments <montecarlo.html>`__
