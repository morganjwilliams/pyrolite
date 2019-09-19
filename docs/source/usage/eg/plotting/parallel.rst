Parallel Coordinate Plots
============================

Parallel coordinate plots are one way to visualise data relationships and clusters in
higher dimensional data. pyrolite now includes an implementation of this which allows
a handy quick exploratory visualisation.

.. literalinclude:: ../../../../examples/plotting/parallel.py
   :language: python
   :end-before: # %% Data

.. literalinclude:: ../../../../examples/plotting/parallel.py
   :language: python
   :start-after: # %% Data
   :end-before: # %% Default

.. literalinclude:: ../../../../examples/plotting/parallel.py
   :language: python
   :start-after: # %% Default
   :end-before: # %% save

.. image:: ../../../_static/parallel_default.png
   :width: 50%
   :align: center

By rescaling this using the mean and standard deviation, we can account for scale
differences between variables:

.. literalinclude:: ../../../../examples/plotting/parallel.py
  :language: python
  :start-after: # %% Default Rescale
  :end-before: # %% save

.. image:: ../../../_static/parallel_rescale.png
   :width: 50%
   :align: center

We can also use a centred-log transform for compositional data to reduce the effects
of spurious correlation:

.. literalinclude:: ../../../../examples/plotting/parallel.py
  :language: python
  :start-after: # %% CLR
  :end-before: # %% save

.. image:: ../../../_static/parallel_CLR.png
   :width: 50%
   :align: center

Further, we can also rescale this if there remain significant scale differences (e.g.
majors and traces):

.. literalinclude:: ../../../../examples/plotting/parallel.py
  :language: python
  :start-after: # %% CLR Rescale
  :end-before: # %% save

.. image:: ../../../_static/parallel_CLR_rescale.png
   :width: 50%
   :align: center


.. note:: Feature suggested by `Justin Gosses <https://github.com/JustinGOSSES>`__.
