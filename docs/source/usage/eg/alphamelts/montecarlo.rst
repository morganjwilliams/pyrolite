Analytical Uncertainties and alphaMELTS Experiments
=================================================


.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :end-before: # %% setup an

First we can configure an environment:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% setup an
  :end-before: # %% get the MORB melts file


.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% get the MORB melts file
  :end-before: # %% replicate and add noise


.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% replicate and add noise
  :end-before: # %% run the models


.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% run the models
  :end-before: # %% aggregate the results

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% aggregate the results
  :end-before: # %% save figure

.. image:: ../../../_static/melts_montecarlo.png
   :width: 80%
   :align: center
