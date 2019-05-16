Uncertainties and alphaMELTS Experiments
===========================================

While alphaMELTS is a useful tool for formulating hypotheses around magmatic processes,
analytical uncertainties for compositional parameters are difficult to propagate. Here
I've given an example of taking the composition of average MORB, adding 'noise' to
represent multiple possible realisations under analytical uncertainties, and conducted
replicate alphaMELTS experiments to provide some quantification of the uncertainties in
the results. Note that the 'noise' added here is uncorrelated, and as such may usefully
represent analytical uncertainty. Geological uncertainties are typically strongly
correlated, and the uncertainties associated with e.g. variable mineral assemblages
should be modelled differently.

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :end-before: # %% setup an

First we can configure an environment:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% setup an
  :end-before: # %% get the MORB melts file

Then we create a melts file from the MORB composition:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% get the MORB melts file
  :end-before: # %% replicate and add noise

We can replicate this as number of times, then add a small amount of noise in
compositional space:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% replicate and add noise
  :end-before: # %% compostional variation

This represents a relatively small variation in composition:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% compostional variation
  :end-before: # %% save figure

.. image:: ../../../_static/melt_blurredmorb.png
   :width: 60%
   :align: center

Now we can conduct a MELTS run for each of these compositions:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% run the models
  :end-before: # %% aggregate the results

And finally, we can visualise the results, and start to understand uncertainties in the
various parameters we extract from these models. Notably, even 'relatively small'
variations in composition can manifest as significant uncertainties in outputs:

.. literalinclude:: ../../../../examples/alphamelts/montecarlo.py
  :language: python
  :start-after: # %% aggregate the results
  :end-before: # %% save figure


.. image:: ../../../_static/melts_montecarlo.png
   :width: 80%
   :align: center
