Ternary Plots
=============

Note that this is a thin wrapper around Marc Harper's
`python-ternary <https://github.com/marcharper/python-ternary>`__ package. This may
change in the near future as needs change, but works well for scatter plots.

.. literalinclude:: ../../../../examples/plotting/ternaryplot.py
   :language: python
   :end-before: # %% Minimal Example

.. literalinclude:: ../../../../examples/plotting/ternaryplot.py
   :language: python
   :start-after: # %% Minimal Example
   :end-before: # %% Save Figure

.. image:: ../../../_static/ternaryplot_minimal.png
   :width: 50%
   :align: center

.. literalinclude:: ../../../../examples/plotting/ternaryplot.py
   :language: python
   :start-after: # %% Specify External Axis
   :end-before: # %% Save Figure

.. image:: ../../../_static/ternaryplot_dual.png
   :width: 100%
   :align: center

.. seealso:: `Density Plots <../plotting/densityplot.html>`__
