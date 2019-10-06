Ternary Plots
=============

Note that this is a thin wrapper around Marc Harper's
`python-ternary <https://github.com/marcharper/python-ternary>`__ package. This may
change in the near future as needs change, but works well for scatter plots.

.. literalinclude:: ../../../../examples/plotting/tern.py
   :language: python
   :end-before: # %% Minimal Example

.. literalinclude:: ../../../../examples/plotting/tern.py
   :language: python
   :start-after: # %% Minimal Example
   :end-before: # %% Save Figure

.. image:: ../../../_static/tern_minimal.png
   :width: 50%
   :align: center
   :alt: A minimal example of a simple ternary plot with three synthetic data points.

.. literalinclude:: ../../../../examples/plotting/tern.py
   :language: python
   :start-after: # %% Specify External Axis
   :end-before: # %% Save Figure

.. image:: ../../../_static/tern_dual.png
   :width: 100%
   :align: center
   :alt: Two ternary plots placed on separate axes by specifying a pre-existing axis.

.. seealso:: `Density Plots <../plotting/density.html>`__,
             `Heatmapped Scatter Plots <heatscatter.html>`__
