Stem Plots
==================================

Stem and leaf plots are commonly used to visualise discrete distributions of data,
and are useful to highlight discrete observations where the precision of values along
one axis is high (e.g. a spatial measure like depth) and the other is less so.

.. literalinclude:: ../../../../examples/plotting/stem.py
   :language: python
   :end-before: # %% Minimal Example

A minimal density plot can be constructed as follows:

.. literalinclude:: ../../../../examples/plotting/stem.py
   :language: python
   :start-after: # %% Minimal Example
   :end-before: # %% Save Figure

.. image:: ../../../_static/stem_minimal.png
   :width: 50%
   :align: center

Stem plots can also be used in a vertical orientation, such as for visualising
discrete observations down a drill hole:

.. literalinclude:: ../../../../examples/plotting/stem.py
   :language: python
   :start-after: # %% Orientation
   :end-before: # %% Save Figure

.. image:: ../../../_static/stem_vertical.png
   :width: 50%
   :align: center
