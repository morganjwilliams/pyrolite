Dimensional Reduction
======================

Orthogonal polynomial decomposition can be used for dimensional reduction of smooth
function over an independent variable, producing an array of independent values
representing the relative weights for each order of component polynomial.

In geochemistry, the most applicable use case is for reduction Rare Earth Element (REE)
profiles. The REE are a collection of elements with broadly similar physicochemical
properties (the lanthanides), which vary with ionic radii. Given their similar behaviour
and typically smooth function of normalised abundance vs. ionic radii, the REE profiles
and their shapes can be effectively parameterised and dimensionally reduced (14 elements
summarised by 3-4 shape parameters).

Here we generate some example data, reduce these to lambda values, and plot the
resulting dimensionally reduced data.

.. literalinclude:: ../../../../examples/lambdas/lambdas.py
  :language: python
  :end-before: # %% Generate Some Example Data

.. literalinclude:: ../../../../examples/lambdas/lambdas.py
  :language: python
  :start-after: # %% Generate Some Example Data
  :end-before: # %% Plot Data

.. image:: ../../../_static/LambdaData.png
  :width: 60%
  :align: center
  :alt: Rare Earth Element spider plot of synthetic data showing variable Light REE enrichment.

.. literalinclude:: ../../../../examples/lambdas/lambdas.py
  :language: python
  :start-after: # %% Reduce to Orthogonal Polynomials
  :end-before: # %% Plot the Results

.. literalinclude:: ../../../../examples/lambdas/lambdas.py
  :language: python
  :start-after: # %% Plot the Results
  :end-before: # %% End

.. image:: ../../../_static/Lambdas.png
   :align: center
   :alt: Scatter plot of orthogonal polynomial component weights, here termed lambdas, for the parameterisation of the synthetic dataset.

For more on using orthogonal polynomials to describe geochemical pattern data, see:
O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth
Element Patterns in Basalts. J Petrology 57, 1463–1508.
`doi: 10.1093/petrology/egw047 <https://doi.org/10.1093/petrology/egw047>`__.

.. seealso::

  Examples:
    `Visualising Orthogonal Polynomials <lambdavis.html>`__,
    `Pandas Lambda Ln(REE) Function <pandaslambdas.html>`__
