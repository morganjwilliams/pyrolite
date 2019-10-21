Pandas Lambdas from Ln(REE/Reference)
======================================

.. note:: This page is currently incomplete and a work in progress.

.. literalinclude:: ../../../../_examples/lambdas/pandas_lnREE.py
   :language: python
   :end-before: # %% Generate Some Example Data

First we generate some example REE data, and plot this on a
:func:`pyrolite.plot.REE_v_radii`:

.. literalinclude:: ../../../../_examples/lambdas/pandas_lnREE.py
  :language: python
  :start-after: # %% Generate Some Example Data
  :end-before: # %% Plot Data

.. image:: ../../../_static/PandasLambdaData.png
  :width: 60%
  :align: center
  :alt: Rare Earth Element spider plot of synthetic data indexed by radii generated using the pyroplot interface.

The reduction to lambdas using the pandas interface is much simpler than using the
numpy-based utility functions (see :func:`pyrolite.util.math.lambdas`, `Dimensional Reduction <lambdadimreduction.html>`__):

.. literalinclude:: ../../../../_examples/lambdas/pandas_lnREE.py
  :language: python
  :start-after: # %% Reduce to Orthogonal Polynomials
  :end-before: # %% Plot the Results

.. literalinclude:: ../../../../_examples/lambdas/pandas_lnREE.py
  :language: python
  :start-after: # %% Plot the Results
  :end-before: # %% End

.. image:: ../../../_static/PandasLambdas.png
   :align: center
   :alt: Scatter plot of orthogonal polynomial component weights, here termed lambdas, generated using the DataFrame lambda_lnREE method.


For more on using orthogonal polynomials to describe geochemical pattern data, see:
O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth
Element Patterns in Basalts. J Petrology 57, 1463–1508.
`doi: 10.1093/petrology/egw047 <https://doi.org/10.1093/petrology/egw047>`__.

.. seealso::

  Examples:
    `Visualising Orthogonal Polynomials <lambdavis.html>`__,
    `Dimensional Reduction <lambdadimreduction.html>`__,
    `REE Radii Plot <../plotting/REE_v_radii.html>`__
