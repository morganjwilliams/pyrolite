Normalisation
==============

A selection of reference compositions are included:

.. code-block:: python

  >>> from pyrolite.geochem.norm import ReferenceCompositions
  >>> refcomp = ReferenceCompositions()
  >>> refcomp
  {
   'BCC_RG2003': Model of BulkContinentalCrust (Rudnick & Gao 2003),
   'BCC_RG2014': Model of BulkContinentalCrust (Rudnick & Gao 2014),
   'Chondrite_MS95': Model of Chondrite (McDonough & Sun 1995),
   ...
   'UCC_RG2014': Model of UpperContinentalCrust (Rudnick & Gao 2014)
  }


.. code-block:: python

  >>> CH = refcomp['Chondrite_PON']
  >>> PM = refcomp['PM_PON']
  >>> CH[REE()]
        value  unc_2sigma units
  var
  La    0.2414    0.014484   ppm
  Ce    0.6194    0.037164   ppm
  ...
  Tm   0.02609    0.001565   ppm
  Yb    0.1687    0.010122   ppm
  Lu   0.02503    0.001502   ppm


The `normalize` method can be used to normalise dataframes to a given reference (e.g. for spiderplots):

.. code-block:: python

  >>> from pyrolite.plot import spiderplot
  >>> refcomp = ReferenceCompositions()
  >>> CH = refcomp['Chondrite_PON']
  >>> DMM = refcomp['DMM_WH2005']

  >>> df = DMM.data.loc[REE(), ['value']]
  >>> spiderplot(CH.normalize(df), label=f'{DMM.Reference}')


.. image:: ../../../_static/NormSpiderplot.png
   :height: 250px
   :align: center

.. seealso:: `Pandas Lambda Ln(REE) Function <../lambdas/pandaslambdas.html>`__,
             `Lambdas for Dimensional Reduction <../lambdas/lambdadimreduction.html>`__,
             `REE Radii Plot <../plotting/REE_radii_plot.html>`__

Currently available models include:

|refcomps|
