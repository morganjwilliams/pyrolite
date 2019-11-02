TAS Classifier
==============

Some simple discrimination methods are implemented, including the Total Alkali-Silica (TAS) classification:

.. code-block:: python

    >>> from pyrolite.classification import Geochemistry
    >>>
    >>> cm = Geochemistry.TAS()
    >>> df.TotalAlkali = df.Na2O + df.K2O
    >>> df['TAS'] = cm.classify(df)


This classifier can be quickly added to a bivariate plot, assuming you have data in a pandas DataFrame:

.. code-block:: python

    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>>
    >>> df['TotalAlkali'] = df['Na2O'] + df['K2O']
    >>>
    >>> fig, ax = plt.subplots(1, figsize=(6, 4))
    >>> cm.add_to_axes(ax, facecolor='0.9', edgecolor='k',
    >>>                linewidth=0.5, zorder=-1)
    >>> classnames = cm.clsf.fclasses + ['none']
    >>> df['TAScolors'] = df['TAS'].map(lambda x: classnames.index(x))
    >>> ax.scatter(df.SiO2, df.TotalAlkali, c=df.TAScolors,
    >>>            alpha=0.5, marker='D', s=8, cmap='tab20c')
