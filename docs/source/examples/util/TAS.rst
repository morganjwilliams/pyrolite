.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_util_TAS.py:


TAS Classifier
==============

Some simple discrimination methods are implemented,
including the Total Alkali-Silica (TAS) classification.


.. code-block:: default

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pyrolite.util.classification import TAS
    from pyrolite.util.synthetic import test_df, random_cov_matrix








We'll first generate some synthetic data to play with:



.. code-block:: default

    df = (
        test_df(
            cols=["SiO2", "Na2O", "K2O", "Al2O3"],
            mean=[0.5, 0.04, 0.05, 0.4],
            index_length=100,
            seed=49,
        )
        * 100
    )

    df.head(3)





.. only:: builder_html

    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }

            .dataframe tbody tr th {
                vertical-align: top;
            }

            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>SiO2</th>
              <th>Na2O</th>
              <th>K2O</th>
              <th>Al2O3</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>0</th>
              <td>48.597785</td>
              <td>3.833775</td>
              <td>4.695366</td>
              <td>42.873074</td>
            </tr>
            <tr>
              <th>1</th>
              <td>50.096300</td>
              <td>3.960378</td>
              <td>5.196130</td>
              <td>40.747192</td>
            </tr>
            <tr>
              <th>2</th>
              <td>51.381566</td>
              <td>4.126436</td>
              <td>5.181051</td>
              <td>39.310947</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

We can visualise how this chemistry corresponds to the TAS diagram:



.. code-block:: default

    import pyrolite.plot

    df["Na2O + K2O"] = df["Na2O"] + df["K2O"]
    cm = TAS()

    fig, ax = plt.subplots(1)
    cm.add_to_axes(
        ax, alpha=0.5, linewidth=0.5, zorder=-1, labels="ID",
    )
    df[["SiO2", "Na2O + K2O"]].pyroplot.scatter(ax=ax, c="k", alpha=0.2)





.. image:: /examples/util/images/sphx_glr_TAS_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.axes._subplots.AxesSubplot object at 0x000001670FE96448>



We can now classify this data according to the fields of the TAS diagram, and
add this as a column to the dataframe. Similarly, we can extract which rock names
the TAS fields correspond to:



.. code-block:: default

    df["TAS"] = cm.predict(df)
    df["Rocknames"] = df.TAS.apply(lambda x: cm.fields.get(x, {"name": None})["name"])
    df["Rocknames"].sample(10) # randomly check 10 sample rocknames




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    78    [Phonotephrite, Foid Monzodiorite]
    2     [Phonotephrite, Foid Monzodiorite]
    56    [Phonotephrite, Foid Monzodiorite]
    41    [Phonotephrite, Foid Monzodiorite]
    93    [Phonotephrite, Foid Monzodiorite]
    60    [Phonotephrite, Foid Monzodiorite]
    61    [Phonotephrite, Foid Monzodiorite]
    47    [Phonotephrite, Foid Monzodiorite]
    4            [Trachyandesite, Foidolite]
    12           [Trachyandesite, Foidolite]
    Name: Rocknames, dtype: object



We could now take the TAS classes and use them to colorize our points for plotting
on the TAS diagram, or more likely, on another plot. Here the relationship to the
TAS diagram is illustrated:



.. code-block:: default


    fig, ax = plt.subplots(1)

    cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, labels="ID")
    df[["SiO2", "Na2O + K2O"]].pyroplot.scatter(ax=ax, c=df['TAS'], alpha=0.7)



.. image:: /examples/util/images/sphx_glr_TAS_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.axes._subplots.AxesSubplot object at 0x000001670E957288>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.488 seconds)


.. _sphx_glr_download_examples_util_TAS.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/util/TAS.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: TAS.py <TAS.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: TAS.ipynb <TAS.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
