.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_comp_logtransforms.py:


Log-transforms
----------------

pyrolite includes a few functions for dealing with compositional data, at the heart of
which are i) closure (i.e. everything sums to 100%) and ii) log-transforms to deal with
the compositional space. The commonly used log-transformations include the
Additive Log-Ratio (:func:`~pyrolite.comp.pyrocomp.ALR`), Centred Log-Ratio
(:func:`~pyrolite.comp.pyrocomp.CLR`), and Isometric Log-Ratio
(:func:`~pyrolite.comp.pyrocomp.ILR`) [#ref_1]_ [#ref_2]_.

This example will show you how to access and use some of these functions in pyrolite.

First let's create some example data:



.. code-block:: default

    from pyrolite.util.synthetic import test_df, random_cov_matrix

    df = test_df(
        index_length=100,
        cov=random_cov_matrix(sigmas=[0.1, 0.05, 0.3, 0.6], dim=4, seed=32),
        seed=32,
    )
    df.describe()





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
              <th>CaO</th>
              <th>MgO</th>
              <th>FeO</th>
              <th>TiO2</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>count</th>
              <td>100.000000</td>
              <td>100.000000</td>
              <td>100.000000</td>
              <td>100.000000</td>
              <td>100.000000</td>
            </tr>
            <tr>
              <th>mean</th>
              <td>0.240169</td>
              <td>0.389336</td>
              <td>0.092622</td>
              <td>0.105441</td>
              <td>0.172431</td>
            </tr>
            <tr>
              <th>std</th>
              <td>0.037349</td>
              <td>0.017043</td>
              <td>0.008204</td>
              <td>0.027824</td>
              <td>0.057136</td>
            </tr>
            <tr>
              <th>min</th>
              <td>0.121600</td>
              <td>0.338156</td>
              <td>0.072372</td>
              <td>0.045954</td>
              <td>0.073863</td>
            </tr>
            <tr>
              <th>25%</th>
              <td>0.215840</td>
              <td>0.378866</td>
              <td>0.087937</td>
              <td>0.084228</td>
              <td>0.131423</td>
            </tr>
            <tr>
              <th>50%</th>
              <td>0.239499</td>
              <td>0.390381</td>
              <td>0.092616</td>
              <td>0.102684</td>
              <td>0.168427</td>
            </tr>
            <tr>
              <th>75%</th>
              <td>0.262676</td>
              <td>0.400574</td>
              <td>0.098641</td>
              <td>0.121029</td>
              <td>0.207193</td>
            </tr>
            <tr>
              <th>max</th>
              <td>0.343031</td>
              <td>0.430064</td>
              <td>0.111053</td>
              <td>0.180004</td>
              <td>0.407645</td>
            </tr>
          </tbody>
        </table>
        </div>
        <br />
        <br />

Let's have a look at some of the log-transforms, which can be accessed directly from
your dataframes (via :class:`pyrolite.comp.pyrocomp`), after you've imported
:mod:`pyrolite.comp`. Note that the transformations will return *new* dataframes,
rather than modify their inputs. For example:



.. code-block:: default

    import pyrolite.comp

    lr_df = df.pyrocomp.CLR()  # using a centred log-ratio transformation







The transformations are implemented such that the column names generally make it
evident which transformations have been applied:



.. code-block:: default

    lr_df.columns




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Index(['CLR(SiO2/g)', 'CLR(CaO/g)', 'CLR(MgO/g)', 'CLR(FeO/g)', 'CLR(TiO2/g)'], dtype='object')



To invert these transformations, you can call the respective inverse transform:



.. code-block:: default

    back_transformed = lr_df.pyrocomp.inverse_CLR()







Given we haven't done anything to our dataframe in the meantime, we should be back
where we started, and our values should all be equal within numerical precision.
To verify this, we can use :func:`numpy.allclose`:



.. code-block:: default

    import numpy as np

    np.allclose(back_transformed, df)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    True



In addition to easy access to the transforms, there's also a convenience function
for taking a log-transformed mean (log-transforming, taking a mean, and inverse log
transforming; :func:`~pyrolite.comp.codata.pyrocomp.logratiomean`):



.. code-block:: default


    df.pyrocomp.logratiomean()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    SiO2    0.241138
    CaO     0.395404
    MgO     0.093779
    FeO     0.103583
    TiO2    0.166095
    dtype: float64



While this function defaults to using :func:`~pyrolite.comp.codata.clr`,
you can specify other log-transforms to use:



.. code-block:: default

    df.pyrocomp.logratiomean(transform="CLR")




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    SiO2    0.241138
    CaO     0.395404
    MgO     0.093779
    FeO     0.103583
    TiO2    0.166095
    dtype: float64



Notably, however, the logratio means should all give you the same result:



.. code-block:: default

    np.allclose(
        df.pyrocomp.logratiomean(transform="CLR"),
        df.pyrocomp.logratiomean(transform="ALR"),
    ) & np.allclose(
        df.pyrocomp.logratiomean(transform="CLR"),
        df.pyrocomp.logratiomean(transform="ILR"),
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    True



.. [#ref_1] Aitchison, J., 1984. The statistical analysis of geochemical compositions.
      Journal of the International Association for Mathematical Geology 16, 531–564.
      doi: `10.1007/BF01029316 <https://doi.org/10.1007/BF01029316>`__

.. [#ref_2]  Egozcue, J.J., Pawlowsky-Glahn, V., Mateu-Figueras, G.,
      Barceló-Vidal, C., 2003.
      Isometric Logratio Transformations for Compositional Data Analysis.
      Mathematical Geology 35, 279–300.
      doi: `10.1023/A:1023818214614 <https://doi.org/10.1023/A:1023818214614>`__

.. seealso::

  Examples:
    `Log Ratio Means <logratiomeans.html>`__,
    `Compositional Data <compositional_data.html>`__,
    `Ternary Plots <../ternary.html>`__

  Tutorials:
    `Ternary Density Plots <../../tutorials/ternary_density.html>`__,
    `Making the Logo <../../tutorials/logo.html>`__

  Modules and Functions:
    :mod:`pyrolite.comp.codata`,
    :func:`~pyrolite.comp.codata.boxcox`,
    :func:`~pyrolite.comp.pyrocomp.renormalise`


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.025 seconds)


.. _sphx_glr_download_examples_comp_logtransforms.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/morganjwilliams/pyrolite/develop?filepath=docs/source/examples/comp/logtransforms.ipynb
      :width: 150 px


  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: logtransforms.py <logtransforms.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: logtransforms.ipynb <logtransforms.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
