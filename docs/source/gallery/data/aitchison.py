"""
Aitchison Examples
==================

:mod:`pyrolite` includes four synthetic datasets which are used in [Aitchison1984]_
which can be accessed using each of the respective functions
:func:`~pyrolite.data.Aitchison.load_boxite`,
:func:`~pyrolite.data.Aitchison.load_coxite`,
:func:`~pyrolite.data.Aitchison.load_hongite` and
:func:`~pyrolite.data.Aitchison.load_kongite`
(all returning a :class:`~pandas.DataFrame`).

"""
from pyrolite.data.Aitchison import load_boxite, load_coxite, load_hongite, load_kongite

df = load_boxite()
df.head()
########################################################################################
import matplotlib.pyplot as plt
import pyrolite.plot

fig, ax = plt.subplots(1)
for loader in [load_boxite, load_coxite, load_hongite, load_kongite]:
    df = loader()
    ax = df[["A", "B", "C"]].pyroplot.scatter(ax=ax, label=df.attrs["name"])

ax.legend()
plt.show()
########################################################################################
# References
# ~~~~~~~~~~~
#
# .. [Aitchison1984] Aitchison, J., 1984.
#    The statistical analysis of geochemical compositions.
#    Journal of the International Association for Mathematical Geology 16, 531–564.
#    `doi: 10.1007/BF01029316 <https://doi.org/10.1007/BF01029316>`__
#
# .. seealso::
#
#   Examples:
#     `Log Ratio Means <../examples/comp/logratiomeans.html>`__,
#     `Log Transforms <../examples/comp/logtransforms.html>`__,
#     `Compositional Data <../examples/comp/compositional_data.html>`__,
#     `Ternary Plots <../examples/plotting/ternary.html>`__
#
#   Tutorials:
#     `Ternary Density Plots <../tutorials/ternary_density.html>`__,
#     `Making the Logo <../tutorials/logo.html>`__
#
#   Modules and Functions:
#     :mod:`pyrolite.comp.codata`,
#     :func:`~pyrolite.comp.pyrocomp.renormalise`
