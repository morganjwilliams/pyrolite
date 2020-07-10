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

ax.legend(bbox_to_anchor=(1, 1))
########################################################################################
# References
# ~~~~~~~~~~~
#
# .. [Aitchison1984] Box, G.E.P. (1976). Science and Statistics.
#     Journal of the American Statistical Association 71, 791–799.
#     `doi: 10.1080/01621459.1976.10480949 <https://doi.org/10.1080/01621459.1976.10480949>`__
#
# .. seealso::
#
#   Examples:
#     `Log Ratio Means <../examples/logratiomeans.html>`__,
#     `Log Transforms <../examples/logtransforms.html>`__,
#     `Compositional Data <../examples/compositional_data.html>`__,
#     `Ternary Plots <../examples/plotting/ternary.html>`__
#
#   Tutorials:
#     `Ternary Density Plots <../tutorials/ternary_density.html>`__,
#     `Making the Logo <../tutorials/logo.html>`__
#
#   Modules and Functions:
#     :mod:`pyrolite.comp.codata`,
#     :func:`~pyrolite.comp.pyrocomp.renormalise`
