"""
Plotly backend for a pandas-based pyrolite plot accessor.

Todo
-----
* Make margins smaller
* Enable passing labels to markers
* Make plot variant for density plots
"""

import warnings

import matplotlib.colors
import numpy as np
import plotly.graph_objects as go

from ... import geochem
from ...comp.codata import ILR, close
from ...plot.color import process_color
from ..distributions import get_scaler, sample_kde
from ..log import Handle
from ..meta import get_additional_params, subkwargs
from ..pd import _check_components, to_frame

logger = Handle(__name__)


def to_plotly_color(color, alpha=1):
    # note that alpha isn't 255 scaled
    return "rgba" + str(
        tuple([int(i * 255) for i in matplotlib.colors.to_rgb(color)] + [alpha])
    )


class pyroplot_plotly(object):
    def __init__(self, obj):
        """
        Custom dataframe accessor for pyrolite plotting.

        Notes
        -----
            This accessor enables the coexistence of array-based plotting functions and
            methods for pandas objects. This enables some separation of concerns.
        """
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def scatter(self, color="black", alpha=1, **kwargs):
        if self._obj.columns.size == 3:
            return self._ternary(color=color, alpha=alpha, **kwargs)
        else:
            layout = dict(width=600, plot_bgcolor="white")
            fig = go.Figure()
            marker = dict(color=to_plotly_color(color, alpha=alpha))
            traces = [
                go.Scatter(
                    x=self._obj.iloc[:, 0],
                    y=self._obj.iloc[:, 1],
                    mode="markers",
                    marker=marker,
                    showlegend=False,
                    text=self._obj.index.map("Sample {}".format),
                )
            ]
            fig.add_traces(traces)
            fig.update_layout(layout)
            fig.update_xaxes(
                linecolor="black", mirror=True, title=self._obj.columns[0]
            )  # todo: add this to layout
            fig.update_yaxes(
                linecolor="black", mirror=True, title=self._obj.columns[1]
            )  # todo: add this to layout
            return fig

    def _ternary(self, color="black", alpha=1, **kwargs):
        layout = dict(
            width=600,
            plot_bgcolor="white",
            ternary={
                **{
                    a: {
                        "title": c,
                        "showgrid": False,
                        "linecolor": "black",
                    }
                    for a, c in zip(["aaxis", "baxis", "caxis"], self._obj.columns)
                },
                "bgcolor": "white",
            },
        )
        layout.update(kwargs)
        marker = {"color": to_plotly_color(color, alpha=alpha)}
        data = {
            "mode": "markers",
            **dict(zip("abc", [self._obj[c] for c in self._obj.columns])),
            "text": self._obj.index.values,
            "marker": marker,
        }
        fig = go.Figure(go.Scatterternary(data))

        fig.update_layout(layout)
        return fig

    def spider(self, color="black", unity_line=True, alpha=1, text=None, **kwargs):
        layout = dict(width=600, plot_bgcolor="white")
        fig = go.Figure()
        line = dict(color=to_plotly_color(color, alpha=alpha))
        # hovertemplate = "%{text}<br><extra></extra>" if (text is not None) else None
        traces = [
            go.Scatter(
                x=self._obj.columns,
                y=row,
                mode="lines+markers",
                line=line,
                showlegend=False,
                hoverinfo="text",
                # hovertemplate =hovertemplate if (text is not None) else None,
                text=None if text is None else text[idx],
                name="Sample {}".format(idx),
            )
            for idx, row in self._obj.iterrows()
        ]
        if unity_line:
            traces += [
                go.Scatter(
                    x=self._obj.columns,
                    y=np.ones(self._obj.columns.size),
                    mode="lines",
                    showlegend=False,
                    name=None,
                    line={"color": "black", "dash": "dot", "width": 0.5},
                )
            ]
        fig.add_traces(traces)
        fig.update_layout(**layout)
        fig.update_yaxes(
            type="log", linecolor="black", mirror=True
        )  # todo: add this to layout
        fig.update_xaxes(linecolor="black", mirror=True)  # todo: add this to layout
        return fig


# class pyroplot_plotly(object):
#     def __init__(self, obj):
#         """
#         Custom dataframe accessor for pyrolite plotting.

#         Notes
#         -----
#             This accessor enables the coexistence of array-based plotting functions and
#             methods for pandas objects. This enables some separation of concerns.
#         """
#         self._validate(obj)
#         self._obj = obj

#     @staticmethod
#     def _validate(obj):
#         pass

#     def heatscatter(
#         self,
#         components: list = None,
#         ax=None,
#         axlabels=True,
#         logx=False,
#         logy=False,
#         **kwargs,
#     ):
#         r"""
#         Heatmapped scatter plots using the pyroplot API. See further parameters
#         for `matplotlib.pyplot.scatter` function below.

#         Parameters
#         ----------
#         components : :class:`list`, :code:`None`
#             Elements or compositional components to plot.
#         ax : :class:`matplotlib.axes.Axes`, :code:`None`
#             The subplot to draw on.
#         axlabels : :class:`bool`, :code:`True`
#             Whether to add x-y axis labels.
#         logx : :class:`bool`, `False`
#             Whether to log-transform x values before the KDE for bivariate plots.
#         logy : :class:`bool`, `False`
#             Whether to log-transform y values before the KDE for bivariate plots.

#         {otherparams}

#         Returns
#         -------
#         :class:`matplotlib.axes.Axes`
#             Axes on which the heatmapped scatterplot is added.

#         """
#         obj = to_frame(self._obj)
#         components = _check_components(obj, components=components)
#         data, samples = (
#             obj.reindex(columns=components).values,
#             obj.reindex(columns=components).values,
#         )
#         kdetfm = [  # log transforms
#             get_scaler([None, np.log][logx], [None, np.log][logy]),
#             lambda x: ILR(close(x)),
#         ][len(components) == 3]
#         zi = sample_kde(
#             data, samples, transform=kdetfm, **subkwargs(kwargs, sample_kde)
#         )
#         kwargs.update({"c": zi})
#         ax = obj.reindex(columns=components).pyroplot.scatter(
#             ax=ax, axlabels=axlabels, **kwargs
#         )
#         return ax

#     def plot(self, components: list = None, ax=None, axlabels=True, **kwargs):
#         r"""
#         Convenience method for line plots using the pyroplot API. See
#         further parameters for `matplotlib.pyplot.scatter` function below.

#         Parameters
#         ----------
#         components : :class:`list`, :code:`None`
#             Elements or compositional components to plot.
#         ax : :class:`matplotlib.axes.Axes`, :code:`None`
#             The subplot to draw on.
#         axlabels : :class:`bool`, :code:`True`
#             Whether to add x-y axis labels.
#         {otherparams}

#         Returns
#         -------
#         :class:`matplotlib.axes.Axes`
#             Axes on which the plot is added.

#         """
#         obj = to_frame(self._obj)
#         components = _check_components(obj, components=components)
#         projection = [None, "ternary"][len(components) == 3]
#         # ax = init_axes(ax=ax, projection=projection, **kwargs)
#         # kw = linekwargs(kwargs)
#         ax.plot(*obj.reindex(columns=components).values.T, **kw)
#         # if color is multi, could update line colors here
#         # if axlabels:
#         #    label_axes(ax, labels=components)

#         ax.tick_params("both")
#         # ax.grid()
#         # ax.set_aspect("equal")
#         return ax

#     def REE(
#         self,
#         index="elements",
#         ax=None,
#         mode="plot",
#         dropPm=True,
#         scatter_kw={},
#         line_kw={},
#         **kwargs,
#     ):
#         """Pass the pandas object to :func:`pyrolite.plot.spider.REE_v_radii`.

#         Parameters
#         ----------
#         ax : :class:`matplotlib.axes.Axes`, :code:`None`
#             The subplot to draw on.
#         index : :class:`str`
#             Whether to plot radii ('radii') on the principal x-axis, or elements
#             ('elements').
#         mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
#             Mode for plot. Plot will produce a line-scatter diagram. Fill will return
#             a filled range. Density will return a conditional density diagram.
#         dropPm : :class:`bool`
#             Whether to exclude the (almost) non-existent element Promethium from the REE
#             list.
#         scatter_kw : :class:`dict`
#             Keyword parameters to be passed to the scatter plotting function.
#         line_kw : :class:`dict`
#             Keyword parameters to be passed to the line plotting function.
#         {otherparams}

#         Returns
#         -------
#         :class:`matplotlib.axes.Axes`
#             Axes on which the REE plot is added.

#         """
#         obj = to_frame(self._obj)
#         ree = [i for i in geochem.ind.REE(dropPm=dropPm) if i in obj.columns]

#         ax = spider.REE_v_radii(
#             obj.reindex(columns=ree).astype(float).values,
#             index=index,
#             ree=ree,
#             mode=mode,
#             ax=ax,
#             scatter_kw=scatter_kw,
#             line_kw=line_kw,
#             **kwargs,
#         )
#         ax.set_ylabel(r"$\mathrm{X / X_{Reference}}$")
#         return ax

#     def scatter(self, components: list = None, ax=None, axlabels=True, **kwargs):
#         r"""
#         Convenience method for scatter plots using the pyroplot API. See
#         further parameters for `matplotlib.pyplot.scatter` function below.

#         Parameters
#         ----------
#         components : :class:`list`, :code:`None`
#             Elements or compositional components to plot.
#         ax : :class:`matplotlib.axes.Axes`, :code:`None`
#             The subplot to draw on.
#         axlabels : :class:`bool`, :code:`True`
#             Whether to add x-y axis labels.
#         {otherparams}

#         Returns
#         -------
#         :class:`matplotlib.axes.Axes`
#             Axes on which the scatterplot is added.

#         """
#         obj = to_frame(self._obj)
#         components = _check_components(obj, components=components)

#         projection = [None, "ternary"][len(components) == 3]
#         # ax = init_axes(ax=ax, projection=projection, **kwargs)
#         size = obj.index.size
#         kw = process_color(size=size, **kwargs)
#         with warnings.catch_warnings():
#             # ternary transform where points add to zero will give an unnecessary
#             # warning; here we supress it
#             warnings.filterwarnings(
#                 "ignore", message="invalid value encountered in divide"
#             )
#             ax.scatter(*obj.reindex(columns=components).values.T, **kw)

#         # if axlabels:
#         #    label_axes(ax, labels=components)

#         ax.tick_params("both")
#         # ax.grid()
#         # ax.set_aspect("equal")
#         return ax

#     def spider(
#         self,
#         components: list = None,
#         indexes: list = None,
#         ax=None,
#         mode="plot",
#         index_order=None,
#         autoscale=True,
#         scatter_kw={},
#         line_kw={},
#         **kwargs,
#     ):
#         r"""
#         Method for spider plots. Convenience access function to
#         :func:`~pyrolite.plot.spider.spider` (see `Other Parameters`, below), where
#         further parameters for relevant `matplotlib` functions are also listed.

#         Parameters
#         ----------
#         components : :class:`list`, `None`
#             Elements or compositional components to plot.
#         indexes :  :class:`list`, `None`
#             Elements or compositional components to plot.
#         ax : :class:`matplotlib.axes.Axes`, :code:`None`
#             The subplot to draw on.
#         index_order
#             Function to order spider plot indexes (e.g. by incompatibility).
#         autoscale : :class:`bool`
#             Whether to autoscale the y-axis limits for standard spider plots.
#         mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
#             Mode for plot. Plot will produce a line-scatter diagram. Fill will return
#             a filled range. Density will return a conditional density diagram.
#         scatter_kw : :class:`dict`
#             Keyword parameters to be passed to the scatter plotting function.
#         line_kw : :class:`dict`
#             Keyword parameters to be passed to the line plotting function.
#         {otherparams}

#         Returns
#         -------
#         :class:`matplotlib.axes.Axes`
#             Axes on which the spider diagram is plotted.

#         Todo
#         ----
#             * Add 'compositional data' filter for default components if None is given

#         """
#         obj = to_frame(self._obj)

#         if components is None:  # default to plotting elemental data
#             components = [
#                 el for el in obj.columns if el in geochem.ind.common_elements()
#             ]

#         assert len(components) != 0

#         if index_order is not None:
#             if isinstance(index_order, str):
#                 try:
#                     index_order = geochem.ind.ordering[index_order]
#                 except KeyError:
#                     msg = (
#                         "Ordering not applied, as parameter '{}' not recognized."
#                         " Select from: {}"
#                     ).format(index_order, ", ".join(list(geochem.ind.ordering.keys())))
#                     logger.warning(msg)
#                 components = index_order(components)
#             else:
#                 components = index_order(components)

#         # ax = init_axes(ax=ax, **kwargs)

#         if hasattr(ax, "_pyrolite_components"):
#             # TODO: handle spider diagrams which have specified components
#             pass

#         ax = spider.spider(
#             obj.reindex(columns=components).astype(float).values,
#             indexes=indexes,
#             ax=ax,
#             mode=mode,
#             autoscale=autoscale,
#             scatter_kw=scatter_kw,
#             line_kw=line_kw,
#             **kwargs,
#         )
#         ax._pyrolite_components = components
#         ax.set_xticklabels(components, rotation=60)
#         return ax
