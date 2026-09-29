"""
Submodule with various plotting and visualisation functions.
"""

import warnings

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "Unknown section")

from .. import geochem
from ..comp.codata import ILR, close
from ..util.distributions import get_scaler, sample_kde
from ..util.log import Handle
from ..util.meta import subkwargs
from ..util.pd import to_frame
from ..util.plot.axes import init_axes, label_axes
from ..util.plot.helpers import plot_cooccurence
from ..util.plot.style import linekwargs, scatterkwargs
from . import density, parallel, spider, stem
from .color import process_color

logger = Handle(__name__)

# pyroplot added to __all__ for docs
__all__ = ["density", "pyroplot", "spider"]


def _check_components(
    obj: pd.DataFrame | pd.Series,
    components: list[str] | None = None,
    check_size: bool = True,
    valid_sizes: list[int] | None = None,
):
    """
    Check that the components provided within a dataframe are consistent with the
    form of plot being used.

    Parameters
    ----------
    obj : pandas.DataFrame
        Object to check.
    components : list
        List of components, optionally specified.
    check_size : bool
        Whether to verify the size of the column index.
    valid_sizes : list
        Component list lengths which are valid for the plot type.

    Returns
    -------
    list
        Components for the plot.
    """
    if valid_sizes is None:
        valid_sizes = [2, 3]
    try:
        if check_size and (obj.columns.size not in valid_sizes):
            assert len(components) in valid_sizes

        if components is None:
            components = obj.columns.values
    except AssertionError:
        msg = "Suggest components or provide a slice of the dataframe."
        raise AssertionError(msg)
    return components


# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyroplot")
@pd.api.extensions.register_dataframe_accessor("pyroplot")
class pyroplot:
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

    def cooccurence(
        self,
        ax: matplotlib.axes.Axes | None = None,
        normalize: bool = True,
        log: bool = False,
        colorbar: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot the co-occurence frequency matrix for a given input.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        normalize : bool
            Whether to normalize the cooccurence to compare disparate variables.
        log : bool
            Whether to take the log of the cooccurence.
        colorbar : bool
            Whether to append a colorbar.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the cooccurence plot is added.

        """
        obj = to_frame(self._obj)
        ax = plot_cooccurence(
            obj.values, ax=ax, normalize=normalize, log=log, colorbar=colorbar, **kwargs
        )
        ax.set_xticklabels(obj.columns, minor=False, rotation=90)
        ax.set_yticklabels(obj.columns, minor=False)
        return ax

    def density(
        self,
        components: list[str] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        axlabels: bool = True,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        r"""
        Method for plotting histograms (mode='hist2d'|'hexbin') or kernel density
        esitimates from point data. Convenience access function to
        `~pyrolite.plot.density.density`, where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        ----------
        components : list
            Elements or compositional components to plot.
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        axlabels : bool
            Whether to add x-y axis labels.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the density diagram is plotted.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.density`, :func:`pyrolite.plot.density.density`.
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components)

        ax = density.density(
            obj.reindex(columns=components).astype(float).values, ax=ax, **kwargs
        )
        if axlabels:
            label_axes(ax, labels=components)

        return ax

    def heatscatter(
        self,
        components: list[str] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        axlabels: bool = True,
        logx: bool = False,
        logy: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        r"""
        Heatmapped scatter plots using the pyroplot API. See further parameters
        for `matplotlib.pyplot.scatter` function below.

        Parameters
        ----------
        components : list
            Elements or compositional components to plot.
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        axlabels : bool
            Whether to add x-y axis labels.
        logx : bool
            Whether to log-transform x values before the KDE for bivariate plots.
        logy : bool
            Whether to log-transform y values before the KDE for bivariate plots.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the heatmapped scatterplot is added.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.scatter`.
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components)
        data, samples = (
            obj.reindex(columns=components).values,
            obj.reindex(columns=components).values,
        )
        kdetfm = [  # log transforms
            get_scaler([None, np.log][logx], [None, np.log][logy]),
            lambda x: ILR(close(x)),
        ][len(components) == 3]
        zi = sample_kde(
            data, samples, transform=kdetfm, **subkwargs(kwargs, sample_kde)
        )
        kwargs.update({"c": zi})
        ax = obj.reindex(columns=components).pyroplot.scatter(
            ax=ax, axlabels=axlabels, **kwargs
        )
        return ax

    def parallel(
        self,
        components: list[str] | None = None,
        rescale: bool = False,
        legend: bool = False,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Create a `pyrolite.plot.parallel.parallel`. coordinate plot from
        the columns of the pandas.DataFrame.

        Parameters
        ----------
        components : list
            Components to use as axes for the plot.
        rescale : bool
            Whether to rescale values to [-1, 1].
        legend : bool
            Whether to include or suppress the legend.
        ax : matplotlib.axes.Axes
            The subplot to draw on.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the parallel coordinates plot is added.

        Todo
        ----
        * Adapt figure size based on number of columns.

        Notes
        -----
        See also: `pyrolite.plot.pyroplot.parallel`, `pyrolite.plot.parallel.parallel`.
        """

        obj = to_frame(self._obj)
        ax = parallel.parallel(
            obj,
            components=components,
            rescale=rescale,
            legend=legend,
            ax=ax,
            **kwargs,
        )
        return ax

    def plot(
        self,
        components: list[str] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        axlabels: bool = True,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        r"""
        Convenience method for line plots using the pyroplot API. See
        further parameters for `matplotlib.pyplot.scatter` function below.

        Parameters
        ----------
        components : list
            Elements or compositional components to plot.
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        axlabels : bool
            Whether to add x-y axis labels.
        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the plot is added.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.plot`, :func:`matplotlib.pyplot.plot`
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components)
        projection = [None, "ternary"][len(components) == 3]
        ax = init_axes(ax=ax, projection=projection, **kwargs)
        kw = linekwargs(kwargs)
        ax.plot(*obj.reindex(columns=components).values.T, **kw)
        # if color is multi, could update line colors here
        if axlabels:
            label_axes(ax, labels=components)

        ax.tick_params("both")
        # ax.grid()
        # ax.set_aspect("equal")
        return ax

    def REE(
        self,
        index: str = "elements",
        ax: matplotlib.axes.Axes | None = None,
        mode: str = "plot",
        dropPm: bool = True,
        scatter_kw: dict | None = None,
        line_kw: dict | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Pass the pandas object to :func:`pyrolite.plot.spider.REE_v_radii`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        index : str
            Whether to plot radii ('radii') on the principal x-axis, or elements
            ('elements').
        mode : str
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.
        dropPm : bool
            Whether to exclude the (almost) non-existent element Promethium from the REE
            list.
        scatter_kw : dict
            Keyword parameters to be passed to the scatter plotting function.
        line_kw : dict
            Keyword parameters to be passed to the line plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the REE plot is added.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.REE`, :func:`pyrolite.plot.spider.REE_v_radii`.
        """
        if scatter_kw is None:
            scatter_kw = {}
        if line_kw is None:
            line_kw = {}
        obj = to_frame(self._obj)
        ree = [i for i in geochem.ind.REE(dropPm=dropPm) if i in obj.columns]

        ax = spider.REE_v_radii(
            obj.reindex(columns=ree).astype(float).values,
            index=index,
            ree=ree,
            mode=mode,
            ax=ax,
            scatter_kw=scatter_kw,
            line_kw=line_kw,
            **kwargs,
        )
        ax.set_ylabel(r"$\mathrm{X / X_{Reference}}$")
        return ax

    def scatter(
        self,
        components: list[str] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        axlabels: bool = True,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        r"""
        Convenience method for scatter plots using the pyroplot API. See
        further parameters for `matplotlib.pyplot.scatter` function below.

        Parameters
        ----------
        components : list
            Elements or compositional components to plot.
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        axlabels : bool
            Whether to add x-y axis labels.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the scatterplot is added.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.scatter`, :func:`matplotlib.pyplot.scatter`
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components)

        projection = [None, "ternary"][len(components) == 3]
        ax = init_axes(ax=ax, projection=projection, **kwargs)
        size = obj.index.size
        kw = process_color(size=size, **kwargs)
        with warnings.catch_warnings():
            # ternary transform where points add to zero will give an unnecessary
            # warning; here we supress it
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide"
            )
            ax.scatter(*obj.reindex(columns=components).values.T, **scatterkwargs(kw))

        if axlabels:
            label_axes(ax, labels=components)

        ax.tick_params("both")
        # ax.grid()
        # ax.set_aspect("equal")
        return ax

    def spider(
        self,
        components: list[str] | None = None,
        indexes: list[float] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        mode: str = "plot",
        index_order: str | list[str] | None = None,
        autoscale: bool = True,
        scatter_kw: dict | None = None,
        line_kw: dict | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        r"""
        Method for spider plots. Convenience access function to
        :func"`~pyrolite.plot.spider.spider`, where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        ----------
        components : list
            Elements or compositional components to plot.
        indexes :  list
            Positions of the elements along the xaxis.
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        index_order
            Function to order spider plot indexes (e.g. by incompatibility).
        autoscale : bool
            Whether to autoscale the y-axis limits for standard spider plots.
        mode : str
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.
        scatter_kw : dict
            Keyword parameters to be passed to the scatter plotting function.
        line_kw : dict
            Keyword parameters to be passed to the line plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the spider diagram is plotted.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.spider`, :meth:`pyrolite.plot.spider.spider`.
        """
        if line_kw is None:
            line_kw = {}
        if scatter_kw is None:
            scatter_kw = {}
        obj = to_frame(self._obj)

        if components is None:  # default to plotting elemental data
            components = [
                el for el in obj.columns if el in geochem.ind.common_elements()
            ]

        assert len(components) != 0

        if index_order is not None:
            if isinstance(index_order, str):
                try:
                    index_order = geochem.ind.ordering[index_order]
                except KeyError:
                    msg = (
                        "Ordering not applied, as parameter '{}' not recognized."
                        " Select from: {}"
                    ).format(index_order, ", ".join(list(geochem.ind.ordering.keys())))
                    logger.warning(msg)
                components = index_order(components)
            else:
                components = index_order(components)

        ax = init_axes(ax=ax, **kwargs)

        if hasattr(ax, "_pyrolite_components"):
            # TODO: handle spider diagrams which have specified components
            pass

        ax = spider.spider(
            obj.reindex(columns=components).astype(float).values,
            indexes=indexes,
            ax=ax,
            mode=mode,
            autoscale=autoscale,
            scatter_kw=scatter_kw,
            line_kw=line_kw,
            **kwargs,
        )
        ax._pyrolite_components = components
        ax.set_xticklabels(components, rotation=60)
        return ax

    def stem(
        self,
        components: list[str] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        orientation: str = "horizontal",
        axlabels: bool = True,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        r"""
        Method for creating stem plots. Convenience access function to
        :func:`~pyrolite.plot.stem.stem`, where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        ----------
        components : list
            Elements or compositional components to plot.
        ax : matplotlib.axes.Axes
            The subplot to draw on.
        orientation : str
            Orientation of the plot (horizontal or vertical).
        axlabels : bool
            Whether to add x-y axis labels.

        Returns
        -------
        matplotlib.axes.Axes
            Axes on which the stem diagram is plotted.

        Notes
        -----
        See also: :meth:`pyrolite.plot.pyroplot.stem`,:func:`pyrolite.plot.stem.stem`.
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components, valid_sizes=[2])

        ax = stem.stem(
            *obj.reindex(columns=components).values.T,
            ax=ax,
            orientation=orientation,
            **process_color(**kwargs),
        )

        if axlabels:
            if "h" not in orientation.lower():
                components = components[::-1]
            label_axes(ax, labels=components)

        return ax
