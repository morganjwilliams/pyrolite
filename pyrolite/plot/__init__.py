"""
Submodule with various plotting and visualisation functions.
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import mpltern
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "Unknown section")

from .. import geochem
from ..comp.codata import ILR, close
from ..util.distributions import get_scaler, sample_kde
from ..util.log import Handle
from ..util.meta import get_additional_params, subkwargs
from ..util.pd import to_frame, _check_components
from ..util.plot.axes import init_axes, label_axes
from ..util.plot.helpers import plot_cooccurence
from ..util.plot.style import _export_nonRCstyles, linekwargs, scatterkwargs
from . import density, parallel, spider, stem
from .color import process_color

logger = Handle(__name__)

# pyroplot added to __all__ for docs
__all__ = ["density", "spider", "pyroplot"]


class pyroplot_matplotlib(object):
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

        # refresh custom styling on creation?
        _export_nonRCstyles()

    @staticmethod
    def _validate(obj):
        pass

    def cooccurence(self, ax=None, normalize=True, log=False, colorbar=False, **kwargs):
        """
        Plot the co-occurence frequency matrix for a given input.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        normalize : :class:`bool`
            Whether to normalize the cooccurence to compare disparate variables.
        log : :class:`bool`
            Whether to take the log of the cooccurence.
        colorbar : :class:`bool`
            Whether to append a colorbar.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the cooccurence plot is added.

        """
        obj = to_frame(self._obj)
        ax = plot_cooccurence(
            obj.values, ax=ax, normalize=normalize, log=log, colorbar=colorbar, **kwargs
        )
        ax.set_xticklabels(obj.columns, minor=False, rotation=90)
        ax.set_yticklabels(obj.columns, minor=False)
        return ax

    def density(self, components: list = None, ax=None, axlabels=True, **kwargs):
        r"""
        Method for plotting histograms (mode='hist2d'|'hexbin') or kernel density
        esitimates from point data. Convenience access function to
        :func:`~pyrolite.plot.density.density` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        ----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, True
            Whether to add x-y axis labels.

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the density diagram is plotted.

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
        components: list = None,
        ax=None,
        axlabels=True,
        logx=False,
        logy=False,
        **kwargs,
    ):
        r"""
        Heatmapped scatter plots using the pyroplot API. See further parameters
        for `matplotlib.pyplot.scatter` function below.

        Parameters
        ----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, :code:`True`
            Whether to add x-y axis labels.
        logx : :class:`bool`, `False`
            Whether to log-transform x values before the KDE for bivariate plots.
        logy : :class:`bool`, `False`
            Whether to log-transform y values before the KDE for bivariate plots.

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the heatmapped scatterplot is added.

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
        components=None,
        rescale=False,
        legend=False,
        ax=None,
        **kwargs,
    ):
        """
        Create a :func:`pyrolite.plot.parallel.parallel`. coordinate plot from
        the columns of the :class:`~pandas.DataFrame`.

        Parameters
        ----------
        components : :class:`list`, :code:`None`
            Components to use as axes for the plot.
        rescale : :class:`bool`
            Whether to rescale values to [-1, 1].
        legend : :class:`bool`, :code:`False`
            Whether to include or suppress the legend.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the parallel coordinates plot is added.

        Todo
        ----
        * Adapt figure size based on number of columns.

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

    def plot(self, components: list = None, ax=None, axlabels=True, **kwargs):
        r"""
        Convenience method for line plots using the pyroplot API. See
        further parameters for `matplotlib.pyplot.scatter` function below.

        Parameters
        ----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, :code:`True`
            Whether to add x-y axis labels.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the plot is added.

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
        index="elements",
        ax=None,
        mode="plot",
        dropPm=True,
        scatter_kw={},
        line_kw={},
        **kwargs,
    ):
        """Pass the pandas object to :func:`pyrolite.plot.spider.REE_v_radii`.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        index : :class:`str`
            Whether to plot radii ('radii') on the principal x-axis, or elements
            ('elements').
        mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.
        dropPm : :class:`bool`
            Whether to exclude the (almost) non-existent element Promethium from the REE
            list.
        scatter_kw : :class:`dict`
            Keyword parameters to be passed to the scatter plotting function.
        line_kw : :class:`dict`
            Keyword parameters to be passed to the line plotting function.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the REE plot is added.

        """
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

    def scatter(self, components: list = None, ax=None, axlabels=True, **kwargs):
        r"""
        Convenience method for scatter plots using the pyroplot API. See
        further parameters for `matplotlib.pyplot.scatter` function below.

        Parameters
        ----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, :code:`True`
            Whether to add x-y axis labels.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the scatterplot is added.

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
        components: list = None,
        indexes: list = None,
        ax=None,
        mode="plot",
        index_order=None,
        autoscale=True,
        scatter_kw={},
        line_kw={},
        **kwargs,
    ):
        r"""
        Method for spider plots. Convenience access function to
        :func:`~pyrolite.plot.spider.spider` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        ----------
        components : :class:`list`, `None`
            Elements or compositional components to plot.
        indexes :  :class:`list`, `None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        index_order
            Function to order spider plot indexes (e.g. by incompatibility).
        autoscale : :class:`bool`
            Whether to autoscale the y-axis limits for standard spider plots.
        mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.
        scatter_kw : :class:`dict`
            Keyword parameters to be passed to the scatter plotting function.
        line_kw : :class:`dict`
            Keyword parameters to be passed to the line plotting function.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the spider diagram is plotted.

        Todo
        ----
            * Add 'compositional data' filter for default components if None is given

        """
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
        components: list = None,
        ax=None,
        orientation="horizontal",
        axlabels=True,
        **kwargs,
    ):
        r"""
        Method for creating stem plots. Convenience access function to
        :func:`~pyrolite.plot.stem.stem` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        ----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        orientation : :class:`str`
            Orientation of the plot (horizontal or vertical).
        axlabels : :class:`bool`, True
            Whether to add x-y axis labels.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the stem diagram is plotted.
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


pyroplot = pyroplot_matplotlib


# note that only some of these methods will be valid for series
pd.api.extensions.register_series_accessor("pyroplot")(pyroplot)
pd.api.extensions.register_dataframe_accessor("pyroplot")(pyroplot)


# ideally we would i) check for the same params and ii) aggregate all others across
# inherited or chained functions. This simply imports the params from another docstring
_add_additional_parameters = True

pyroplot.density.__doc__ = pyroplot.density.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.density,
            density.density,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)

pyroplot.parallel.__doc__ = pyroplot.parallel.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.parallel,
            parallel.parallel,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)


pyroplot.REE.__doc__ = pyroplot.REE.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.REE,
            spider.REE_v_radii,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)


pyroplot.scatter.__doc__ = pyroplot.scatter.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.scatter,
            plt.scatter,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)

pyroplot.plot.__doc__ = pyroplot.plot.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.plot,
            plt.plot,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)

pyroplot.spider.__doc__ = pyroplot.spider.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.spider,
            spider.spider,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)


pyroplot.stem.__doc__ = pyroplot.stem.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.stem,
            stem.stem,
            header="Other Parameters",
            indent=8,
            subsections=True,
        ),
    ][_add_additional_parameters]
)

pyroplot.heatscatter.__doc__ = pyroplot.heatscatter.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.scatter, header="Other Parameters", indent=8, subsections=True
        ),
    ][_add_additional_parameters]
)
