"""
Submodule with various plotting and visualisation functions.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", "Unknown section")

from ..util.plot import (
    plot_cooccurence,
    init_axes,
    label_axes,
    linekwargs,
    scatterkwargs,
)
from ..util.pd import to_frame
from ..util.meta import get_additional_params, subkwargs
from ..geochem import common_elements, REE
from . import density
from . import spider
from . import stem
from . import parallel
from .color import process_color

from ..comp.codata import close, ilr
from ..util.distributions import sample_kde, get_scaler

# pyroplot added to __all__ for docs
__all__ = ["density", "spider", "tern", "pyroplot"]

import pandas as pd

# todo: global style variables
FONTSIZE = 12


def _check_components(obj, components=None, valid_sizes=[2, 3]):
    """
    Check that the components provided within a dataframe are consistent with the
    form of plot being used.

    Parameters
    ----------
    obj : :class:`pandas.DataFrame`
        Object to check.
    components : :class:`list`
        List of components, optionally specified.
    valid_sizes : :class:`list`
        Component list lengths which are valid for the plot type.

    Returns
    ---------
    :class:`list`
        Components for the plot.
    """
    try:
        if obj.columns.size not in valid_sizes:
            assert len(components) in valid_sizes

        if components is None:
            components = obj.columns.values
    except:
        msg = "Suggest components or provide a slice of the dataframe."
        raise AssertionError(msg)
    return components


# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyroplot")
@pd.api.extensions.register_dataframe_accessor("pyroplot")
class pyroplot(object):
    """
    Custom dataframe accessor for pyrolite plotting.

    Notes
    -----
        This accessor enables the coexistence of array-based plotting functions and
        methods for pandas objects. This enables some separation of concerns.
    """

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def cooccurence(self, ax=None, normalize=True, log=False, colorbar=False, **kwargs):
        """
        Plot the co-occurence frequency matrix for a given input.

        Parameters
        -----------
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        normalize : :class:`bool`
            Whether to normalize the cooccurence to compare disparate variables.
        log : :class:`bool`
            Whether to take the log of the cooccurence.
        colorbar : :class:`bool`
            Whether to append a colorbar.

        Returns
        --------
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

    def density(
        self,
        components: list = None,
        ax=None,
        axlabels=True,
        fontsize=FONTSIZE,
        **kwargs
    ):
        r"""
        Method for plotting histograms (mode='hist2d'|'hexbin') or kernel density
        esitimates from point data. Convenience access function to
        :func:`~pyrolite.plot.density.density` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        -----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, True
            Whether to add x-y axis labels.
        fontsize : :class:`int`
            Fontsize for axis labels.

        Other Parameters
        ------------------
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the density diagram is plotted.
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components)

        ax = density.density(
            obj.loc[:, components].astype(np.float).values, ax=ax, **kwargs
        )
        if axlabels:
            label_axes(ax, labels=components, fontsize=fontsize)

        return ax

    def heatscatter(
        self,
        components: list = None,
        ax=None,
        axlabels=True,
        logx=False,
        logy=False,
        fontsize=FONTSIZE,
        **kwargs
    ):
        r"""
        Heatmapped scatter plots using the pyroplot API. See further parameters
        for `matplotlib.pyplot.scatter` function below.

        Parameters
        -----------
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
        fontsize : :class:`int`
            Fontsize for axis labels.

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the heatmapped scatterplot is added.
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components)
        data, samples = obj.loc[:, components].values, obj.loc[:, components].values
        kdetfm = [  # log transforms
            get_scaler([None, np.log][logx], [None, np.log][logy]),
            lambda x: ilr(close(x)),
        ][len(components) == 3]
        zi = sample_kde(
            data, samples, transform=kdetfm, **subkwargs(kwargs, sample_kde)
        )
        kwargs.update({"c": zi})
        ax = obj.loc[:, components].pyroplot.scatter(
            ax=ax,
            axlabels=axlabels,
            fontsize=fontsize,
            **scatterkwargs(process_color(**kwargs)),
        )
        return ax

    def parallel(
        self,
        columns=None,
        rescale=False,
        color_by=None,
        legend=False,
        cmap=plt.cm.viridis,
        ax=None,
        **kwargs
    ):

        """
        Create a :func:`pyrolite.plot.parallel.parallel`. coordinate plot from
        the columns of the :class:`~pandas.DataFrame`.

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the parallel coordinates plot is added.

        Todo
        ------
        * Adapt figure size based on number of columns.
        """

        obj = to_frame(self._obj)
        ax = parallel.parallel(
            obj,
            columns=columns,
            rescale=rescale,
            color_by=color_by,
            legend=legend,
            cmap=cmap,
            ax=ax,
            **kwargs
        )
        return ax

    def plot(
        self,
        components: list = None,
        ax=None,
        axlabels=True,
        fontsize=FONTSIZE,
        **kwargs,
    ):
        r"""
        Convenience method for line plots using the pyroplot API. See
        further parameters for `matplotlib.pyplot.scatter` function below.

        Parameters
        -----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, :code:`True`
            Whether to add x-y axis labels.
        fontsize : :class:`int`
            Fontsize for axis labels.
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
        lines = ax.plot(*obj.loc[:, components].values.T, **kw)
        # if color is multi, could update line colors here
        if axlabels:
            label_axes(ax, labels=components, fontsize=fontsize)

        ax.tick_params("both", labelsize=fontsize * 0.9)
        # ax.grid()
        # ax.set_aspect("equal")
        return ax

    def REE(self, index="elements", ax=None, mode="plot", **kwargs):
        """Pass the pandas object to :func:`pyrolite.plot.spider.REE_v_radii`.

        Parameters
        ------------
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        index : :class:`str`
            Whether to plot radii ('radii') on the principal x-axis, or elements
            ('elements').
        mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the REE plot is added.
        """
        obj = to_frame(self._obj)
        ree = REE()

        ax = spider.REE_v_radii(
            obj.loc[:, ree].astype(np.float).values,
            index=index,
            ree=ree,
            mode=mode,
            ax=ax,
            **process_color(**kwargs),
        )
        ax.set_ylabel(" $\mathrm{X / X_{Reference}}$")
        return ax

    def scatter(
        self,
        components: list = None,
        ax=None,
        axlabels=True,
        fontsize=FONTSIZE,
        **kwargs
    ):
        r"""
        Convenience method for scatter plots using the pyroplot API. See
        further parameters for `matplotlib.pyplot.scatter` function below.

        Parameters
        -----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, :code:`True`
            Whether to add x-y axis labels.
        fontsize : :class:`int`
            Fontsize for axis labels.
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
        sc = ax.scatter(
            *obj.loc[:, components].values.T, **scatterkwargs(process_color(**kwargs))
        )

        if axlabels:
            label_axes(ax, labels=components, fontsize=fontsize)

        ax.tick_params("both", labelsize=fontsize * 0.9)
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
        fontsize=FONTSIZE,
        **kwargs,
    ):
        r"""
        Method for spider plots. Convenience access function to
        :func:`~pyrolite.plot.spider.spider` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        -----------
        components : :class:`list`, `None`
            Elements or compositional components to plot.
        indexes :  :class:`list`, `None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        index_order
            Function to order spider plot indexes (e.g. by incompatibility).
        mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.
        fontsize : :class:`int`
            Fontsize for axis labels.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the spider diagram is plotted.

        Todo
        -----
            * Add 'compositional data' filter for default components if None is given
        """
        obj = to_frame(self._obj)

        if components is None:  # default to plotting elemental data
            components = [el for el in obj.columns if el in common_elements()]

        assert len(components) != 0

        if index_order is not None:
            components = index_order(components)

        ax = spider.spider(
            obj.loc[:, components].astype(np.float).values,
            indexes=indexes,
            ax=ax,
            mode=mode,
            **process_color(**kwargs),
        )
        ax.set_xticklabels(components, rotation=60)
        return ax

    def stem(
        self,
        components: list = None,
        ax=None,
        orientation="horizontal",
        axlabels=True,
        fontsize=FONTSIZE,
        **kwargs
    ):
        r"""
        Method for creating stem plots. Convenience access function to
        :func:`~pyrolite.plot.stem.stem` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.

        Parameters
        -----------
        components : :class:`list`, :code:`None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        orientation : :class:`str`
            Orientation of the plot (horizontal or vertical).
        axlabels : :class:`bool`, True
            Whether to add x-y axis labels.
        fontsize : :class:`int`
            Fontsize for axis labels.
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the stem diagram is plotted.
        """
        obj = to_frame(self._obj)
        components = _check_components(obj, components=components, valid_sizes=[2])

        ax = stem.stem(
            *obj.loc[:, components].values.T,
            ax=ax,
            orientation=orientation,
            **process_color(**kwargs),
        )

        if axlabels:
            if "h" not in orientation.lower():
                components = components[::-1]
            label_axes(ax, labels=components, fontsize=fontsize)

        return ax


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
