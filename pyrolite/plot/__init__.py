"""
Submodule with various plotting and visualisation functions.
"""
import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from ..util.plot import plot_cooccurence
from ..util.pd import to_frame
from ..util.meta import get_additional_params, subkwargs
from ..geochem import common_elements, REE
from . import density
from . import spider
from . import tern
from . import stem
from . import parallel

from ..comp.codata import close, ilr
from ..util.distributions import sample_kde, get_scaler

# pyroplot added to __all__ for docs
__all__ = ["density", "spider", "tern", "pyroplot"]

import pandas as pd


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

    def density(self, components: list = None, ax=None, axlabels=True, **kwargs):
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

        Other Parameters
        ------------------
        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the density diagram is plotted.
        """
        obj = to_frame(self._obj)
        try:
            if obj.columns.size not in [2, 3]:
                assert len(components) in [2, 3]

            if components is None:
                components = obj.columns.values
        except:
            msg = "Suggest components or provide a slice of the dataframe."
            raise AssertionError(msg)

        fontsize = kwargs.get("fontsize", 8.0)
        ax = density.density(
            obj.loc[:, components].astype(np.float).values, ax=ax, **kwargs
        )
        if axlabels and len(components) == 2:
            ax.set_xlabel(components[0], fontsize=fontsize)
            ax.set_ylabel(components[1], fontsize=fontsize)

            ax.tick_params("both", labelsize=fontsize * 0.9)
        elif axlabels and len(components) == 3:
            tax = ax.tax
            # python-ternary uses "right, top, left"
            # Check if there's already labels
            offset = kwargs.get("offset", 0.2)  # offset axes labels
            if not len(tax._labels.keys()):
                tax.right_axis_label(components[0], fontsize=fontsize, offset=offset)
                tax.left_axis_label(components[1], fontsize=fontsize, offset=offset)
                tax.bottom_axis_label(components[2], fontsize=fontsize, offset=offset)
        else:
            pass

        return ax

    def heatscatter(
        self,
        components: list = None,
        ax=None,
        axlabels=True,
        logx=False,
        logy=False,
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

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the heatmapped scatterplot is added.
        """
        obj = to_frame(self._obj)
        try:
            if obj.columns.size not in [2, 3]:
                assert len(components) in [2, 3]

            if components is None:
                components = obj.columns.values
        except:
            msg = "Suggest components or provide a slice of the dataframe."
            raise AssertionError(msg)

        data, samples = obj.loc[:, components].values, obj.loc[:, components].values
        kdetfm = [  # log transforms
            get_scaler([None, np.log][logx], [None, np.log][logy]),
            lambda x: ilr(close(x)),
        ][len(components) == 3]
        zi = sample_kde(
            data, samples, transform=kdetfm, **subkwargs(kwargs, sample_kde)
        )
        ax = obj.loc[:, components].pyroplot.scatter(
            ax=ax, axlabels=axlabels, c=zi, **kwargs
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
            **kwargs
        )
        ax.set_ylabel(" $\mathrm{X / X_{Reference}}$")
        return ax

    def scatter(self, components: list = None, ax=None, axlabels=True, **kwargs):
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

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the scatterplot is added.
        """
        obj = to_frame(self._obj)
        try:
            if obj.columns.size not in [2, 3]:
                assert len(components) in [2, 3]

            if components is None:
                components = obj.columns.values
        except:
            msg = "Suggest components or provide a slice of the dataframe."
            raise AssertionError(msg)

        if ax is None:
            fig, ax = plt.subplots(1, **subkwargs(kwargs, plt.subplots))

        fontsize = kwargs.get("fontsize", 8.0)

        if len(components) == 3:
            ax = obj.loc[:, components].pyroplot.ternary(
                ax=ax, axlabels=axlabels, **kwargs
            )
        else:  # len(components) == 2
            xvar, yvar = components

            sc = ax.scatter(
                obj.loc[:, xvar].values,
                obj.loc[:, yvar].values,
                **subkwargs(kwargs, ax.scatter)
            )
            if axlabels:
                ax.set_xlabel(xvar, fontsize=fontsize)
                ax.set_ylabel(yvar, fontsize=fontsize)

            ax.tick_params("both", labelsize=fontsize * 0.9)
        return ax

    def spider(
        self,
        components: list = None,
        indexes: list = None,
        ax=None,
        mode="plot",
        **kwargs
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
        mode : :class:`str`, :code`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
            Mode for plot. Plot will produce a line-scatter diagram. Fill will return
            a filled range. Density will return a conditional density diagram.

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

        ax = spider.spider(
            obj.loc[:, components].astype(np.float).values,
            indexes=indexes,
            ax=ax,
            mode=mode,
            **kwargs
        )
        ax.set_xticklabels(components, rotation=60)
        return ax

    def stem(
        self,
        components: list = None,
        ax=None,
        orientation="horizontal",
        axlabels=True,
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

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the stem diagram is plotted.
        """
        obj = to_frame(self._obj)
        try:
            if obj.columns.size not in [2, 3]:
                assert len(components) in [2, 3]

            if components is None:
                components = obj.columns.values
        except:
            msg = "Suggest components or provide a slice of the dataframe."
            raise AssertionError(msg)
        ax = stem.stem(
            obj[components[0]].values,
            obj[components[1]].values,
            ax=ax,
            orientation=orientation,
            **kwargs
        )

        if axlabels:
            if "h" not in orientation.lower():
                components = components[::-1]
            ax.set_xlabel(components[0])
            ax.set_ylabel(components[1])

        return ax

    def ternary(self, components: list = None, ax=None, axlabels=None, **kwargs):
        r"""
        Method for ternary scatter plots. Convenience access function to
        :func:`~pyrolite.plot.tern.ternary` (see `Other Parameters`, below), where
        further parameters for relevant `matplotlib` functions are also listed.


        Parameters
        -----------
        components : :class:`list`, `None`
            Elements or compositional components to plot.
        ax : :class:`matplotlib.axes.Axes`, :code:`None`
            The subplot to draw on.
        axlabels : :class:`bool`, True
            Whether to add axis labels.

        {otherparams}

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes on which the ternary diagram is plotted.
        """
        obj = to_frame(self._obj)

        try:
            if not obj.columns.size == 3:
                assert len(components) == 3

            if components is None:
                components = obj.columns.values
        except:
            msg = "Suggest components or provide a slice of the dataframe."
            raise AssertionError(msg)

        fontsize = kwargs.get("fontsize", 10.0)
        ax = tern.ternary(
            obj.loc[:, components].astype(np.float).values, ax=ax, **kwargs
        )
        tax = ax.tax

        offset = kwargs.get("offset", 0.2)  # offset axes labels

        def set_labels(labels):  # local function to set ternary labels
            tax.right_axis_label(labels[0], fontsize=fontsize, offset=offset)
            tax.left_axis_label(labels[1], fontsize=fontsize, offset=offset)
            tax.bottom_axis_label(labels[2], fontsize=fontsize, offset=offset)

        if axlabels is not None:
            if not len(tax._labels.keys()) and axlabels:
                set_labels(components)
            elif len(tax._labels.keys()) and not axlabels:  # are labels, should be none
                set_labels([None, None, None])
            else:
                pass
        else:  # label by default
            set_labels(components)
        ax.set_aspect("equal")
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

pyroplot.ternary.__doc__ = pyroplot.ternary.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            pyroplot.ternary,
            tern.ternary,
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
