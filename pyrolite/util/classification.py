"""
Utilities for rock chemistry and mineral abundance classification.

Todo
-------

* Petrological classifiers: QAPF (aphanitic/phaneritic),
  gabbroic Pyroxene-Olivine-Plagioclase,
  ultramafic Olivine-Orthopyroxene-Clinopyroxene
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.text
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines
from ..comp.codata import close
from .plot.style import patchkwargs
from .plot.axes import init_axes
from .plot.helpers import get_centroid
from .plot.transform import affine_transform
from .meta import (
    pyrolite_datafolder,
    subkwargs,
    sphinx_doi_link,
    update_docstring_references,
)
from .log import Handle

logger = Handle(__name__)


def tlr_to_xy(tlr):
    shear = affine_transform(np.array([[1, 1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    return shear(close(np.array(tlr)[:, [2, 0, 1]])).T


class PolygonClassifier(object):
    """
    A classifier model built form a series of polygons defining specific classes.

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.
    scale : :class:`float`
        Default maximum scale for the axes. Typically 100 (wt%) or 1 (fractional).
    xlim : :class:`tuple`
        Default x-limits for this classifier for plotting.
    ylim : :class:`tuple`
        Default y-limits for this classifier for plotting.
    """

    def __init__(
        self, name=None, axes=None, fields=None, scale=1.0, transform=None, **kwargs
    ):
        self.default_scale = scale
        self._scale = self.default_scale
        self.lims = {
            k: v for (k, v) in kwargs.items() if ("lim" in k) and (len(k) == 4)
        }
        self.projection = None
        if transform is not None:
            if isinstance(transform, str):
                if transform.lower().startswith("tern"):
                    self.transform = tlr_to_xy
                    self.projection = "ternary"
                else:
                    raise NotImplementedError
            else:
                self.transform = transform
        else:
            self.transform = lambda x: x  # passthrough

        self.name = name
        self.axes = axes or []
        # check axes for ratios, adition/subtraction etc
        self.fields = fields or {}
        self.classes = list(self.fields.keys())

    def predict(self, X, data_scale=None):
        """
        Predict the classification of samples using the polygon-based classifier.

        Parameters
        -----------
        X : :class:`numpy.ndarray` | :class:`pandas.DataFrame`
            Data to classify.
        data_scale : :class:`float`
            Maximum scale for the data. Typically 100 (wt%) or 1 (fractional).

        Returns
        -------
        :class:`pandas.Series`
            Series containing classifer predictions. If a dataframe was input,
            it inherit the index.
        """
        classes = [k for (k, cfg) in self.fields.items() if cfg["poly"]]
        # transformed polys
        polys = [
            matplotlib.patches.Polygon(
                self.transform(self.fields[k]["poly"]), closed=True
            )
            for k in classes
        ]
        if isinstance(X, pd.DataFrame):
            # check whether the axes names are in the columns
            axes = self.axis_components
            idx = X.index
            X = X.loc[:, axes].values
        else:
            idx = np.arange(X.shape[0])

        out = pd.Series(index=idx, dtype="object")

        rescale_by = 1.0  # rescaling the data to fit the classifier scale
        if data_scale is not None:
            if not np.isclose(self.default_scale, data_scale):
                rescale_by = self.default_scale / data_scale

        X = self.transform(X) * rescale_by  # transformed X
        indexes = np.array([p.contains_points(X) for p in polys]).T
        notfound = np.logical_not(indexes.sum(axis=-1))
        outlist = list(map(lambda ix: classes[ix], np.argmax(indexes, axis=-1)))
        out.loc[:] = outlist
        out.loc[(notfound)] = "none"
        # for those which are none, we could check if they're on polygon boundaries
        # and assign to the closest centroid (e.g. for boundary points on axes)
        return out

    @property
    def axis_components(self):
        """
        Get the axis components used by the classifier.

        Returns
        -------
        :class:`tuple`
            Names of the x and y axes for the classifier.
        """
        return list(self.axes.values())

    def _add_polygons_to_axes(
        self, ax=None, fill=False, axes_scale=100.0, add_labels=False, **kwargs
    ):
        """
        Add the polygonal fields from the classifier to an axis.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Axis to add the polygons to.
        fill : :class:`bool`
            Whether to fill the polygons.
        axes_scale : :class:`float`
            Maximum scale for the axes. Typically 100 (for wt%) or 1 (fractional).
        add_labels : :class:`bool`
            Whether to add labels at polygon centroids.

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        if ax is None:
            ax = init_axes(projection=self.projection, **kwargs)
        else:
            if self.projection:
                if not isinstance(ax, get_projection_class(self.projection)):
                    logger.warning(
                        "Projection of axis for {} should be {}.".format(
                            self.name or self.__class.__name__, self.projection
                        )
                    )
        rescale_by = 1.0
        if axes_scale is not None:  # rescale polygons to fit ax
            if not np.isclose(self.default_scale, axes_scale):
                rescale_by = axes_scale / self.default_scale

        pgns = []

        for (k, cfg) in self.fields.items():
            if cfg["poly"]:
                if not fill:
                    kwargs["facecolor"] = "none"
                verts = self.transform(np.array(cfg["poly"])) * rescale_by
                pg = matplotlib.patches.Polygon(
                    verts,
                    closed=True,
                    edgecolor="k",
                    transform=ax.transAxes
                    if self.projection is not None
                    else ax.transData,
                    **patchkwargs(kwargs),
                )
                pgns.append(pg)
                ax.add_patch(pg)
                if add_labels:
                    x, y = get_centroid(pg)
                    label = cfg["name"]
                    ax.annotate(
                        "\n".join(label.split()),
                        xy=(x, y),
                        ha="center",
                        va="center",
                        fontsize=kwargs.get("fontsize", 8),
                        xycoords=ax.transAxes
                        if self.projection is not None
                        else ax.transData,
                        **subkwargs(kwargs, ax.annotate),
                    )

        # if the axis has the default scaling, there's a good chance that it hasn't
        # been rescaled/rendered. We need to rescale to show the polygons.
        if self.projection is None:
            if np.allclose(ax.get_xlim(), [0, 1]) & np.allclose(ax.get_ylim(), [0, 1]):
                ax.set_xlim(np.array(self.lims["xlim"]) * rescale_by)
                ax.set_ylim(np.array(self.lims["ylim"]) * rescale_by)
        return ax

    def add_to_axes(
        self, ax=None, fill=False, axes_scale=1.0, add_labels=False, **kwargs
    ):
        """
        Add the TAS fields from the classifier to an axis.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Axis to add the polygons to.
        fill : :class:`bool`
            Whether to fill the polygons.
        axes_scale : :class:`float`
            Maximum scale for the axes. Typically 100 (for wt%) or 1 (fractional).
        add_labels : :class:`bool`
            Whether to add labels for the polygons.

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        ax = self._add_polygons_to_axes(
            ax=ax, fill=fill, axes_scale=axes_scale, add_labels=add_labels, **kwargs
        )
        if self.axes:  # may be none?
            if len(self.axes) == 2 and self.projection is None:
                ax.set_ylabel(self.axes[0])
                ax.set_xlabel(self.axes[1])
            elif len(self.axes) == 3 and (self.projection == "ternary"):
                pass
            else:
                raise NotImplementedError

        ax.set(**{"{}label".format(a): var for a, var in self.axes.items()})
        return ax


class TAS(PolygonClassifier):
    """
    Total-alkali Silica Diagram classifier from Le Bas (1992) [#ref_1]_.

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.
    scale : :class:`float`
        Default maximum scale for the axes. Typically 100 (wt%) or 1 (fractional).
    xlim : :class:`tuple`
        Default x-limits for this classifier for plotting.
    ylim : :class:`tuple`
        Default y-limits for this classifier for plotting.

    References
    -----------
    .. [#ref_1] Le Bas, M.J., Le Maitre, R.W., Woolley, A.R., 1992.
                The construction of the Total Alkali-Silica chemical
                classification of volcanic rocks.
                Mineralogy and Petrology 46, 1–22.
                doi: {LeBas1992}
    """

    @update_docstring_references
    def __init__(self, **kwargs):
        src = pyrolite_datafolder(subfolder="models") / "TAS" / "config.json"

        with open(src, "r") as f:
            config = json.load(f)
        kw = dict(scale=100.0, xlim=[35, 85], ylim=[0, 20])
        kw.update(kwargs)
        poly_config = {**config, **kw}
        super().__init__(**poly_config)

    def add_to_axes(
        self,
        ax=None,
        fill=False,
        axes_scale=100.0,
        add_labels=False,
        which_labels="",
        **kwargs
    ):
        """
        Add the TAS fields from the classifier to an axis.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Axis to add the polygons to.
        fill : :class:`bool`
            Whether to fill the polygons.
        axes_scale : :class:`float`
            Maximum scale for the axes. Typically 100 (for wt%) or 1 (fractional).
        add_labels : :class:`bool`
            Whether to add labels for the polygons.
        which_labels : :class:`str`
            Which labels to add to the polygons (e.g. for TAS, 'volcanic', 'intrusive'
            or the field 'ID').

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        # use and override the default add_to_axes
        # here we don't want to add the labels in the normal way, because there
        # are two sets - one for volcanic rocks and one for plutonic rocks
        ax = self._add_polygons_to_axes(
            ax=ax, fill=fill, axes_scale=axes_scale, add_labels=False, **kwargs
        )
        rescale_by = 1.0
        if axes_scale is not None:  # rescale polygons to fit ax
            if not np.isclose(self.default_scale, axes_scale):
                rescale_by = axes_scale / self.default_scale
        if add_labels:
            for k, cfg in self.fields.items():
                if cfg["poly"]:
                    if "volc" in which_labels:  # use the volcanic name
                        label = cfg["name"][0]
                    elif "intr" in which_labels:  # use the intrusive name
                        label = cfg["name"][-1]
                    else:  # use the field identifier
                        label = k
                    verts = np.array(cfg["poly"]) * rescale_by
                    _poly = matplotlib.patches.Polygon(verts)
                    x, y = get_centroid(_poly)
                    ax.annotate(
                        "\n".join(label.split()),
                        xy=(x, y),
                        ha="center",
                        va="center",
                        **subkwargs(kwargs, ax.annotate),
                    )

        ax.set(xlabel="$SiO_2$", ylabel="$Na_2O + K_2O$")
        return ax


class USDASoilTexture(PolygonClassifier):
    """
    United States Department of Agriculture Soil Texture classification model
    [#ref_1]_ [#ref_2]_.

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.

    References
    -----------
    .. [#ref_1] Soil Science Division Staff (2017). Soil survey sand.
                C. Ditzler, K. Scheffe, and H.C. Monger (eds.).
                USDA Handbook 18. Government Printing Office, Washington, D.C.
    .. [#ref_2] Thien, Steve J. (1979). A Flow Diagram for Teaching
                Texture-by-Feel Analysis. Journal of Agronomic Education 8:54–55.
                doi: {Thien1979}
    """

    @update_docstring_references
    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models") / "USDASoilTexture" / "config.json"
        )

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs, "transform": "ternary"}
        super().__init__(**poly_config)


class PeralkalinityClassifier(object):
    def __init__(self):
        self.fields = None

    def predict(self, df: pd.DataFrame):
        TotalAlkali = df.Na2O + df.K2O
        perkalkaline_where = (df.Al2O3 < (TotalAlkali + df.CaO)) & (
            TotalAlkali > df.Al2O3
        )
        metaluminous_where = (df.Al2O3 > (TotalAlkali + df.CaO)) & (
            TotalAlkali < df.Al2O3
        )
        peraluminous_where = (df.Al2O3 < (TotalAlkali + df.CaO)) & (
            TotalAlkali < df.Al2O3
        )
        out = pd.Series(index=df.index, dtype="object")
        out.loc[peraluminous_where] = "Peraluminous"
        out.loc[metaluminous_where] = "Metaluminous"
        out.loc[perkalkaline_where] = "Peralkaline"
        return out


TAS.__init__.__doc__ = TAS.__init__.__doc__.format(
    LeBas1992=sphinx_doi_link("10.1007/BF01160698")
)
USDASoilTexture.__init__.__doc__ = USDASoilTexture.__init__.__doc__.format(
    Thien1979=sphinx_doi_link("10.2134/jae.1979.0054")
)
