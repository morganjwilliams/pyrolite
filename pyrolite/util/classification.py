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
from .plot.style import patchkwargs
from .plot.axes import init_axes
from .plot.helpers import get_centroid
from .meta import (
    pyrolite_datafolder,
    subkwargs,
    sphinx_doi_link,
    update_docstring_references,
)
from .log import Handle

logger = Handle(__name__)


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
        self, name=None, axes=None, fields=None, scale=1.0, xlim=None, ylim=None,
    ):
        self.default_scale = scale
        self._scale = self.default_scale
        self.xlim = xlim
        self.ylim = ylim

        self.name = name
        self.axes = axes or []
        # check axes for ratios, adition/subtraction etc
        self.fields = fields or []
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
        polys = [
            matplotlib.patches.Polygon(self.fields[k]["poly"], closed=True)
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
        X = X * rescale_by

        indexes = np.array([p.contains_points(X) for p in polys]).T
        notfound = np.logical_not(indexes.sum(axis=-1))

        outlist = list(map(lambda ix: classes[ix], np.argmax(indexes, axis=-1)))
        out.loc[:] = outlist
        out.loc[(notfound)] = "none"
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
        return self.axes.get("x"), self.axes.get("y")

    def _add_polygons_to_axes(
        self, ax=None, fill=False, axes_scale=100.0, labels=None, **kwargs
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
        labels : :class:`str`
            Which labels to add to the polygons (e.g. for TAS, 'volcanic', 'intrusive'
            or the field 'ID').

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        if ax is None:
            ax = init_axes(**kwargs)

        rescale_by = 1.0
        if axes_scale is not None:  # rescale polygons to fit ax
            if not np.isclose(self.default_scale, axes_scale):
                rescale_by = axes_scale / self.default_scale
        pgns = []
        for k, cfg in self.fields.items():
            if cfg["poly"]:
                if not fill:
                    kwargs["facecolor"] = "none"
                verts = np.array(cfg["poly"]) * rescale_by
                pg = matplotlib.patches.Polygon(
                    verts, closed=True, edgecolor="k", **patchkwargs(kwargs)
                )
                pgns.append(pg)
                ax.add_patch(pg)

        # if the axis has the default scaling, there's a good chance that it hasn't
        # been rescaled/rendered. We need to rescale to show the polygons.
        if np.allclose(ax.get_xlim(), [0, 1]) & np.allclose(ax.get_ylim(), [0, 1]):
            ax.set_xlim(np.array(self.xlim) * rescale_by)
            ax.set_ylim(np.array(self.ylim) * rescale_by)

        return ax

    def add_to_axes(self, ax=None, fill=False, axes_scale=1, **kwargs):
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

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        ax = self._add_polygons_to_axes(
            ax=ax, fill=fill, axes_scale=axes_scale, **kwargs
        )
        if self.axes:  # may be none?
            ax.set_ylabel(self.axes[0])
            ax.set_xlabel(self.axes[1])
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

    def add_to_axes(self, ax=None, fill=False, axes_scale=100.0, labels=None, **kwargs):
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
        labels : :class:`str`
            Which labels to add to the polygons (e.g. for TAS, 'volcanic', 'intrusive'
            or the field 'ID').

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        # use and override the default add_to_axes
        ax = self._add_polygons_to_axes(
            ax=ax, fill=fill, axes_scale=axes_scale, **kwargs
        )
        rescale_by = 1.0
        if axes_scale is not None:  # rescale polygons to fit ax
            if not np.isclose(self.default_scale, axes_scale):
                rescale_by = axes_scale / self.default_scale
        if labels is not None:
            for k, cfg in self.fields.items():
                if cfg["poly"]:
                    verts = np.array(cfg["poly"]) * rescale_by
                    x, y = get_centroid(matplotlib.patches.Polygon(verts))
                    if "volc" in labels:  # use the volcanic name
                        label = cfg["name"][0]
                    elif "intr" in labels:  # use the intrusive name
                        label = cfg["name"][-1]
                    else:  # use the field identifier
                        label = k
                    ax.annotate(
                        "\n".join(label.split()),
                        xy=(x, y),
                        ha="center",
                        va="center",
                        **subkwargs(kwargs, ax.annotate, matplotlib.text.Text)
                    )

        ax.set_ylabel("$Na_2O + K_2O$")
        ax.set_xlabel("$SiO_2$")
        return ax


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
