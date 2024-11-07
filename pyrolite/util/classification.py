"""
Utilities for rock chemistry and mineral abundance classification.

Todo
-------

* Petrological classifiers: QAPF (aphanitic/phaneritic),
  gabbroic Pyroxene-Olivine-Plagioclase,
  ultramafic Olivine-Orthopyroxene-Clinopyroxene
"""

import json

import matplotlib.lines
import matplotlib.patches
import matplotlib.text
import numpy as np
import pandas as pd
from matplotlib.projections import get_projection_class

from .log import Handle
from .meta import (
    pyrolite_datafolder,
    sphinx_doi_link,
    subkwargs,
    update_docstring_references,
)
from .plot.axes import init_axes
from .plot.helpers import get_centroid, get_visual_center
from .plot.style import patchkwargs
from .plot.transform import tlr_to_xy

logger = Handle(__name__)


def _read_poly(poly):
    """
    Read points from a polygon, allowing ratio values to be specified.
    """

    def get_ratio(s):
        a, b = s.split("/")
        return float(a) / float(b)

    return [
        [
            get_ratio(c) if isinstance(c, str) and c.count("/") == 1 else float(c)
            for c in pt
        ]
        for pt in poly
    ]


class PolygonClassifier(object):
    """
    A classifier model built form a series of polygons defining specific classes.

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`dict`
        Mapping from plot axes to variables to be used for labels.
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
        self,
        name=None,
        axes=None,
        fields=None,
        scale=1.0,
        transform=None,
        mode=None,
        **kwargs,
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
        self.axes = axes or {}

        # addition for multiple modes of one diagram
        # the diagram itself is assigned at instantiation time, so
        # to swap modes, another diagram would need to be created
        valid_modes = kwargs.pop("modes", None)  # should be a list of valid modes
        if mode is None:
            mode = "default"
        elif valid_modes is not None:
            if mode not in valid_modes:
                raise ValueError(
                    "{} is an invalid mode for {}. Valid modes: {}".format(
                        mode, self.__class__.__name__, ", ".join(valid_modes)
                    )
                )
        else:
            pass

        # check axes for ratios, addition/subtraction etc
        self.fields = fields or {}

        if mode in self.fields:
            self.fields = self.fields[mode]
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
                self.transform(_read_poly(self.fields[k]["poly"])), closed=True
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
            Ordered names for axes used by the classifier.
        """
        return list(self.axes.values())

    def _add_polygons_to_axes(
        self,
        ax=None,
        fill=False,
        axes_scale=100.0,
        add_labels=False,
        which_labels="ID",
        which_ids=None,
        **kwargs,
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
        which_labels : :class:`str`
            Which data to use for field labels - field 'name' or 'ID'.
        which_ids : :class:`list`
            List of field IDs corresponding to the polygons to add to the axes object.
            (e.g. for TAS, ['F', 'T1'] to plot the Foidite and Trachyte fields).
            An empty list corresponds to plotting all the polygons.

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
        poly_config = patchkwargs(kwargs)
        poly_config["edgecolor"] = kwargs.get("edgecolor", kwargs.get("color", "k"))
        poly_config["zorder"] = poly_config.get("zorder", -1)
        if not fill:
            poly_config["facecolor"] = "none"
            poly_config.pop("color", None)

        use_keys = not which_labels.lower().startswith("name")

        if which_ids is None:
            which_ids = list(self.fields.keys())

        for k, cfg in self.fields.items():
            if cfg["poly"] and (k in which_ids):
                verts = self.transform(np.array(_read_poly(cfg["poly"]))) * rescale_by
                pg = matplotlib.patches.Polygon(
                    verts,
                    closed=True,
                    transform=ax.transAxes
                    if self.projection is not None
                    else ax.transData,
                    **poly_config,
                )
                pgns.append(pg)
                ax.add_patch(pg)
                if add_labels:
                    label = k if use_keys else cfg["name"]
                    x, y = get_centroid(pg)
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
        # for the moment we're only doing this for standard projections
        # todo: automatically find the relevant lim function,
        # such that e.g. ternary limits might be able to be specified?
        if self.projection is None:
            if np.allclose(ax.get_xlim(), [0, 1]) & np.allclose(ax.get_ylim(), [0, 1]):
                if "xlim" in self.lims:
                    ax.set_xlim(np.array(self.lims["xlim"]) * rescale_by)
                if "ylim" in self.lims:
                    ax.set_ylim(np.array(self.lims["ylim"]) * rescale_by)
        return ax

    def add_to_axes(
        self,
        ax=None,
        fill=False,
        axes_scale=1.0,
        add_labels=False,
        which_labels="ID",
        which_ids=None,
        **kwargs,
    ):
        """
        Add the fields from the classifier to an axis.

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
            Which data to use for field labels - field 'name' or 'ID'.
        which_ids : :class:`list`
            List of field IDs corresponding to the polygons to add to the axes object.
            (e.g. for TAS, ['F', 'T1'] to plot the Foidite and Trachyte fields).
            An empty list corresponds to plotting all the polygons.

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        ax = init_axes(ax=ax, projection=self.projection)

        ax = self._add_polygons_to_axes(
            ax=ax,
            fill=fill,
            axes_scale=axes_scale,
            add_labels=add_labels,
            which_labels=which_labels,
            which_ids=which_ids,
            **kwargs,
        )
        if self.axes is not None:
            ax.set(**{"{}label".format(a): var for a, var in self.axes.items()})
        return ax


@update_docstring_references
class TAS(PolygonClassifier):
    """
    Total-alkali Silica Diagram classifier from Middlemost (1994) [#ref_1]_,
    a closed-polygon variant after Le Bas et al. (1992) [#ref_2]_.

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
    which_model : :class:`str`
        The name of the model variant to use, if not Middlemost.

    References
    -----------
    .. [#ref_1] Middlemost, E. A. K. (1994).
                Naming materials in the magma/igneous rock system.
                Earth-Science Reviews, 37(3), 215–224.
                doi: {Middlemost1994}
    .. [#ref_2] Le Bas, M.J., Le Maitre, R.W., Woolley, A.R. (1992).
                The construction of the Total Alkali-Silica chemical
                classification of volcanic rocks.
                Mineralogy and Petrology 46, 1–22.
                doi: {LeBas1992}
    .. [#ref_3] Le Maitre, R.W. (2002). Igneous Rocks: A Classification and Glossary
                of Terms : Recommendations of International Union of Geological
                Sciences Subcommission on the Systematics of Igneous Rocks.
                Cambridge University Press, 236pp.
                doi: {LeMaitre2002}
    """

    def __init__(self, which_model=None, **kwargs):
        if which_model == "LeMaitre":
            src = (
                pyrolite_datafolder(subfolder="models") / "TAS" / "config_lemaitre.json"
            )
        elif which_model == "LeMaitreCombined":
            src = (
                pyrolite_datafolder(subfolder="models")
                / "TAS"
                / "config_lemaitre_combined.json"
            )
        else:
            # fallback to Middlemost
            src = pyrolite_datafolder(subfolder="models") / "TAS" / "config.json"

        with open(src, "r") as f:
            config = json.load(f)
        kw = dict(scale=100.0, xlim=[30, 90], ylim=[0, 20])
        kw.update(kwargs)
        poly_config = {**config, **kw}
        super().__init__(**poly_config)

    def add_to_axes(
        self,
        ax=None,
        fill=False,
        axes_scale=100.0,
        add_labels=False,
        which_labels="ID",
        which_ids=None,
        label_at_centroid=True,
        **kwargs,
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
        which_ids : :class:`list`
            List of field IDs corresponding to the polygons to add to the axes object.
            (e.g. for TAS, ['F', 'T1'] to plot the Foidite and Trachyte fields).
            An empty list corresponds to plotting all the polygons.
        label_at_centroid : :class:`bool`
            Whether to label the fields at the centroid (True) or at the visual
            center of the field (False).

        Returns
        --------
        ax : :class:`matplotlib.axes.Axes`
        """
        # use and override the default add_to_axes
        # here we don't want to add the labels in the normal way, because there
        # are two sets - one for volcanic rocks and one for plutonic rocks
        ax = self._add_polygons_to_axes(
            ax=ax,
            fill=fill,
            axes_scale=axes_scale,
            add_labels=False,
            which_ids=which_ids,
            **kwargs,
        )

        if not label_at_centroid:
            # Calculate the effective vertical exaggeration that
            # produces nice positioning of labels. The true vertical
            # exaggeration is increased by a scale_factor because
            # the text labels are typically wider than they are long,
            # so we want to promote the labels
            # being placed at the widest part of the field.
            scale_factor = 1.5
            p = ax.transData.transform([[0.0, 0.0], [1.0, 1.0]])
            yx_scaling = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * scale_factor

        rescale_by = 1.0
        if axes_scale is not None:  # rescale polygons to fit ax
            if not np.isclose(self.default_scale, axes_scale):
                rescale_by = axes_scale / self.default_scale

        if which_ids is None:
            which_ids = list(self.fields.keys())

        if add_labels:
            for k, cfg in self.fields.items():
                if cfg["poly"] and (k in which_ids):
                    if which_labels.lower().startswith("id"):
                        label = k
                    elif which_labels.lower().startswith(
                        "volc"
                    ):  # use the volcanic name
                        label = cfg["name"][0]
                    elif which_labels.lower().startswith(
                        "intr"
                    ):  # use the intrusive name
                        label = cfg["name"][-1]
                    else:  # use the field identifier
                        raise NotImplementedError(
                            "Invalid specification for labels: {}; chose from {}".format(
                                which_labels, ", ".join(["volcanic", "intrusive", "ID"])
                            )
                        )
                    verts = np.array(_read_poly(cfg["poly"])) * rescale_by
                    _poly = matplotlib.patches.Polygon(verts)
                    if label_at_centroid:
                        x, y = get_centroid(_poly)
                    else:
                        x, y = get_visual_center(_poly, yx_scaling)
                    ax.annotate(
                        "\n".join(label.split()),
                        xy=(x, y),
                        ha="center",
                        va="center",
                        fontsize=kwargs.get("fontsize", 8),
                        **subkwargs(kwargs, ax.annotate),
                    )

        ax.set(xlabel="SiO$_2$", ylabel="Na$_2$O + K$_2$O")
        return ax


@update_docstring_references
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
    .. [#ref_1] Soil Science Division Staff (2017). Soil Survey Manual.
                C. Ditzler, K. Scheffe, and H.C. Monger (eds.).
                USDA Handbook 18. Government Printing Office, Washington, D.C.
    .. [#ref_2] Thien, Steve J. (1979). A Flow Diagram for Teaching
                Texture-by-Feel Analysis. Journal of Agronomic Education 8:54–55.
                doi: {Thien1979}
    """

    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models") / "USDASoilTexture" / "config.json"
        )

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs, "transform": "ternary"}
        super().__init__(**poly_config)


@update_docstring_references
class QAP(PolygonClassifier):
    """
    IUGS QAP ternary classification
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
    .. [#ref_1] Streckeisen, A. (1974). Classification and nomenclature of plutonic
                rocks: recommendations of the IUGS subcommission on the systematics
                of Igneous Rocks. Geol Rundsch 63, 773–786.
                doi: {Streckeisen1974}
    .. [#ref_2] Le Maitre,R.W. (2002). Igneous Rocks: A Classification and Glossary
                of Terms : Recommendations of International Union of Geological
                Sciences Subcommission on the Systematics of Igneous Rocks.
                Cambridge University Press, 236pp
                doi: {LeMaitre2002}
    """

    def __init__(self, **kwargs):
        src = pyrolite_datafolder(subfolder="models") / "QAP" / "config.json"

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs, "transform": "ternary"}
        super().__init__(**poly_config)


@update_docstring_references
class FeldsparTernary(PolygonClassifier):
    """
    Simplified feldspar diagram classifier, based on a version printed in the
    second edition of 'An Introduction to the Rock Forming Minerals' (Deer,
    Howie and Zussman).

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.
    mode : :class:`str`
        Mode of the diagram to use; two are currently available - 'default',
        which fills the entire ternary space, and 'miscibility-gap' which gives
        a simplified approximation of the miscibility gap.

    References
    -----------
    .. [#ref_1] Deer, W. A., Howie, R. A., & Zussman, J. (2013).
        An introduction to the rock-forming minerals (3rd ed.).
        Mineralogical Society of Great Britain and Ireland.
    """

    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models") / "FeldsparTernary" / "config.json"
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


@update_docstring_references
class JensenPlot(PolygonClassifier):
    """
    Jensen Plot for classification of subalkaline volcanic rocks  [#ref_1]_.

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
    .. [#ref_1] Jensen, L. S. (1976). A new cation plot for classifying
                sub-alkaline volcanic rocks.
                Ontario Division of Mines. Miscellaneous Paper No. 66.

    Notes
    -----
    Diagram used for the classification classification of subalkalic volcanic rocks.
    The diagram is constructed for molar cation percentages of Al, Fe+Ti and Mg,
    on account of these elements' stability upon metamorphism.
    This particular version uses updated labels relative to Jensen (1976),
    in which the fields have been extended to the full range of the ternary plot.
    """

    def __init__(self, **kwargs):
        src = pyrolite_datafolder(subfolder="models") / "JensenPlot" / "config.json"

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs, "transform": "ternary"}
        super().__init__(**poly_config)


class SpinelTrivalentTernary(PolygonClassifier):
    """
    Spinel Trivalent Ternary classification  - designed for data in atoms per formula unit

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.
    """

    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models")
            / "SpinelTrivalentTernary"
            / "config.json"
        )

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs, "transform": "ternary"}
        super().__init__(**poly_config)


class SpinelFeBivariate(PolygonClassifier):
    """
    Fe-Spinel classification, designed for data in atoms per formula unit.

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.
    """

    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models")
            / "SpinelFeBivariate"
            / "config.json"
        )

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs}
        super().__init__(**poly_config)


@update_docstring_references
class Pettijohn(PolygonClassifier):
    """
    Pettijohn (1973) sandstones classification
    [#ref_1]_.

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
    .. [#ref_1] Pettijohn, F. J., Potter, P. E. and Siever, R. (1973).
                Sand  and Sandstone. New York, Springer-Verlag. 618p.
                doi: {Pettijohn1973}
    """

    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models")
            / "sandstones"
            / "config_pettijohn.json"
        )

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs}
        super().__init__(**poly_config)


@update_docstring_references
class Herron(PolygonClassifier):
    """
    Herron (1988) sandstones classification
    [#ref_1]_.

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
    .. [#ref_1] Herron, M.M. (1988).
                Geochemical classification of terrigenous sands and shales
                from core or log data.
                Journal of Sedimentary Research, 58(5), pp.820-829.
                doi: {Herron1988}
    """

    def __init__(self, **kwargs):
        src = (
            pyrolite_datafolder(subfolder="models")
            / "sandstones"
            / "config_herron.json"
        )

        with open(src, "r") as f:
            config = json.load(f)

        poly_config = {**config, **kwargs}
        super().__init__(**poly_config)


TAS.__doc__ = TAS.__doc__.format(
    LeBas1992=sphinx_doi_link("10.1007/BF01160698"),
    Middlemost1994=sphinx_doi_link("10.1016/0012-8252(94)90029-9"),
    LeMaitre2002=sphinx_doi_link("10.1017/CBO9780511535581"),
)
USDASoilTexture.__doc__ = USDASoilTexture.__doc__.format(
    Thien1979=sphinx_doi_link("10.2134/jae.1979.0054")
)
QAP.__doc__ = QAP.__doc__.format(
    Streckeisen1974=sphinx_doi_link("10.1007/BF01820841"),
    LeMaitre2002=sphinx_doi_link("10.1017/CBO9780511535581"),
)
Pettijohn.__doc__ = Pettijohn.__doc__.format(
    Pettijohn1973=sphinx_doi_link("10.1007/978-1-4615-9974-6"),
)
Herron.__doc__ = Herron.__doc__.format(
    Herron1988=sphinx_doi_link("10.1306/212F8E77-2B24-11D7-8648000102C1865D"),
)
