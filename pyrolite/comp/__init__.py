"""
Submodule for working with compositional data.
"""

from sympy.liealgebras.type_e import TypeE

from matplotlib.pylab import isin

import functools
import inspect
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..util.log import Handle
from . import codata

logger = Handle(__name__)


def attribute_transform(f: Callable, *args, **kwargs) -> Callable:
    """
    Decorator to add transform function as a dataframe attribute after
    transformation, for traceability.

    Parameters
    -----------
    f : Callable
        Transform function.

    Returns
    -------
    Callable
        Object with modified docstring.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        output = f(*args, **kwargs)
        output.attrs["transform"] = f.__name__
        return output

    wrapper.__signature__ = inspect.signature(f)
    wrapper.__doc__ = f.__doc__  # keep the docstring!
    return wrapper


# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrocomp")
@pd.api.extensions.register_dataframe_accessor("pyrocomp")
class pyrocomp:
    def __init__(self, obj: pd.DataFrame | pd.Series):
        """
        Custom dataframe accessor for pyrolite compositional transforms.
        """
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def renormalise(
        self, components: list[str] | None = None, scale: float = 100.0
    ) -> pd.DataFrame | pd.Series:
        """
        Renormalises compositional data to ensure closure.

        Parameters
        ----------
        components : list
            Option subcompositon to renormalise to 100. Useful for the use case
            where compostional data and non-compositional data are stored in the
            same dataframe.
        scale : float
            Closure parameter. Typically either 100 or 1.

        Returns
        -------
        pandas.DataFrame
            Renormalized dataframe.

        Notes
        ------
        This won't modify the dataframe in place, you'll need to assign it to something.
        If you specify components, those components will be summed to 100%,
        and others remain unchanged.
        """
        if components is None:
            components = []
        obj = self._obj
        return codata.renormalise(obj, components=components, scale=scale)

    @attribute_transform
    def ALR(
        self,
        components: list[str] | None = None,
        ind: int | str = -1,
        null_col: bool = False,
        label_mode: str = "simple",
    ) -> pd.DataFrame | pd.Series:
        """
        Additive Log Ratio transformation.

        Parameters
        ----------
        ind: int | str
            Index or name of column used as denominator.
        null_col : bool
            Whether to keep the redundant column.

        Returns
        -------
        pandas.DataFrame
            ALR-transformed array, of shape `(N, D-1)`.
        """
        if components is None:
            components = []
        components = self._obj.columns.values.tolist()

        if isinstance(ind, int):
            index_col_no = ind
        elif isinstance(ind, str):
            assert ind in components
            index_col_no = components.index(ind)
        if index_col_no == -1:
            index_col_no += len(components)

        if label_mode.lower().startswith("num"):
            colnames = [f"ALR{ix}" for ix in range(self._obj.columns.size)]
        else:
            colnames = codata.get_ALR_labels(
                self._obj, mode=label_mode, ind=index_col_no
            )

        if not null_col:
            colnames = [n for ix, n in enumerate(colnames) if ix != index_col_no]
        tfm_df = pd.DataFrame(
            codata.ALR(
                self._obj[components].values, ind=index_col_no, null_col=null_col
            ),
            index=self._obj.index,
            columns=colnames,
        )
        tfm_df.attrs["ALR_index"] = index_col_no  # save parameter for inverse_transform
        tfm_df.attrs["inverts_to"] = self._obj.columns.to_list()
        return tfm_df

    def inverse_ALR(
        self, ind: int | str | None = None, null_col: bool = False
    ) -> pd.DataFrame | pd.Series:
        """
        Inverse Additive Log Ratio transformation.

        Parameters
        ----------
        ind: int | str
            Index or name of column used as denominator.
        null_col : bool
            Whether the array contains an extra redundant column
            (i.e. shape is `(N, D)`).

        Returns
        -------
        pandas.DataFrame
            Inverse-ALR transformed array, of shape `(N, D)`.
        """

        colnames = self._obj.attrs.get("inverts_to")

        if ind is None:
            ind = self._obj.attrs.get("ALR_index", -1)

        itfm_df = pd.DataFrame(
            codata.inverse_ALR(self._obj.values, ind=ind, null_col=null_col),
            index=self._obj.index,
            columns=colnames,
        )
        return itfm_df

    @attribute_transform
    def CLR(self, label_mode: str = "simple") -> pd.DataFrame | pd.Series:
        """
        Centred Log Ratio transformation.

        Parameters
        ----------
        label_mode : str
            Labelling mode for the output dataframe (`numeric`, `simple`,
            `LaTeX`). If you plan to use the outputs for automated visualisation
            and want to know which components contribute, use `simple` or
            `LaTeX`.

        Returns
        -------
        pandas.DataFrame
            CLR-transformed array, of shape `(N, D)`.
        """
        if label_mode.lower().startswith("num"):
            colnames = [f"CLR{ix}" for ix in range(self._obj.columns.size)]
        else:
            colnames = codata.get_CLR_labels(self._obj, mode=label_mode)

        tfm_df = pd.DataFrame(
            codata.CLR(self._obj.values),
            index=self._obj.index,
            columns=colnames,
        )
        tfm_df.attrs["inverts_to"] = (
            self._obj.columns.to_list()
        )  # save parameter for inverse_transform
        return tfm_df

    def inverse_CLR(self) -> pd.DataFrame | pd.Series:
        """
        Inverse Centred Log Ratio transformation.

        Parameters
        ----------

        Returns
        -------
        pandas.DataFrame
            Inverse-CLR transformed array, of shape `(N, D)`.
        """
        colnames = self._obj.attrs.get("inverts_to")
        itfm_df = pd.DataFrame(
            codata.inverse_CLR(self._obj.values),
            index=self._obj.index,
            columns=colnames,
        )
        return itfm_df

    @attribute_transform
    def ILR(self, label_mode: str = "simple") -> pd.DataFrame | pd.Series:
        """
        Isometric Log Ratio transformation.

        Parameters
        ----------
        label_mode : str
            Labelling mode for the output dataframe (`numeric`, `simple`,
            `LaTeX`). If you plan to use the outputs for automated visualisation
            and want to know which components contribute, use `simple` or
            `LaTeX`.

        Returns
        -------
        pandas.DataFrame
            ILR-transformed array, of shape `(N, D-1)`
        """
        if label_mode.lower().startswith("num"):
            colnames = [f"ILR{ix}" for ix in range(self._obj.columns.size - 1)]
        else:
            colnames = codata.get_ILR_labels(self._obj, mode=label_mode)

        tfm_df = pd.DataFrame(
            codata.ILR(self._obj.values),
            index=self._obj.index,
            columns=colnames,
        )
        tfm_df.attrs["inverts_to"] = (
            self._obj.columns.to_list()
        )  # save parameter for inverse_transform
        return tfm_df

    def inverse_ILR(self, X: np.ndarray | None = None) -> pd.DataFrame | pd.Series:
        """
        Inverse Isometric Log Ratio transformation.

        Parameters
        ----------
        X : numpy.ndarray
            Optional specification for an array from which to derive the orthonormal basis,
            with shape `(N, D)`.

        Returns
        --------
        pandas.DataFrame
            Inverse-ILR transformed array, of shape `(N, D)`.
        """
        colnames = self._obj.attrs.get("inverts_to")

        itfm_df = pd.DataFrame(
            codata.inverse_ILR(self._obj.values),
            index=self._obj.index,
            columns=colnames,
        )
        return itfm_df

    @attribute_transform
    def boxcox(
        self,
        lmbda: np.number | None = None,
        lmbda_search_space: tuple[float, float] = (-1, 5),
        search_steps: int = 100,
        return_lmbda: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        Box-Cox transformation.

        Parameters
        ---------------
        lmbda : numpy.number
            Lambda value used to forward-transform values. If none, it will be calculated
            using the mean
        lmbda_search_space : tuple
            Range tuple (min, max).
        search_steps : int
            Steps for lambda search range.

        Returns
        -------
        pandas.DataFrame
            Box-Cox transformed array.
        """
        arr, lmbda = codata.boxcox(
            self._obj.values,
            lmbda=lmbda,
            lmbda_search_space=lmbda_search_space,
            search_steps=search_steps,
            return_lmbda=True,
        )
        tfm_df = pd.DataFrame(arr, index=self._obj.index, columns=self._obj.columns)
        tfm_df.attrs["boxcox_lmbda"] = lmbda  # save parameter for inverse_transform
        return tfm_df

    def inverse_boxcox(self, lmbda: float | None = None) -> pd.DataFrame | pd.Series:
        """
        Inverse Box-Cox transformation.

        Parameters
        ---------------
        lmbda : float
            Lambda value used to forward-transform values.

        Returns
        -------
        pandas.DataFrame
            Inverse Box-Cox transformed array.
        """
        if lmbda is None:
            lmbda: float | None = self._obj.attrs.get("boxcox_lmbda")
            assert lmbda is not None, (
                "Can't invert a box-cox transform without a lambda parameter."
            )

        itfm_df = pd.DataFrame(
            codata.inverse_boxcox(self._obj.values, lmbda=lmbda),
            index=self._obj.index,
            columns=self._obj.columns,
        )
        return itfm_df

    @attribute_transform
    def sphere(self) -> pd.DataFrame | pd.Series:
        r"""
        Spherical coordinate transformation for compositional data.

        Returns
        -------
        θ : pandas.DataFrame
            Array of angles in radians (:math:`(0, \pi / 2]`)
        """
        arr = codata.sphere(self._obj.values)
        tfm_df = pd.DataFrame(
            arr,
            index=self._obj.index,
            columns=["θ_" + c for c in self._obj.columns[1:]],
        )
        # save column names for inverse_sphere
        tfm_df.attrs["variables"] = self._obj.columns
        return tfm_df

    def inverse_sphere(self, variables=None) -> pd.DataFrame | pd.Series:
        """
        Inverse spherical coordinate transformation to revert back to compositional data
        in the simplex.

        Parameters
        ----------
        variables : list
            List of names for the compositional data variables, optionally specified
            (for when they may not be stored in the dataframe attributes through
            the :func:`~pyrolite.comp.pyrocomp` functions).

        Returns
        -------
        df : pandas.DataFrame
            Dataframe of original compositional (simplex) coordinates, normalised to 1.
        """
        if variables is None:
            variables = self._obj.attrs.get(
                "variables", np.arange(self._obj.columns.size)
            )

        itfm_df = pd.DataFrame(
            codata.inverse_sphere(self._obj.values),
            index=self._obj.index,
            columns=variables,
        )
        return itfm_df

    def logratiomean(
        self,
        transform: Callable = codata.CLR,
        inverse_transform: Callable = codata.inverse_CLR,
    ) -> pd.Series:
        """
        Take a mean of log-ratios along the index of a dataframe.

        Parameters
        ----------
        transform : Callable
            Log transform to use.
        inverse_transform : Callable
            Inverse transform to use.

        Returns
        -------
        pandas.Series
            Mean values as a pandas series.

        Notes
        -----
        Only makes sense for a dataframe.
        """
        if not isinstance(self._obj, pd.DataFrame):
            raise TypeError("Can't take a compositional mean of a series.")
        return codata.logratiomean(self._obj, transform=transform)

    def invert_transform(self, **kwargs) -> pd.Series | pd.DataFrame:
        """
        Try to inverse-transform a transformed dataframe.
        """
        _colnames = self._obj.attrs.get("inverts_to")

        tfm = self._obj.attrs.get("transform")
        try:
            tfm, inv_tfm = codata.get_transforms(tfm)
        except ValueError:
            raise ValueError("DataFrame has no transform history.")

        _invert_method = getattr(self, inv_tfm.__name__)
        return _invert_method(**kwargs)
