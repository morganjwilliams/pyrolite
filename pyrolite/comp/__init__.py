"""
Submodule for working with compositional data.
"""

import pandas as pd
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from .codata import *
from .aggregate import *
from .impute import *

# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrocomp")
@pd.api.extensions.register_dataframe_accessor("pyrocomp")
class pyrocomp(object):
    """
    Custom dataframe accessor for pyrolite compositional transforms.
    """

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def renormalise(self, components: list = [], scale=100.0):
        """
        Renormalises compositional data to ensure closure.

        Parameters
        ------------
        components : :class:`list`
            Option subcompositon to renormalise to 100. Useful for the use case
            where compostional data and non-compositional data are stored in the
            same dataframe.
        scale : :class:`float`, :code:`100.`
            Closure parameter. Typically either 100 or 1.

        Returns
        --------
        :class:`pandas.DataFrame`
            Renormalized dataframe.

        Notes
        ------
        This won't modify the dataframe in place, you'll need to assign it to something.
        If you specify components, those components will be summed to 100%,
        and others remain unchanged.
        """
        obj = self._obj
        return renormalise(obj, components=components, scale=scale)

    def ALR(self, components=[], ind=-1, null_col=False):
        """
        Additive Log Ratio transformation.

        Parameters
        ---------------
        ind: :class:`int`, :class:`str`
            Index or name of column used as denominator.
        null_col : :class:`bool`
            Whether to keep the redundant column.

        Returns
        ---------
        :class:`pandas.DataFrame`
            ALR-transformed array, of shape :code:`(N, D-1)`.
        """
        components = self._obj.columns.values.tolist()

        if isinstance(ind, int):
            index_col_no = ind
        elif isinstance(ind, str):
            assert ind in components
            index_col_no = components.index(ind)
        if index_col_no == -1:
            index_col_no += len(components)
        index_col = components[index_col_no]
        colnames = ["ALR({}/{})".format(c, index_col) for c in components]

        if not null_col:
            colnames = [n for ix, n in enumerate(colnames) if ix != index_col_no]
        tfm_df = pd.DataFrame(
            alr(self._obj[components].values, ind=index_col_no, null_col=null_col),
            index=self._obj.index,
            columns=colnames,
        )
        tfm_df.alr_index = index_col_no  # save parameter for inverse_transform
        tfm_df.inverts_to = self._obj.columns.to_list()
        return tfm_df

    def inverse_ALR(self, ind=None, null_col=False):
        """
        Inverse Additive Log Ratio transformation.

        Parameters
        ---------------
        ind: :class:`int`, :class:`str`
            Index or name of column used as denominator.
        null_col : :class:`bool`, :code:`False`
            Whether the array contains an extra redundant column
            (i.e. shape is :code:`(N, D)`).

        Returns
        --------
        :class:`pandas.DataFrame`
            Inverse-ALR transformed array, of shape :code:`(N, D)`.
        """

        if hasattr(self._obj, "inverts_to"):
            colnames = self._obj.inverts_to

        if ind is None:
            ind = self._obj.alr_index or -1

        itfm_df = pd.DataFrame(
            inverse_alr(self._obj.values, ind=ind, null_col=null_col),
            index=self._obj.index,
            columns=colnames,
        )
        return itfm_df

    def CLR(self):
        """
        Centred Log Ratio transformation.

        Parameters
        ---------------

        Returns
        ---------
        :class:`pandas.DataFrame`
            CLR-transformed array, of shape :code:`(N, D)`.
        """
        colnames = ["CLR({}/g)".format(c) for c in self._obj.columns]
        tfm_df = pd.DataFrame(
            clr(self._obj.values), index=self._obj.index, columns=colnames,
        )
        tfm_df.inverts_to = (
            self._obj.columns.to_list()
        )  # save parameter for inverse_transform
        return tfm_df

    def inverse_CLR(self):
        """
        Inverse Centred Log Ratio transformation.

        Parameters
        --------------

        Returns
        ---------
        :class:`pandas.DataFrame`
            Inverse-CLR transformed array, of shape :code:`(N, D)`.
        """
        if hasattr(self._obj, "inverts_to"):
            colnames = self._obj.inverts_to

        itfm_df = pd.DataFrame(
            inverse_clr(self._obj.values), index=self._obj.index, columns=colnames,
        )
        return itfm_df

    def ILR(self):
        """
        Isometric Log Ratio transformation.

        Parameters
        ---------------

        Returns
        --------
        :class:`pandas.DataFrame`
            ILR-transformed array, of shape :code:`(N, D-1)`.
        """
        colnames = ["ILR{}".format(ix) for ix in range(self._obj.columns.size - 1)]
        tfm_df = pd.DataFrame(
            ilr(self._obj.values), index=self._obj.index, columns=colnames,
        )
        tfm_df.inverts_to = (
            self._obj.columns.to_list()
        )  # save parameter for inverse_transform
        return tfm_df

    def inverse_ILR(self, X=None):
        """
        Inverse Isometric Log Ratio transformation.

        Parameters
        ---------------
        X : :class:`numpy.ndarray`, :code:`None`
            Optional specification for an array from which to derive the orthonormal basis,
            with shape :code:`(N, D)`.

        Returns
        --------
        :class:`pandas.DataFrame`
            Inverse-ILR transformed array, of shape :code:`(N, D)`.
        """
        if hasattr(self._obj, "inverts_to"):
            colnames = self._obj.inverts_to

        itfm_df = pd.DataFrame(
            inverse_ilr(self._obj.values), index=self._obj.index, columns=colnames,
        )
        return itfm_df

    def boxcox(
        self,
        lmbda=None,
        lmbda_search_space=(-1, 5),
        search_steps=100,
        return_lmbda=False,
    ):
        """
        Box-Cox transformation.

        Parameters
        ---------------
        lmbda : :class:`numpy.number`, :code:`None`
            Lambda value used to forward-transform values. If none, it will be calculated
            using the mean
        lmbda_search_space : :class:`tuple`
            Range tuple (min, max).
        search_steps : :class:`int`
            Steps for lambda search range.

        Returns
        -------
        :class:`pandas.DataFrame`
            Box-Cox transformed array.
        """
        arr, lmbda = boxcox(
            self._obj.values,
            lmbda=lmbda,
            lmbda_search_space=lmbda_search_space,
            search_steps=search_steps,
            return_lmbda=True,
        )
        tfm_df = pd.DataFrame(arr, index=self._obj.index, columns=self._obj.columns)
        tfm_df.boxcox_lmbda = lmbda  # save parameter for inverse_transform
        return tfm_df

    def inverse_boxcox(self, lmbda=None):
        """
        Inverse Box-Cox transformation.

        Parameters
        ---------------
        lmbda : :class:`float`
            Lambda value used to forward-transform values.

        Returns
        -------
        :class:`pandas.DataFrame`
            Inverse Box-Cox transformed array.
        """
        if lmbda is None:
            assert hasattr(
                self._obj, "boxcox_lmbda"
            ), "Can't invert a box-cox transform without a lambda parameter."
            lmbda = self._obj.boxcox_lmbda
        itfm_df = pd.DataFrame(
            inverse_boxcox(self._obj.values, lmbda=lmbda),
            index=self._obj.index,
            columns=self._obj.columns,
        )
        return itfm_df

    def logratiomean(self, transform=clr, inverse_transform=inverse_clr):
        """
        Take a mean of log-ratios along the index of a dataframe.

        Parameters
        -----------
        transform : :class:`callable`
            Log transform to use.
        inverse_transform : :class:`callable`
            Inverse of log transform.

        Returns
        ---------
        :class:`pandas.Series`
            Mean values as a pandas series.
        """
        return logratiomean(
            self._obj, transform=transform, inverse_transform=inverse_transform
        )
