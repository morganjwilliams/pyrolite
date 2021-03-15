"""
Submodule for working with compositional data.
"""

import pandas as pd
from . import codata
from ..util.log import Handle

logger = Handle(__name__)


def attribute_transform(f, *args, **kwargs):
    """
    Decorator to add transform function as a dataframe attribute after
    transformation, for traceability.

    Parameters
    -----------
    f : :class:`func` | :class:`class`
        Transform function.

    Returns
    -------
    :class:`func` | :class:`class`
        Object with modified docstring.
    """

    def wrapper(*args, **kwargs):
        output = f(*args, **kwargs)
        output.attrs["transform"] = f.__name__
        return output

    return wrapper


# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrocomp")
@pd.api.extensions.register_dataframe_accessor("pyrocomp")
class pyrocomp(object):
    def __init__(self, obj):
        """
        Custom dataframe accessor for pyrolite compositional transforms.
        """
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def renormalise(self, components: list = [], scale=100.0):
        """
        Renormalises compositional data to ensure closure.

        Parameters
        ----------
        components : :class:`list`
            Option subcompositon to renormalise to 100. Useful for the use case
            where compostional data and non-compositional data are stored in the
            same dataframe.
        scale : :class:`float`, :code:`100.`
            Closure parameter. Typically either 100 or 1.

        Returns
        -------
        :class:`pandas.DataFrame`
            Renormalized dataframe.

        Notes
        ------
        This won't modify the dataframe in place, you'll need to assign it to something.
        If you specify components, those components will be summed to 100%,
        and others remain unchanged.
        """
        obj = self._obj
        return codata.renormalise(obj, components=components, scale=scale)

    @attribute_transform
    def ALR(self, components=[], ind=-1, null_col=False, label_mode="simple"):
        """
        Additive Log Ratio transformation.

        Parameters
        ----------
        ind: :class:`int`, :class:`str`
            Index or name of column used as denominator.
        null_col : :class:`bool`
            Whether to keep the redundant column.

        Returns
        -------
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

        if label_mode.lower().startswith("num"):
            colnames = ["ALR{}".format(ix) for ix in range(self._obj.columns.size)]
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

    def inverse_ALR(self, ind=None, null_col=False):
        """
        Inverse Additive Log Ratio transformation.

        Parameters
        ----------
        ind: :class:`int`, :class:`str`
            Index or name of column used as denominator.
        null_col : :class:`bool`, :code:`False`
            Whether the array contains an extra redundant column
            (i.e. shape is :code:`(N, D)`).

        Returns
        -------
        :class:`pandas.DataFrame`
            Inverse-ALR transformed array, of shape :code:`(N, D)`.
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
    def CLR(self, label_mode="simple"):
        """
        Centred Log Ratio transformation.

        Parameters
        ----------
        label_mode : :class:`str`
            Labelling mode for the output dataframe (:code:`numeric`, :code:`simple`,
            :code:`LaTeX`). If you plan to use the outputs for automated visualisation
            and want to know which components contribute, use :code:`simple` or
            :code:`LaTeX`.

        Returns
        -------
        :class:`pandas.DataFrame`
            CLR-transformed array, of shape :code:`(N, D)`.
        """
        if label_mode.lower().startswith("num"):
            colnames = ["CLR{}".format(ix) for ix in range(self._obj.columns.size)]
        else:
            colnames = codata.get_CLR_labels(self._obj, mode=label_mode)

        tfm_df = pd.DataFrame(
            codata.CLR(self._obj.values), index=self._obj.index, columns=colnames,
        )
        tfm_df.attrs[
            "inverts_to"
        ] = self._obj.columns.to_list()  # save parameter for inverse_transform
        return tfm_df

    def inverse_CLR(self):
        """
        Inverse Centred Log Ratio transformation.

        Parameters
        ----------

        Returns
        -------
        :class:`pandas.DataFrame`
            Inverse-CLR transformed array, of shape :code:`(N, D)`.
        """
        colnames = self._obj.attrs.get("inverts_to")
        itfm_df = pd.DataFrame(
            codata.inverse_CLR(self._obj.values),
            index=self._obj.index,
            columns=colnames,
        )
        return itfm_df

    @attribute_transform
    def ILR(self, label_mode="simple"):
        """
        Isometric Log Ratio transformation.

        Parameters
        ----------
        label_mode : :class:`str`
            Labelling mode for the output dataframe (:code:`numeric`, :code:`simple`,
            :code:`LaTeX`). If you plan to use the outputs for automated visualisation
            and want to know which components contribute, use :code:`simple` or
            :code:`LaTeX`.

        Returns
        -------
        :class:`pandas.DataFrame`
            ILR-transformed array, of shape :code:`(N, D-1)`.
        """
        if label_mode.lower().startswith("num"):
            colnames = ["ILR{}".format(ix) for ix in range(self._obj.columns.size - 1)]
        else:
            colnames = codata.get_ILR_labels(self._obj, mode=label_mode)

        tfm_df = pd.DataFrame(
            codata.ILR(self._obj.values), index=self._obj.index, columns=colnames,
        )
        tfm_df.attrs[
            "inverts_to"
        ] = self._obj.columns.to_list()  # save parameter for inverse_transform
        return tfm_df

    def inverse_ILR(self, X=None):
        """
        Inverse Isometric Log Ratio transformation.

        Parameters
        ----------
        X : :class:`numpy.ndarray`, :code:`None`
            Optional specification for an array from which to derive the orthonormal basis,
            with shape :code:`(N, D)`.

        Returns
        --------
        :class:`pandas.DataFrame`
            Inverse-ILR transformed array, of shape :code:`(N, D)`.
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
            lmbda = self._obj.attrs.get("boxcox_lmbda")
            assert (
                lmbda is not None
            ), "Can't invert a box-cox transform without a lambda parameter."

        itfm_df = pd.DataFrame(
            codata.inverse_boxcox(self._obj.values, lmbda=lmbda),
            index=self._obj.index,
            columns=self._obj.columns,
        )
        return itfm_df

    def logratiomean(self, transform=codata.CLR, inverse_transform=codata.inverse_CLR):
        """
        Take a mean of log-ratios along the index of a dataframe.

        Parameters
        ----------
        transform : :class:`callable` : :class:`str`
            Log transform to use.

        Returns
        -------
        :class:`pandas.Series`
            Mean values as a pandas series.
        """
        return codata.logratiomean(self._obj, transform=transform)

    def invert_transform(self, **kwargs):
        """
        Try to inverse-transform a transformed dataframe.
        """
        colnames = self._obj.attrs.get("inverts_to")

        tfm = self._obj.attrs.get("transform")
        try:
            tfm, inv_tfm = codata.get_transforms(tfm)
        except ValueError:
            raise ValueError("DataFrame has no transform history.")

        _invert_method = getattr(self, inv_tfm.__name__)
        return _invert_method(**kwargs)
