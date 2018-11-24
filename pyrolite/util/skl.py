import numpy as np
import pandas as pd
from functools import partial
from sklearn.base import TransformerMixin
from ..comp.codata import *
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class LinearTransform(TransformerMixin):
    """
    Linear Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = "Feedthrough"
        self.forward = lambda x: x
        self.inverse = lambda x: x

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            out = X.copy(deep=True)
            out.loc[:, :] = self.forward(X.values, *args, **kwargs)
        elif isinstance(X, pd.Series):
            out = X.copy(deep=True)
            out.loc[:] = self.forward(X.values, *args, **kwargs)
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if isinstance(Y, pd.DataFrame):
            out = Y.copy(deep=True)
            out.loc[:, :] = self.inverse(Y.values, *args, **kwargs)
        elif isinstance(Y, pd.Series):
            out = Y.copy(deep=True)
            out.loc[:] = self.inverse(Y.values, *args, **kwargs)
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args):
        return self


class ALRTransform(TransformerMixin):
    """
    Additive Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = "ALR"
        self.forward = alr
        self.inverse = inv_alr

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            out = pd.DataFrame(
                index=X.index, data=self.forward(X.values, *args, **kwargs)
            )
        elif isinstance(X, pd.Series):
            out = pd.Series(index=X.index, data=self.forward(X.values, *args, **kwargs))
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if isinstance(Y, pd.DataFrame):
            out = pd.DataFrame(
                index=Y.index, data=self.inverse(Y.values, *args, **kwargs)
            )
        elif isinstance(Y, pd.Series):
            out = pd.Series(index=Y.index, data=self.inverse(Y.values, *args, **kwargs))
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args, **kwargs):
        return self


class CLRTransform(TransformerMixin):
    """
    Centred Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = "CLR"
        self.forward = clr
        self.inverse = inv_clr

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            out = X.copy(deep=True)
            out.loc[:, :] = self.forward(X.values, *args, **kwargs)
        elif isinstance(X, pd.Series):
            out = X.copy(deep=True)
            out.loc[:] = self.forward(X.values, *args, **kwargs)
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if isinstance(Y, pd.DataFrame):
            out = Y.copy(deep=True)
            out.loc[:, :] = self.inverse(Y.values, *args, **kwargs)
        elif isinstance(Y, pd.Series):
            out = Y.copy(deep=True)
            out.loc[:] = self.inverse(Y.values, *args, **kwargs)
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args, **kwargs):
        return self


class ILRTransform(TransformerMixin):
    """
    Isometric Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = "ILR"
        self.forward = ilr
        self.inverse = inv_ilr

    def transform(self, X, *args, **kwargs):
        self.X = np.array(X)
        if isinstance(X, pd.DataFrame):
            out = pd.DataFrame(
                index=X.index, data=self.forward(X.values, *args, **kwargs)
            )
        elif isinstance(X, pd.Series):
            out = X.copy(deep=True)
            out.loc[:] = self.forward(X.values, *args, **kwargs)
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if "X" not in kwargs:
            kwargs.update(dict(X=self.X))
        if isinstance(Y, pd.DataFrame):
            out = pd.DataFrame(
                index=Y.index, data=self.inverse(Y.values, *args, **kwargs)
            )
        elif isinstance(Y, pd.Series):
            out = pd.Series(index=Y.index, data=self.inverse(Y.values, *args, **kwargs))
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args, **kwargs):
        return self


class BoxCoxTransform(TransformerMixin):
    """
    BoxCox Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = "BoxCox"
        self.forward = boxcox
        self.inverse = inv_boxcox
        self.lmbda = None

    def transform(self, X, *args, **kwargs):
        self.X = np.array(X)
        if "lmbda" not in kwargs:
            if not (self.lmbda is None):
                kwargs.update(dict(lmbda=self.lmbda))
                data = self.forward(X, *args, **kwargs)
            else:
                kwargs.update(dict(return_lmbda=True))
                data, lmbda = self.forward(X, *args, **kwargs)
                self.lmbda = lmbda
        return data

    def inverse_transform(self, Y, *args, **kwargs):
        if "lmbda" not in kwargs:
            kwargs.update(dict(lmbda=self.lmbda))
        return self.inverse(Y, *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        bc_data, lmbda = boxcox(X, *args, **kwargs)
        self.lmbda = lmbda
