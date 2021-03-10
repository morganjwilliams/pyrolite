import numpy as np
import pandas as pd
from ...geochem import transform
from ...geochem import ind
from ...geochem import parse
from ...comp import codata
from ..lambdas import orthogonal_polynomial_constants
from ..log import Handle

logger = Handle(__name__)

try:
    from sklearn.base import TransformerMixin, BaseEstimator
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)


class DropBelowZero(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        """Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label = "Feedthrough"

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            out = X.where(X > 0, np.nan)
        else:
            out = np.where(X > 0, X, np.nan)
        return out

    def fit(self, X, *args):
        return self


class LinearTransform(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        """Linear Transformer for scikit-learn like use."""
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


class ExpTransform(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        """Exponential Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label = "Feedthrough"
        self.forward = np.exp
        self.inverse = np.log

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            out = X.applymap(self.forward)
        elif isinstance(X, pd.Series):
            out = X.apply(self.forward)
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if isinstance(Y, pd.DataFrame):
            out = Y.applymap(self.inverse)
        elif isinstance(Y, pd.Series):
            out = Y.apply(self.inverse)
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args):
        return self


class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        """Log Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label = "Feedthrough"
        self.forward = np.log
        self.inverse = np.exp

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


class ALRTransform(BaseEstimator, TransformerMixin):
    def __init__(self, label_mode="numeric", **kwargs):
        """Additive Log Ratio Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label_mode = label_mode
        self.label = "ALR"
        self.forward = codata.ALR
        self.inverse = codata.inverse_ALR

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            out = X.pyrocomp.ALR(
                label_mode=self.label_mode, **{**self.kpairs, **kwargs}
            )
        elif isinstance(X, pd.Series):
            out = X.to_frame().T.pyrocomp.ALR(
                label_mode=self.label_mode, **{**self.kpairs, **kwargs}
            )
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if isinstance(Y, pd.DataFrame):
            out = Y.pyrocomp.inverse_ALR(**kwargs)
        elif isinstance(Y, pd.Series):
            out = Y.to_frame().T.pyrocomp.inverse_ALR(**kwargs)
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args, **kwargs):
        return self


class CLRTransform(BaseEstimator, TransformerMixin):
    def __init__(self, label_mode="numeric", **kwargs):
        """Centred Log Ratio Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label_mode = label_mode
        self.label = "CLR"
        self.forward = codata.CLR
        self.inverse = codata.inverse_CLR

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            out = X.pyrocomp.CLR(
                label_mode=self.label_mode, **{**self.kpairs, **kwargs}
            )
        elif isinstance(X, pd.Series):
            out = X.to_frame().T.pyrocomp.CLR(
                label_mode=self.label_mode, **{**self.kpairs, **kwargs}
            )
        else:
            out = self.forward(np.array(X), *args, **{**self.kpairs, **kwargs})
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if isinstance(Y, pd.DataFrame):
            out = Y.pyrocomp.inverse_CLR(**{**self.kpairs, **kwargs})
        elif isinstance(Y, pd.Series):
            out = Y.to_frame().T.pyrocomp.inverse_CLR(**{**self.kpairs, **kwargs})
        else:
            out = self.inverse(np.array(Y), *args, **{**self.kpairs, **kwargs})
        return out

    def fit(self, X, *args, **kwargs):
        return self


class ILRTransform(BaseEstimator, TransformerMixin):
    def __init__(self, label_mode="numeric", **kwargs):
        """Isometric Log Ratio Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label_mode = label_mode
        self.label = "ILR"
        self.forward = codata.ILR
        self.inverse = codata.inverse_ILR
        self.X = None

    def transform(self, X, *args, **kwargs):
        self.X = np.array(X)
        if isinstance(X, pd.DataFrame):
            out = X.pyrocomp.ILR(
                label_mode=self.label_mode, **{**self.kpairs, **kwargs}
            )
        elif isinstance(X, pd.Series):
            out = X.to_frame().T.pyrocomp.ILR(
                label_mode=self.label_mode, **{**self.kpairs, **kwargs}
            )
        else:
            out = self.forward(np.array(X), *args, **kwargs)
        return out

    def inverse_transform(self, Y, *args, **kwargs):
        if "X" not in kwargs:
            if self.X is not None:
                kwargs.update(dict(X=self.X))
        if isinstance(Y, pd.DataFrame):
            out = Y.pyrocomp.inverse_ILR(**kwargs)
        elif isinstance(Y, pd.Series):
            out = Y.to_frame().T.pyrocomp.inverse_ILR(**kwargs)
        else:
            out = self.inverse(np.array(Y), *args, **kwargs)
        return out

    def fit(self, X, *args, **kwargs):
        return self


class BoxCoxTransform(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        """BoxCox Transformer for scikit-learn like use."""
        self.kpairs = kwargs
        self.label = "BoxCox"
        self.forward = codata.boxcox
        self.inverse = codata.inverse_boxcox
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


class Devolatilizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, exclude=["H2O", "H2O_PLUS", "H2O_MINUS", "CO2", "LOI"], renorm=True
    ):
        """Devolatilization transformer for scikit-learn like use."""
        self.exclude = [i.upper() for i in exclude]
        self.renorm = renorm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        exclude = [i for i in X.columns if i.upper() in self.exclude]
        return transform.devolatilise(X, exclude=exclude, renorm=self.renorm)


class ElementAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, renorm=True, form="oxide"):
        """Element-based aggregation transformer for scikit-learn like use."""
        self.renorm = renorm
        self.form = form

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        multiple_entries = parse.check_multiple_cation_inclusion(X)

        for el in multiple_entries:
            if self.form == "oxide":
                out = ind.simple_oxides(el)[0]
            else:
                out = el
            X = transform.aggregate_element(X, to=out)
        return X


class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, norm_to="Chondrite_PON", exclude=["Pm", "Eu", "Ce"], params=None, degree=5
    ):
        """Lambda coefficient transformer for scikit-learn like use."""
        self.norm_to = norm_to
        self.ree = [i for i in ind.REE() if i not in exclude]
        self.radii = np.array(ind.get_ionic_radii(self.ree, charge=3, coordination=8))
        self.exclude = exclude
        if params is None:
            self.degree = degree
            self.params = orthogonal_polynomial_constants(
                self.radii, degree=self.degree
            )
        else:
            self.params = params
            self.degree = len(params)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        ree_present = [i in X.columns for i in self.ree]
        if not all(ree_present):
            self.ree = [i for i in self.ree if i in X.columns]
            self.radii = self.radii[ree_present]
            self.params = orthogonal_polynomial_constants(
                self.radii, degree=self.degree
            )

        return transform.lambda_lnREE(
            X,
            norm_to=self.norm_to,
            params=self.params,
            degree=self.degree,
            exclude=self.exclude,
        )
