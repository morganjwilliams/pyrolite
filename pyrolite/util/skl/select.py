import pandas as pd
from ...geochem.ind import __common_elements__, __common_oxides__, REE
from ..log import Handle

logger = Handle(__name__)

try:
    from sklearn.base import TransformerMixin, BaseEstimator
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        """Select specific data types from a dataframe for further transformation."""
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """Select specific columns from a dataframe for further transformation."""
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X.loc[:, self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(
                "The DataFrame does not include the columns: %s" % cols_error
            )


class CompositionalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, components=None, inverse=False):
        """Select the oxide and element components from a dataframe."""
        if components is None:
            components = __common_elements__ | __common_oxides__
        self.columns = components
        self.inverse = inverse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.inverse:
            out_cols = [i for i in X.columns if i not in self.columns]
        else:
            out_cols = [i for i in X.columns if i in self.columns]
        out = X.loc[:, out_cols]
        return out


class MajorsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, components=None):
        """Select the major element oxides from a dataframe."""
        if components is None:
            components = __common_oxides__
        self.columns = components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_cols = [i for i in X.columns if i in self.columns]
        out = X.loc[:, out_cols]
        return out


class ElementSelector(BaseEstimator, TransformerMixin):
    def __init__(self, components=None):
        """Select the (trace) elements from a dataframe."""
        if components is None:
            components = __common_elements__
        self.columns = components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_cols = [i for i in X.columns if i in self.columns]
        out = X.loc[:, out_cols]
        return out


class REESelector(BaseEstimator, TransformerMixin):
    def __init__(self, components=None):
        """Select the Rare Earth Elements (REE) from a dataframe."""
        if components is None:
            components = REE()
        components = [i for i in components if not i == "Pm"]
        self.columns = components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_cols = [i for i in self.columns if i in X.columns]
        out = X.loc[:, out_cols]
        return out
