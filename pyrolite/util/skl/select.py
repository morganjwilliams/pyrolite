from ..log import Handle

logger = Handle(__file__)
try:
    from sklearn.base import BaseEstimator, TransformerMixin

    HAVE_SKL = True
except ImportError:
    HAVE_SKL = False

if HAVE_SKL:
    import numpy as np
    import pandas as pd

    from ...geochem.ind import REE, _common_elements, _common_oxides

    class TypeSelector(BaseEstimator, TransformerMixin):
        def __init__(self, dtype: str | np.dtype):
            """Select specific data types from a dataframe for further transformation."""
            self.dtype = dtype

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
            return self

        def transform(self, X):
            assert isinstance(X, pd.DataFrame)
            return X.select_dtypes(include=[self.dtype])

    class ColumnSelector(BaseEstimator, TransformerMixin):
        def __init__(self, columns: list[str]):
            """Select specific columns from a dataframe for further transformation."""
            self.columns = columns

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
            return self

        def transform(self, X: pd.DataFrame):
            assert isinstance(X, pd.DataFrame)

            try:
                return X.loc[:, self.columns]
            except KeyError:
                cols_error = list(set(self.columns) - set(X.columns))
                raise KeyError(
                    f"The DataFrame does not include the columns: {cols_error}"
                )

    class CompositionalSelector(BaseEstimator, TransformerMixin):
        def __init__(self, columns: list[str] | None = None, inverse: bool = False):
            """Select the oxide and element components from a dataframe."""
            if columns is None:
                columns = list(_common_elements | _common_oxides)
            self.columns = columns
            self.inverse = inverse

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
            return self

        def transform(self, X: pd.DataFrame):
            assert isinstance(X, pd.DataFrame)
            if self.inverse:
                out_cols = [i for i in X.columns if i not in self.columns]
            else:
                out_cols = [i for i in X.columns if i in self.columns]
            out = X.loc[:, out_cols]
            return out

    class MajorsSelector(BaseEstimator, TransformerMixin):
        def __init__(self, components: list[str] | None = None):
            """Select the major element oxides from a dataframe."""
            if components is None:
                components = list(_common_oxides)
            self.columns = components

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
            return self

        def transform(self, X: pd.DataFrame):
            assert isinstance(X, pd.DataFrame)
            out_cols = [i for i in X.columns if i in self.columns]
            out = X.loc[:, out_cols]
            return out

    class ElementSelector(BaseEstimator, TransformerMixin):
        def __init__(self, components: list[str] | None = None):
            """Select the (trace) elements from a dataframe."""
            if components is None:
                components = list(_common_elements)
            self.columns = components

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
            return self

        def transform(self, X):
            assert isinstance(X, pd.DataFrame)
            out_cols = [i for i in X.columns if i in self.columns]
            out = X.loc[:, out_cols]
            return out

    class REESelector(BaseEstimator, TransformerMixin):
        def __init__(self, components: list[str] | None = None):
            """Select the Rare Earth Elements (REE) from a dataframe."""
            if components is None:
                components = REE()
            components = [i for i in components if i != "Pm"]
            self.columns = components

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
            return self

        def transform(self, X: pd.DataFrame):
            assert isinstance(X, pd.DataFrame)
            out_cols = [i for i in self.columns if i in X.columns]
            out = X.loc[:, out_cols]
            return out
