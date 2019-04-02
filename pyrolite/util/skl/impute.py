import numpy as np
import pandas as pd
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


try:
    from sklearn.base import TransformerMixin, BaseEstimator
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)

try:
    from fancyimpute import IterativeImputer, SoftImpute
except ImportError:
    msg = "fancyimpute not installed"
    logger.warning(msg)


class MultipleImputer(BaseEstimator, TransformerMixin):
    """
    Multiple Imputation via fancyimpute.IterativeImputer.
    """

    def __init__(self, multiple=5, n_iter=10, groupby=None, *args, **kwargs):
        self.multiple = multiple
        self.n_iter = n_iter
        self.args = args
        self.kwargs = kwargs
        self.groupby = groupby

    def transform(self, X, *args, **kwargs):
        assert isinstance(X, pd.DataFrame)
        df = pd.DataFrame(columns=X.columns, index=X.index)
        if isinstance(self.imputers, dict):
            for c, d in self.imputers.items():
                mask = d["mask"]
                imputers = d["impute"]
                imputed_data = np.array([imp.transform(X[mask, :]) for imp in imputers])
                mean = np.mean(imputed_data, axis=0)
                df.loc[mask, ~pd.isnull(X[mask, :]).all(axis=0)] = mean
            return df
        else:
            imputed_data = np.array([imp.transform(X) for imp in self.imputers])
            mean = np.mean(imputed_data, axis=0)
            df.loc[:, ~pd.isnull(X).all(axis=0)] = mean
            return df

    """
    def inverse_transform(self, Y, *args, **kwargs):
        # For non-compositional data, take the mask and reverting to nan
        # for compositional data, renormalisation would be needed
        pass
    """

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        start = X
        y_present = y is not None
        groupby_present = self.groupby is not None
        self.imputers = []
        if y_present or groupby_present:
            assert not (groupby_present and y_present)
            if y_present:
                classes = np.unique(y)
                gen_mask = lambda c: y == c
            if groupby_present:
                classes = X[self.groupby].unique()
                gen_mask = lambda c: X[self.groupby] == c
            self.imputers = {
                c: {
                    "impute": [
                        IterativeImputer(
                            n_iter=self.n_iter,
                            sample_posterior=True,
                            random_state=ix,
                            **self.kwargs
                        )
                        for ix in range(self.multiple)
                    ],
                    "mask": gen_mask(c),
                }
                for c in classes
            }

            msg = """Imputation transformer: {} imputers x {} classes""".format(
                self.multiple, len(classes)
            )
            logger.info(msg)

            for c, d in self.imputers.items():
                for imp in d["impute"]:
                    imp.fit(X[d["mask"], :])

        else:
            for ix in range(self.multiple):
                self.imputers.append(
                    IterativeImputer(
                        n_iter=self.n_iter,
                        sample_posterior=True,
                        random_state=ix,
                        **self.kwargs
                    )
                )
            msg = """Imputation transformer: {} imputers""".format(self.multiple)
            logger.info(msg)
            for ix in range(self.multiple):
                self.imputers[ix].fit(X)

        return self


class PdSoftImputer(BaseEstimator, TransformerMixin):
    """
    Multiple Imputation via fancyimpute.SoftImpute.
    """

    def __init__(self, max_iters=100, groupby=None, donotimpute=[], *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.max_iters = max_iters
        self.groupby = groupby
        self.donotimpute = donotimpute

    def transform(self, X, *args, **kwargs):
        """
        Impute Missing Values

        Todo
        ------
            * Need to use masks to avoid :class:`fancyimpute.SoftImpute` returning 0. where it cannot impute.
        """
        assert isinstance(X, pd.DataFrame)
        df = pd.DataFrame(columns=X.columns, index=X.index)  # df of nans
        df.loc[:, self.donotimpute] = X.loc[:, self.donotimpute]
        to_impute = [i for i in X.columns if not i in self.donotimpute]
        imputable = ~pd.isnull(X.loc[:, to_impute]).all(axis=1)
        if isinstance(self.imputer, dict):
            for c, d in self.imputer.items():
                mask = d["mask"]
                mask = mask & imputable
                imputer = d["impute"]
                imputed_data = imputer.fit_transform(X.loc[mask, to_impute])
                assert imputed_data.shape[0] == X.loc[mask, :].index.size
                df.loc[mask, to_impute] = imputed_data

            return df
        else:
            imputed_data = self.imputer.fit_transform(X.loc[imputable, to_impute])
            assert imputed_data.shape[0] == X.loc[imputable, :].index.size
            df.loc[imputable, to_impute] = imputed_data
            return df

    """
    def inverse_transform(self, Y, *args, **kwargs):
        # For non-compositional data, take the mask and reverting to nan
        # for compositional data, renormalisation would be needed
        pass
    """

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        start = X
        y_present = y is not None
        groupby_present = self.groupby is not None
        self.imputer = []
        if y_present or groupby_present:
            assert not (groupby_present and y_present)
            if y_present:
                classes = np.unique(y)
                gen_mask = lambda c: y == c
            if groupby_present:
                classes = X[self.groupby].unique()
                gen_mask = lambda c: X[self.groupby] == c
            self.imputer = {
                c: {
                    "impute": SoftImpute(max_iters=self.max_iters, **self.kwargs),
                    "mask": gen_mask(c),
                }
                for c in classes
            }

            msg = """Building Soft Imputation Transformers for {} classes""".format(
                len(classes)
            )
            logger.info(msg)

        else:
            self.imputer = SoftImpute(max_iters=self.max_iters, **self.kwargs)
            msg = """Building Soft Imputation Transformer"""
            logger.info(msg)

        return self
