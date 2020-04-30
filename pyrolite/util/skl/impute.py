import numpy as np
import pandas as pd
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.impute import IterativeImputer
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)


class MultipleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, multiple=5, max_iter=10, groupby=None, *args, **kwargs):
        """
        Multiple Imputation via :class:`sklearn.IterativeImputer`.

        Parameters
        ----------
        multiple : :class:`int`
            How many imputers to bulid.
        max_iter : :class:`int`
            Maximum number of iterations for each imputation.
        groupby : :class:`str`
            Column to group by to impute each group separately.
        """
        self.multiple = multiple
        self.max_iter = max_iter
        self.args = args
        self.kwargs = kwargs
        self.groupby = groupby

    def transform(self, X, *args, **kwargs):
        assert isinstance(X, pd.DataFrame)
        df = pd.DataFrame(columns=X.columns, index=X.index)
        if isinstance(self.imputers, dict):
            for d in self.imputers.values():
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


    #def inverse_transform(self, Y, *args, **kwargs):
    #    # For non-compositional data, take the mask and reverting to nan
    #    # for compositional data, renormalisation would be needed
    #    pass

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # start = X
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
                            max_iter=self.max_iter,
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

            for d in self.imputers.values():
                for imp in d["impute"]:
                    imp.fit(X[d["mask"], :])

        else:
            for ix in range(self.multiple):
                self.imputers.append(
                    IterativeImputer(
                        max_iter=self.max_iter,
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
