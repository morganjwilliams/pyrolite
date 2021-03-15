import numpy as np
import pandas as pd
from ..log import Handle

logger = Handle(__name__)

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
            for cls, content in self.imputers.items():
                mask = content["mask"]
                imputers = content["impute"]
                imputed_data = np.array(
                    [imp.transform(X.loc[mask, :]) for imp in imputers]
                )
                mean = np.mean(imputed_data, axis=0)
                df.loc[mask, ~pd.isnull(X.loc[mask, :]).all(axis=0)] = mean
            return df
        else:
            imputed_data = np.array([imp.transform(X) for imp in self.imputers])
            mean = np.mean(imputed_data, axis=0)
            df.loc[:, ~pd.isnull(X).all(axis=0)] = mean
            return df

    # def inverse_transform(self, Y, *args, **kwargs):
    #    # For non-compositional data, take the mask and reverting to nan
    #    # for compositional data, renormalisation would be needed
    #    pass

    def fit(self, X, y=None):
        """
        Fit the imputers.

        Parameters
        ----------
        X : :class:`pandas.DataFrame`
            Data to use to fit the imputations.
        y : :class:`pandas.Series`
            Target class; optionally specified, and used similarly to `groupby`.
        """
        assert isinstance(X, pd.DataFrame)
        # start = X
        y_present = y is not None
        groupby_present = self.groupby is not None
        self.imputers = []
        if y_present or groupby_present:
            # here works for one or the other, but could technically split for this
            assert not (groupby_present and y_present)
            if y_present:
                classes = np.unique(y)
                gen_mask = lambda c: np.array(y == c)
            if groupby_present:
                classes = X[self.groupby].unique()
                gen_mask = lambda c: np.array(X[self.groupby] == c)  # pd.Series values
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

            for cls, content in self.imputers.items():
                for imp in content["impute"]:
                    imp.fit(X.loc[content["mask"], :])

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
