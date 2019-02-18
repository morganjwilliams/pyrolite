import numpy as np
import pandas as pd
from functools import partial
import itertools
from ..geochem import *
from ..geochem.ind import __common_elements__, __common_oxides__
from ..comp.codata import *
from ..comp.aggregate import *
from .plot import *

import matplotlib.colors as mplc

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

try:
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.metrics import confusion_matrix
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)

try:
    from fancyimpute import IterativeImputer, SoftImpute
except ImportError:
    msg = "fancyimpute not installed"
    logger.warning(msg)

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    msg = "imbalanced-learn not installed"
    logger.warning(msg)


def get_confusion_matrix(clf, test_X, test_y):
    y_true = test_y
    y_pred = clf.predict(test_X)
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    *args,
    classes=[],
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    norm=mplc.Normalize(vmin=0, vmax=1.0),
    ax=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if len(args) == 1:
        cm = args[0]
    else:
        cm = get_confusion_matrix(*args)
        if not classes:
            if hasattr(args[0], "classes_"):
                classes = list(args[0].classes_)

    if not classes:
        classes = np.arange(cm.shape[0])

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots(1)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set(
        ylabel="True",
        xlabel="Predicted",
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.tight_layout()
    return ax


def plot_gs_results(gs, xvar=None, yvar=None):
    """Plots the results from a GridSearch showing location of optimum in 2D."""
    labels = gs.param_grid.keys()
    grid_items = list(gs.param_grid.items())
    no_items = len(grid_items)
    if (
        len(grid_items) == 1
    ):  # if there's only one item, there's only one way to plot it.
        (xvar, xx) = grid_items[0]
        (yvar, yy) = "", np.array([0])
    else:
        if xvar is None and yvar is None:
            (yvar, yy), (xvar, xx) = [(k, v) for (k, v) in grid_items][:3]
        elif xvar is not None and yvar is not None:
            yy, xx = gs.param_grid[yvar], gs.param_grid[xvar]
        else:
            if xvar is not None:
                xx = gs.param_grid[xvar]
                (yvar, yy) = [(k, v) for (k, v) in grid_items if not k == xvar][0]
            else:
                yy = gs.param_grid[yvar]
                (xvar, xx) = [(k, v) for (k, v) in grid_items if not k == yvar][0]
    xx, yy = np.array(xx), np.array(yy)
    other_keys = [i for i in gs.param_grid.keys() if i not in [xvar, yvar]]
    if other_keys:
        pass
    else:
        results = np.array(gs.cv_results_["mean_test_score"]).reshape(xx.size, yy.size)
    fig, ax = plt.subplots(1)
    ax.imshow(results.T, cmap=plt.cm.Blues)

    ax.set(
        xlabel=xvar,
        ylabel=yvar,
        xticks=np.arange(len(xx)),
        yticks=np.arange(len(yy)),
        xticklabels=["{:01.2g}".format(i) for i in xx],
        yticklabels=["{:01.2g}".format(i) for i in yy],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.invert_yaxis()

    max = np.nanmax(results)
    locmax = np.where(results == max)
    x, y = locmax
    ax.scatter(x, y, marker="D", s=100, c="k")
    return ax


def plot_cooccurence(
    df,
    ax=None,
    normalize=True,
    log=False,
    norm=mplc.Normalize(vmin=0, vmax=1.0),
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4.2, 4))
    co_occur = df.fillna(0)
    co_occur[co_occur > 0] = 1
    co_occur = co_occur.T.dot(co_occur).astype(int)
    if normalize:
        diags = np.diagonal(co_occur)
        for i in range(diags.shape[0]):
            for j in range(diags.shape[0]):
                co_occur.iloc[i, j] = co_occur.iloc[i, j] / np.max([diags[i], diags[j]])
    if log:
        co_occur = co_occur.applymap(np.log)
    heatmap = ax.pcolor(co_occur, norm=norm, **kwargs)
    ax.set_yticks(np.arange(co_occur.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(co_occur.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(df.columns, minor=False, rotation=90)
    ax.set_yticklabels(df.columns, minor=False)
    add_colorbar(heatmap)
    return ax


class DropBelowZero(BaseEstimator, TransformerMixin):
    """
    Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
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


class ExpTransform(BaseEstimator, TransformerMixin):
    """
    Exponential Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
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
    """
    Log Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
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
    """
    Additive Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = "ALR"
        self.forward = alr
        self.inverse = inverse_alr

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


class CLRTransform(BaseEstimator, TransformerMixin):
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


class ILRTransform(BaseEstimator, TransformerMixin):
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


class BoxCoxTransform(BaseEstimator, TransformerMixin):
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


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
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
    def __init__(
        self, components=__common_elements__ | __common_oxides__, inverse=False
    ):
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
    def __init__(self, components=common_oxides()):
        self.columns = components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_cols = [i for i in X.columns if i in self.columns]
        out = X.loc[:, out_cols]
        return out


class ElementSelector(BaseEstimator, TransformerMixin):
    def __init__(self, components=common_elements()):
        self.columns = components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_cols = [i for i in X.columns if i in self.columns]
        out = X.loc[:, out_cols]
        return out


class REESelector(BaseEstimator, TransformerMixin):
    def __init__(self, components=REE()):
        components = [i for i in components if not i == "Pm"]
        self.columns = components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_cols = [i for i in self.columns if i in X.columns]
        out = X.loc[:, out_cols]
        return out


class Devolatilizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, exclude=["H2O", "H2O_PLUS", "H2O_MINUS", "CO2", "LOI"], renorm=True
    ):
        self.exclude = [i.upper() for i in exclude]
        self.renorm = renorm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        exclude = [i for i in X.columns if i.upper() in self.exclude]
        return devolatilise(X, exclude=exclude, renorm=self.renorm)


class RedoxAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, to_oxidised=False, renorm=True, total_suffix="T"):
        self.to_oxidised = to_oxidised
        self.renorm = renorm
        self.total_suffix = total_suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return recalculate_redox(
            X,
            to_oxidised=self.to_oxidised,
            renorm=self.renorm,
            total_suffix=self.total_suffix,
        )


class ElementAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, renorm=True, form="oxide"):
        self.renorm = renorm
        self.form = form

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        multiple_entries = check_multiple_cation_inclusion(X)

        for el in multiple_entries:
            X = aggregate_cation(X, el, form=self.form)
        return X


class PdUnion(BaseEstimator, TransformerMixin):
    def __init__(self, estimators: list = []):
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        parts = []
        for est in self.estimators:
            if isinstance(est, pd.DataFrame):
                parts.append(est)
            elif isinstance(est, TransformerMixin) or isinstance(est, BaseEstimator):
                if hasattr(est, "fit"):
                    parts.append(est.fit_transform(X))
                else:
                    parts.append(est.transform(X))
            else:  # e.g. Numpy array, try to convert to dataframe
                parts.append(pd.DataFrame(est))

        columns = []
        idxs = []
        for p in parts:
            columns += [i for i in p.columns if not i in columns]
            idxs.append(p.index.size)

        # check the indexes are all the same length
        assert all([idx == idxs[0] for idx in idxs])

        out = pd.DataFrame(columns=columns)
        for p in parts:
            out[p.columns] = p

        return out


class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, norm_to="Chondrite_PON", exclude=["Pm", "Eu", "Ce"], params=None, degree=5
    ):
        self.norm_to = norm_to
        self.ree = [i for i in REE() if not i in exclude]
        self.radii = np.array(get_ionic_radii(self.ree, charge=3, coordination=8))
        self.exclude = exclude
        if params is None:
            self.degree = degree
            self.params = OP_constants(self.radii, degree=self.degree)
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
            self.params = OP_constants(self.radii, degree=self.degree)

        return lambda_lnREE(
            X,
            norm_to=self.norm_to,
            params=self.params,
            degree=self.degree,
            exclude=self.exclude,
        )


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

        Need to use masks to avoid SoftImpute returning 0. where it cannot impute.
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
