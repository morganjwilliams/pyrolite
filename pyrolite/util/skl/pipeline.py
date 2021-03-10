import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from ..plot import save_figure
from ..meta import get_additional_params
from ..log import Handle

logger = Handle(__name__)

try:
    import sklearn.svm
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)

try:
    from imblearn.pipeline import make_pipeline
except ImportError:
    msg = "imbalanced-learn not installed"
    logger.warning(msg)
    from sklearn.pipeline import make_pipeline  # fallback to default skl

from .vis import plot_confusion_matrix, plot_gs_results


def fit_save_classifier(
    clf, X_train, y_train, dir=".", name="clf", extension=".joblib"
):
    """
    Fit and save a classifier model. Also save relevant metadata where possible.

    Parameters
    -----------
    clf : :class:`sklearn.base.BaseEstimator`
        Classifier or gridsearch.
    X_train : :class:`numpy.ndarray` | :class:`pandas.DataFrame`
        Training data.
    y_train : :class:`numpy.ndarray` | :class:`pandas.Series`
        Training true classes.
    dir : :class:`str` | :class:`pathlib.Path`
        Path to the save directory.
    name : :class:`str`
        Name of the classifier.
    extension : :class:`str`
        Extension to give the saved classifier pickled witih joblib.

    Returns
    --------
    clf : :class:`sklearn.base.BaseEstimator`
        Fitted classifier.
    """
    clf_dir = Path(dir) / name
    if not clf_dir.exists():
        clf_dir.mkdir(parents=True)

    clf.fit(X_train, y_train)
    fpath = (clf_dir / name).with_suffix(extension)
    # save metadata
    if isinstance(X_train, pd.DataFrame):  # save the features used in the model for ref
        components = [str(i) for i in X_train.columns]
        with open(
            str(clf_dir / "{}_features.txt".format(name)), "w", encoding="utf-8"
        ) as fp:
            fp.write(",".join(components))
    _ = joblib.dump(clf, str(fpath), compress=9)
    return clf


def classifier_performance_report(clf, X_test, y_test, classes=[], dir=".", name="clf"):
    """
    Output a performance report for a classifier. Currently outputs the overall
    classification score, a confusion matrix and where relevant an indication of
    variation seen across the gridsearch (currently only possible for 2D searches).

    Parameters
    ----------
    clf : :class:`sklearn.base.BaseEstimator` | `sklearn.model_selection.GridSearchCV`
        Classifer or gridsearch.
    X_test:

    """
    clf_dir = Path(dir) / name
    if not clf_dir.exists():
        clf_dir.mkdir(parents=True)

    if isinstance(clf, GridSearchCV):
        gs = True
        gs = clf
        params = gs.best_params_
        clf = gs.best_estimator_
    score = clf.score(X_test, y_test)
    with open(str(clf_dir / "scores_{}.txt".format(name)), "a") as fp:
        line = "Score: {:01.3g}".format(score)
        if gs:  # add the gridsearch parameters
            line += "\t{}\n".format(
                "\t".join(["{}:{:01.2g}".format(k, v) for k, v in params.items()])
            )
        fp.write(line)

    cmax = plot_confusion_matrix(clf, X_test, y_test, normalize=True, classes=classes)
    save_figure(cmax.figure, save_at=clf_dir, name="confusion_matrix_{}".format(name))

    try:
        gsax = plot_gs_results(gs)
        save_figure(
            gsax.figure, save_at=clf_dir, name="gridsearchresults_{}".format(name)
        )
    except ValueError:  # only one param changed in gridsearch
        pass
    return clf


def SVC_pipeline(
    sampler=None,
    balance=True,
    transform=None,
    scaler=None,
    kernel="rbf",
    decision_function_shape="ovo",
    probability=False,
    cv=StratifiedKFold(n_splits=10, shuffle=True),
    param_grid={},
    n_jobs=4,
    verbose=10,
    cache_size=500,
    **kwargs
):
    """
    A convenience function for constructing a Support Vector Classifier pipeline.

    Parameters
    -----------
    sampler : :class:`sklearn.base.TransformerMixin`
        Resampling transformer.
    balance : :class:`bool`
        Whether to balance the class weights for the classifier.
    transform : :class:`sklearn.base.TransformerMixin`
        Preprocessing transformer.
    scaler : :class:`sklearn.base.TransformerMixin`
        Scale transformer.
    kernel : :class:`str` | :class:`callable`
        Name of kernel to use for the support vector classifier
        (:code:`'linear'|'rbf'|'poly'|'sigmoid'`). Optionally, a custom
        kernel function can be supplied (see :mod:`sklearn` docs for more info).
    decision_function_shape : :class:`str`, :code:`'ovo' or 'ovr'`
        Shape of the decision function surface. :code:`'ovo'` one-vs-one classifier
        of libsvm (returning classification of shape
        :code:`(samples, classes*(classes-1)/2))`, or the default :code:`'ovr'
        one-vs-rest classifier which will return classification estimation shape of
        :code:`(samples, classes)`.
    probability : :class:`bool`
        Whether to implement Platt-scaling to enable probability estimates.
        This must be enabled prior to calling fit, and will slow down that method.
    cv : :class:`int` | :class:`sklearn.model_selection.BaseSearchCV`
        Cross validation search. If an integer :code:`k` is provided, results in
        default :code:`k`-fold cross validation. Optionally, if a
        :class:`sklearn.model_selection.BaseSearchCV` instance is provided, it will be
        used directly (enabling finer control, e.g. over sorting/shuffling etc).
    param_grid : :class:`dict`
        Dictionary reprenting a parameter grid for the support vector classifier.
        Typically contains 1D arrays of grid indicies for :func:`~sklearn.svm.SVC`
        parameters each prefixed with :code:`svc__` (e.g.
        :code:`dict(svc__gamma=np.logspace(-1, 3, 5), svc__C=np.logspace(-0.5, 2, 5))`.
    n_jobs : :class:`int`
        Number of processors to use for the SVC construction. Note that providing
        :code:`n_jobs = -1` will use all available processors.
    verbose : :class:`int`
        Level of verbosity for the pipeline logging output.
    cache_size  : :class:`float`
        Specify the size of the kernel cache (in MB).

    {otherparams}

    Returns
    -------
    gs : :class:`sklearn.model_selection.GridSearchCV`
        Gridsearch object containing the results of the SVC training across the
        parameter grid. Access the best estimator with :code:`gs.best_estimator_`
        and its parameters with :code:`gs.best_params_`.
    """
    classifier_kwargs = {
        "kernel": kernel,
        "probability": probability,
        "decision_function_shape": decision_function_shape,
        "cache_size": cache_size,
        "gamma": "scale",  # suppress warnings; 'auto' deprecated, likely changes with gs
        **kwargs,
    }

    if balance:
        classifier_kwargs.update(dict(class_weight="balanced"))

    stages = []
    if sampler is not None:
        stages.append(sampler)

    if transform is not None:
        stages.append(transform)

    if scaler is not None:  # scaler should be the second last item added
        stages.append(scaler)

    stages.append(sklearn.svm.SVC(**classifier_kwargs))  # add the classifier itself
    pipe = make_pipeline(*stages)
    gs = GridSearchCV(
        estimator=pipe, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose
    )
    return gs


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


_add_additional_parameters = True
SVC_pipeline.__doc__ = SVC_pipeline.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            SVC_pipeline,
            sklearn.svm.SVC,
            indent=4,
            header="Other Parameters",
            subsections=True,
        ),
    ][_add_additional_parameters]
)
