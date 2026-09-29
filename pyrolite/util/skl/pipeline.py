from ..log import Handle

logger = Handle(__name__)

try:
    import sklearn.svm
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin,
        TransformerMixin,
    )
    from sklearn.model_selection import (
        BaseCrossValidator,
        GridSearchCV,
        StratifiedKFold,
    )
    from sklearn.pipeline import make_pipeline

    HAVE_SKL = True
except ImportError:
    HAVE_SKL = False


if HAVE_SKL:
    from collections.abc import Callable
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd

    from ..plot import save_figure
    from .vis import plot_confusion_matrix, plot_gs_results

    def fit_save_classifier(
        clf: RegressorMixin | ClassifierMixin,
        X_train: np.ndarray[tuple[int, int]] | pd.DataFrame,
        y_train: np.ndarray[tuple[int]] | pd.Series,
        directory: str | Path = ".",
        name: str = "clf",
        extension: str = ".joblib",
    ) -> BaseEstimator:
        """
        Fit and save a classifier model. Also save relevant metadata where possible.

        Parameters
        -----------
        clf : sklearn.base.BaseEstimator
            Classifier or gridsearch.
        X_train : numpy.ndarray | pandas.DataFrame
            Training data.
        y_train : numpy.ndarray | pandas.Series
            Training true classes.
        directory : str | pathlib.Path
            Path to the save directory.
        name : str
            Name of the classifier.
        extension : str
            Extension to give the saved classifier pickled witih joblib.

        Returns
        --------
        clf : sklearn.base.BaseEstimator
            Fitted classifier.
        """
        clf_dir = Path(directory) / name
        if not clf_dir.exists():
            clf_dir.mkdir(parents=True)

        clf.fit(X_train, y_train)
        fpath = (clf_dir / name).with_suffix(extension)
        # save metadata
        if isinstance(
            X_train, pd.DataFrame
        ):  # save the features used in the model for ref
            components = [str(i) for i in X_train.columns]
            with open(
                str(clf_dir / f"{name}_features.txt"), "w", encoding="utf-8"
            ) as fp:
                fp.write(",".join(components))
        _ = joblib.dump(clf, str(fpath), compress=9)
        return clf

    def classifier_performance_report(
        clf: RegressorMixin | ClassifierMixin,
        X_test: np.ndarray[tuple[int, int]] | pd.DataFrame,
        y_test: np.ndarray[tuple[int]] | pd.Series,
        classes: list[str] | None = None,
        directory: str | Path = ".",
        name: str = "clf",
    ):
        """
        Output a performance report for a classifier. Currently outputs the overall
        classification score, a confusion matrix and where relevant an indication of
        variation seen across the gridsearch (currently only possible for 2D searches).

        Parameters
        ----------
        clf : sklearn.base.BaseEstimator | `sklearn.model_selection.GridSearchCV`
            Classifer or gridsearch.
        X_test : numpy.ndarray | pandas.DataFrame
            Input data for testing.
        y_test : numpy.ndarray | pandas.Series
            Labelled/target data for testing.
        classes : list
            Names of classes.
        directory : str | pathlib.Path
            Path to the save directory.
        name : str
            Name of the classifier.

        Returns
        --------
        clf : sklearn.base.BaseEstimator
            Fitted classifier.
        """
        if classes is None:
            classes = []
        clf_dir = Path(directory) / name
        if not clf_dir.exists():
            clf_dir.mkdir(parents=True)

        if isinstance(clf, GridSearchCV):
            gs = True
            gs = clf
            params = gs.best_params_
            clf = gs.best_estimator_
        score = clf.score(X_test, y_test)
        with open(str(clf_dir / f"scores_{name}.txt"), "a") as fp:
            line = f"Score: {score:01.3g}"
            if gs:  # add the gridsearch parameters
                line += "\t{}\n".format(
                    "\t".join([f"{k}:{v:01.2g}" for k, v in params.items()])
                )
            fp.write(line)

        cmax = plot_confusion_matrix(
            clf, X_test, y_test, normalize=True, classes=classes
        )
        save_figure(cmax.figure, save_at=clf_dir, name=f"confusion_matrix_{name}")

        try:
            gsax = plot_gs_results(gs)
            save_figure(gsax.figure, save_at=clf_dir, name=f"gridsearchresults_{name}")
        except ValueError:  # only one param changed in gridsearch
            pass
        return clf

    def SVC_pipeline(
        sampler: TransformerMixin | None = None,
        balance: bool = True,
        transform: TransformerMixin | None = None,
        scaler: TransformerMixin | None = None,
        kernel: str | Callable = "rbf",
        decision_function_shape: str = "ovo",
        probability: bool = False,
        cv: BaseCrossValidator | None = None,
        param_grid: dict | None = None,
        n_jobs: int = 4,
        verbose: int = 10,
        cache_size: float = 500,
        **kwargs,
    ) -> GridSearchCV:
        """
        A convenience function for constructing a Support Vector Classifier pipeline.

        Parameters
        -----------
        sampler : sklearn.base.TransformerMixin
            Resampling transformer.
        balance : bool
            Whether to balance the class weights for the classifier.
        transform : sklearn.base.TransformerMixin
            Preprocessing transformer.
        scaler : sklearn.base.TransformerMixin
            Scale transformer.
        kernel : str | Callable
            Name of kernel to use for the support vector classifier
            (`'linear'|'rbf'|'poly'|'sigmoid'`). Optionally, a custom
            kernel function can be supplied (see :mod:`sklearn` docs for more info).
        decision_function_shape : str
            Shape of the decision function surface. `'ovo'` one-vs-one classifier
            of libsvm (returning classification of shape
            `(samples, classes*(classes-1)/2))`, or the default `'ovr'
            one-vs-rest classifier which will return classification estimation shape of
            `(samples, classes)`.
        probability : bool
            Whether to implement Platt-scaling to enable probability estimates.
            This must be enabled prior to calling fit, and will slow down that method.
        cv : int | sklearn.model_selection.BaseSearchCV
            Cross validation search. If an integer `k` is provided, results in
            default `k`-fold cross validation. Optionally, if a
            `sklearn.model_selection.BaseSearchCV` instance is provided, it will be
            used directly (enabling finer control, e.g. over sorting/shuffling etc).
        param_grid : dict
            Dictionary reprenting a parameter grid for the support vector classifier.
            Typically contains 1D arrays of grid indicies for :class:`~sklearn.svm.SVC`
            parameters each prefixed with `svc__` (e.g.
            `dict(svc__gamma=np.logspace(-1, 3, 5), svc__C=np.logspace(-0.5, 2, 5))`.
        n_jobs : int
            Number of processors to use for the SVC construction. Note that providing
            `n_jobs = -1` will use all available processors.
        verbose : int
            Level of verbosity for the pipeline logging output.
        cache_size  : float
            Specify the size of the kernel cache (in MB).

        Returns
        -------
        gs : sklearn.model_selection.GridSearchCV
            Gridsearch object containing the results of the SVC training across the
            parameter grid. Access the best estimator with `gs.best_estimator_`
            and its parameters with `gs.best_params_`.

        Notes
        -----
        See also: `sklearn.svm.SVC`
        """
        if param_grid is None:
            param_grid = {}
        classifier_kwargs = {
            "kernel": kernel,
            "probability": probability,
            "decision_function_shape": decision_function_shape,
            "cache_size": cache_size,
            "gamma": "scale",  # suppress warnings; 'auto' deprecated, likely changes with gs
            **kwargs,
        }

        if balance:
            classifier_kwargs.update({"class_weight": "balanced"})

        stages = []
        if sampler is not None:
            stages.append(sampler)

        if transform is not None:
            stages.append(transform)

        if scaler is not None:  # scaler should be the second last item added
            stages.append(scaler)

        stages.append(sklearn.svm.SVC(**classifier_kwargs))  # add the classifier itself
        pipe = make_pipeline(*stages)

        if cv is None:
            cv = StratifiedKFold(n_splits=10, shuffle=True)
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose
        )
        return gs

    class PdUnion(BaseEstimator, TransformerMixin):
        def __init__(self, estimators: list | None = None):
            if estimators is None:
                estimators = []
            self.estimators = estimators

        def fit(
            self,
            X: np.ndarray[tuple[int, int]] | pd.DataFrame,
            y: np.ndarray[tuple[int]] | pd.Series | None = None,
        ):
            return self

        def transform(self, X: np.ndarray[tuple[int, int]] | pd.DataFrame):
            assert isinstance(X, pd.DataFrame)
            parts = []
            for est in self.estimators:
                if isinstance(est, pd.DataFrame):
                    parts.append(est)
                elif isinstance(est, (TransformerMixin, BaseEstimator)):
                    if hasattr(est, "fit"):
                        parts.append(est.fit_transform(X))
                    else:
                        parts.append(est.transform(X))
                else:  # e.g. Numpy array, try to convert to dataframe
                    parts.append(pd.DataFrame(est))

            columns = []
            idxs = []
            for p in parts:
                columns += [i for i in p.columns if i not in columns]
                idxs.append(p.index.size)

            # check the indexes are all the same length
            assert all(idx == idxs[0] for idx in idxs)

            out = pd.DataFrame(columns=columns)
            for p in parts:
                out[p.columns] = p

            return out
