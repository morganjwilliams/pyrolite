from ..log import Handle

logger = Handle(__file__)
try:
    import sklearn.datasets
    import sklearn.manifold
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV

    HAVE_SKL = True
except ImportError:
    HAVE_SKL = False

if HAVE_SKL:
    import itertools

    import matplotlib.axes
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.special
    import scipy.stats

    from pyrolite.util.meta import inargs, subkwargs
    from pyrolite.util.plot import DEFAULT_DISC_COLORMAP

    def plot_confusion_matrix(
        *args,
        ax: matplotlib.axes.Axes | None = None,
        classes: list[str] | None = None,
        class_order: list[str] | None = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        cmap: str | matplotlib.colors.Colormap = plt.cm.Blues,
        norm: matplotlib.colors.Norm | None = None,
        xlabelrotation: float | None = None,
        fmt: str | None = None,
    ) -> matplotlib.axes.Axes:
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        Parameters
        ------------
        args : tuple
            Data to evaluate and visualise a confusion matrix:

            * A single confusion matrix (n x n)
            * A tuple of (y_test, y_predict)
            * A tuple of (classifier_model, X_test, y_test)

        ax : matplotlib.axes.Axes
            Axis to plot on, if one exists.
        classes: list
            List of class names to use as labels, and for ordering (see below).
            This should match the order contained within the model. Where a
            classifier model is passed, the classes will be directly extracted.
        class_order : list
            List of classes in the desired order along the axes. Should match
            the supplied classes where classes are given, or integer indicies
            for where no named classes are given.
        normalize : bool
            Whether to normalize the counts for the confusion matrix to the
            sum of all cases (i.e. be between 0 and 1).
        title : str
            Title for the axes.
        cmap : str | matplotlib.color.Colormap
            Colormap for the visualisation of the confusion matrix.
        norm : bool
            Normalization for the colormap visualisation across the confusion matrix.
        xlabelrotation : float
            Rotation in degrees for the xaxis labels.
        fmt : str
            Format to use for filling values.

        Returns
        --------
        ax : matplotlib.axes.Axes
        """
        if classes is None:
            classes = []
        if len(args) == 1:
            conf_matrix = args[0]
        elif len(args) in [2, 3]:
            if len(args) == 2:
                y_test, y_predict = args
            else:
                clf, X_test, y_test = args
                y_predict = clf.predict(X_test)
                if not classes and hasattr(args[0], "classes_"):
                    classes = list(args[0].classes_)
            conf_matrix = confusion_matrix(y_test, y_predict)
        else:
            raise NotImplementedError(
                "Supply either i) a confusion matrix, ii) the test and"
                " predict arrays or iii) the classifier, X_test and y_test arrays."
            )

        if not classes:
            classes = np.arange(conf_matrix.shape[0])

        if class_order is not None:
            assert all(c in classes for c in class_order) and all(
                c in class_order for c in classes
            )
            _classes = list(classes)  # for .index
            class_indexes = np.array([_classes.index(c) for c in class_order])
            conf_matrix = conf_matrix[np.ix_(class_indexes, class_indexes)]
            classes = class_order

        if normalize:
            conf_matrix = (
                conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
            )
            if norm is None:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
        else:
            # the colormap will need to be normalized across the count range
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(conf_matrix))

        if ax is None:
            _fig, ax = plt.subplots(1)

        im = ax.imshow(conf_matrix, interpolation="none", cmap=cmap, norm=norm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(classes))

        fmt = fmt if fmt is not None else (".2f" if normalize else "d")
        threshold = conf_matrix.max() / 2.0
        for i, j in itertools.product(
            range(conf_matrix.shape[0]), range(conf_matrix.shape[1])
        ):
            ax.text(
                j,
                i,
                format(conf_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > threshold else "black",
            )

        ax.set(
            ylabel="True",
            xlabel="Predicted",
            xticks=tick_marks,
            yticks=tick_marks,
            xticklabels=classes,
            yticklabels=classes,
        )
        if xlabelrotation is not None:
            plt.setp(ax.get_xticklabels(), rotation=xlabelrotation)
        ax.grid(False)
        plt.tight_layout()
        return ax

    def plot_gs_results(
        gs: GridSearchCV, xvar: str | None = None, yvar: str | None = None
    ) -> matplotlib.axes.Axes:
        """Plots the results from a GridSearch showing location of optimum in 2D."""
        labels = gs.param_grid.keys()
        grid_items = list(gs.param_grid.items())
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
                    (yvar, yy) = next((k, v) for (k, v) in grid_items if k != xvar)
                else:
                    yy = gs.param_grid[yvar]
                    (xvar, xx) = next((k, v) for (k, v) in grid_items if k != yvar)
        xx, yy = np.array(xx), np.array(yy)
        other_keys = [i for i in labels if i not in [xvar, yvar]]
        if other_keys:
            pass
        else:
            results = np.array(gs.cv_results_["mean_test_score"]).reshape(
                xx.size, yy.size
            )
        _fig, ax = plt.subplots(1)
        ax.imshow(results.T, cmap=plt.cm.Blues)

        ax.set(
            xlabel=xvar,
            ylabel=yvar,
            xticks=np.arange(len(xx)),
            yticks=np.arange(len(yy)),
            xticklabels=[f"{i:01.2g}" for i in xx],
            yticklabels=[f"{i:01.2g}" for i in yy],
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.invert_yaxis()

        locmax = np.where(results == np.nanmax(results))
        x, y = locmax
        ax.scatter(x, y, marker="D", s=100, c="k")
        return ax

    def alphas_from_multiclass_prob(
        probs: np.ndarray[tuple[int, int]], method: str = "entropy", alpha: float = 1.0
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """
        Take an array of multiclass probabilities and map to an alpha variable.

        Parameters
        -----------
        probs : numpy.ndarray
            Multiclass probabilities with shape (nsamples, nclasses).

        method : str, `entropy` | `kl_div`
            Method for mapping probabilities to alphas.
        alpha : float
            Optional specification of overall maximum alpha value.

        Returns
        ----------
        a : numpy.ndarray
            Alpha values for each sample with shape (nsamples, 1).
        """
        netzero = 1.0 / probs.shape[1] * np.ones(probs.shape[1])
        if method == "entropy":
            # uniform distribution has maximum entropy
            max_H = scipy.stats.entropy(netzero)
            H = np.apply_along_axis(scipy.stats.entropy, 1, probs)
            min_H = np.min(H, axis=0)
            rel_H = (H - min_H) / (max_H - min_H)  # between zero and one
            a = 1.0 - rel_H
            a *= alpha
        else:
            # alpha as sum of information gain
            a = np.apply_along_axis(scipy.special.kl_div, 1, probs, netzero).sum(axis=1)
            a = a / np.max(a, axis=0)
            a *= alpha
        return a

    def plot_mapping(
        X: np.ndarray[tuple[int, int]],
        Y: np.ndarray[tuple[int, int]] | np.ndarray[tuple[int]] | BaseEstimator,
        mapping: np.ndarray[tuple[int, int]] | TransformerMixin | None = None,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str | matplotlib.colors.Colormap | None = None,
        alpha: float = 1.0,
        s: float = 10,
        alpha_method: str = "entropy",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Parameters
        ----------
        X : numpy.ndarray
            Coordinates in multidimensional space.
        Y : numpy.ndarray | sklearn.base.BaseEstimator
            An array of targets, or a method to obtain such an array of targets
            via `Y.predict`. Transformers with probabilistic output
            (via `Y.predict_proba`) will have these probability estimates accounted
            for via the alpha channel.
        mapping : numpy.ndarray | `~sklearn.base.TransformerMixin`
            Mapped points or transformer to create mapped points.
        ax : matplotlib.axes.Axes
            Axes to plot on.
        cmap : `matplotlib.cm.ListedColormap`
            Colormap to use for the classification visualisation (ideally this should be
            a discrete colormap unless the classes are organised ).
        alpha : float
            Coefficient for alpha.
        alpha_method : `'entropy' or 'kl_div'`
            Method to map class probabilities to alpha. `'entropy'` uses a measure of
            entropy relative to null-scenario of equal distribution across classes, while
            `'kl_div'` calculates the information gain relative to the same
            null-scenario.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes on which the mapping is plotted.
        tfm : sklearn.base.BaseEstimator
            Fitted mapping transform.

        Todo
        ------

        * Option to generate colors for individual classes

          This could be based on the distances between their centres in
          multidimensional space (or low dimensional mapping of this space),
          enabling a continuous (n-dimensional) colormap to be used
          to show similar classes, in addition to classification confidence.
        """
        X_ = X.copy()  # avoid modifying input array
        if mapping is None:
            tfm = sklearn.manifold.MDS
            tfm_kwargs = {k: v for k, v in kwargs.items() if inargs(k, tfm)}
            tfm = tfm(n_components=2, metric=True, **tfm_kwargs)
            mapped = tfm.fit_transform(X_)
        elif isinstance(mapping, str):
            if mapping.lower() == "mds":
                cls = sklearn.manifold.MDS
                kw = {"n_components": 2, "metric": True}
            elif mapping.lower() == "isomap":
                # not necessarily consistent orientation, but consistent shape
                cls = sklearn.manifold.Isomap
                kw = {"n_components": 2}
            elif mapping.lower() == "tsne":
                # likely need to optimise!
                cls = sklearn.manifold.TSNE
                kw = {"n_components": 2}
            else:
                raise NotImplementedError
            tfm = cls(**{**kw, **subkwargs(kwargs, cls)})
            mapped = tfm.fit_transform(X_)
        elif isinstance(
            mapping, (sklearn.base.TransformerMixin, sklearn.base.BaseEstimator)
        ):  # manifold transforms can be either
            tfm = mapping
            mapped = tfm.fit_transform(X_)
        else:  # mapping is already performedata, expect a numpy.ndarray
            mapped = mapping
            tfm = None
        assert mapped.shape[0] == X_.shape[0]

        if ax is None:
            _fig, ax = plt.subplots(1, **kwargs)

        if isinstance(Y, (np.ndarray, list)):
            c = Y  # need to encode alpha here
        elif isinstance(Y, (sklearn.base.BaseEstimator)):
            # need to split this into  multiple methods depending on form of classifier
            if hasattr(Y, "predict_proba"):
                classes = Y.predict(X_)
                cmap = cmap or DEFAULT_DISC_COLORMAP
                c = cmap(classes)
                ps = Y.predict_proba(X_)
                a = alphas_from_multiclass_prob(ps, method=alpha_method, alpha=alpha)
                c[:, -1] = a
                cmap = None
            else:
                c = Y.predict(X)
                cmap = cmap or DEFAULT_DISC_COLORMAP

        ax.scatter(*mapped.T, c=c, s=s, edgecolors="none", cmap=cmap)
        return ax, tfm, mapped
