import itertools
import numpy as np
import scipy.stats
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.colors
from pyrolite.util.plot import DEFAULT_DISC_COLORMAP
from pyrolite.util.meta import inargs, subkwargs
from ..log import Handle

logger = Handle(__name__)

try:
    from sklearn.metrics import confusion_matrix
    import sklearn.datasets
    import sklearn.manifold
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)


def plot_confusion_matrix(
    *args,
    classes=[],
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    norm=matplotlib.colors.Normalize(vmin=0, vmax=1.0),
    ax=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if len(args) == 1:
        conf_matrix = args[0]
    else:
        clf, X_test, y_test = args
        conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
        if not classes:
            if hasattr(args[0], "classes_"):
                classes = list(args[0].classes_)

    if not classes:
        classes = np.arange(conf_matrix.shape[0])

    if normalize:
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )

    if ax is None:
        fig, ax = plt.subplots(1)

    im = ax.imshow(conf_matrix, interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))

    fmt = ".2f" if normalize else "d"
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
    plt.tight_layout()
    return ax


def plot_gs_results(gs, xvar=None, yvar=None):
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
                (yvar, yy) = [(k, v) for (k, v) in grid_items if not k == xvar][0]
            else:
                yy = gs.param_grid[yvar]
                (xvar, xx) = [(k, v) for (k, v) in grid_items if not k == yvar][0]
    xx, yy = np.array(xx), np.array(yy)
    other_keys = [i for i in labels if i not in [xvar, yvar]]
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

    locmax = np.where(results == np.nanmax(results))
    x, y = locmax
    ax.scatter(x, y, marker="D", s=100, c="k")
    return ax


def alphas_from_multiclass_prob(probs, method="entropy", alpha=1.0):
    """
    Take an array of multiclass probabilities and map to an alpha variable.

    Parameters
    -----------
    probs : :class:`numpy.ndarray`
        Multiclass probabilities with shape (nsamples, nclasses).

    method : :class:`str`, :code:`entropy` | :code:`kl_div`
        Method for mapping probabilities to alphas.
    alpha : :class:`float`
        Optional specification of overall maximum alpha value.

    Returns
    ----------
    a : :class:`numpy.ndarray`
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
    X,
    Y,
    mapping=None,
    ax=None,
    cmap=None,
    alpha=1.0,
    s=10,
    alpha_method="entropy",
    **kwargs
):
    """
    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Coordinates in multidimensional space.
    Y : :class:`numpy.ndarray` | :class:`sklearn.base.BaseEstimator`
        An array of targets, or a method to obtain such an array of targets
        via :func:`Y.predict`. Transformers with probabilistic output
        (via :func:`Y.predict_proba`) will have these probability estimates accounted
        for via the alpha channel.
    mapping : :class:`numpy.ndarray` | :class:`~sklearn.base.TransformerMixin`
        Mapped points or transformer to create mapped points.
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot on.
    cmap : :class:`matplotlib.cm.ListedColormap`
        Colormap to use for the classification visualisation (ideally this should be
        a discrete colormap unless the classes are organised ).
    alpha : :class:`float`
        Coefficient for alpha.
    alpha_method : :code:`'entropy' or 'kl_div'`
        Method to map class probabilities to alpha. :code:`'entropy'` uses a measure of
        entropy relative to null-scenario of equal distribution across classes, while
        :code:`'kl_div'` calculates the information gain relative to the same
        null-scenario.

    Returns
    -------
    ax : :class:`~matplotlib.axes.Axes`
        Axes on which the mapping is plotted.
    tfm : :class:`~sklearn.base.BaseEstimator`
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
            kw = dict(n_components=2, metric=True)
        elif mapping.lower() == "isomap":
            # not necessarily consistent orientation, but consistent shape
            cls = sklearn.manifold.Isomap
            kw = dict(n_components=2)
        elif mapping.lower() == "tsne":
            # likely need to optimise!
            cls = sklearn.manifold.TSNE
            kw = dict(n_components=2)
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
        fig, ax = plt.subplots(1, **kwargs)

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
            cmap=None
        else:
            c = Y.predict(X)
            cmap = cmap or DEFAULT_DISC_COLORMAP

    ax.scatter(*mapped.T, c=c, s=s, edgecolors="none", cmap=cmap)
    return ax, tfm, mapped
