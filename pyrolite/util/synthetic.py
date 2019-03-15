import numpy as np
import pandas as pd
from ..comp.codata import ilr, inverse_ilr


def random_cov_matrix(dim, validate=False):
    """
    Generate a random covariance matrix which is symmetric positive-semidefinite.

    Parameters
    -----------
    dim : :class:`int`
        Dimensionality of the covariance matrix.
    validate : :class:`bool`
        Whether to validate output.

    Returns
    --------
    :class:`numpy.ndarray`
        Covariance matrix of shape :code:`(dim, dim)`.
    """
    cov = np.random.randn(dim, dim)
    cov = np.dot(cov, cov.T)
    if validate:
        try:
            assert (cov == cov.T).all()
            #eig = np.linalg.eigvalsh(cov)
            for i in range(dim):
                assert np.linalg.det(cov[0:i, 0:i]) > 0.0  # sylvesters criterion
        except:
            cov = random_cov_matrix(dim, validate=validate)
    return cov


def random_composition(size=1000, D=4, mean=None, cov=None, propnan=0.1, missing=None):
    """
    Generate a simulated random unimodal compositional dataset,
    optionally with missing data.

    Parameters
    -----------
    size : :class:`int`
        Size of the dataset.
    D : :class:`int`
        Dimensionality of the dataset.
    E : :class:`numpy.ndarray`, :code:`None`
        Optional specification of mean composition.
    cov : :class:`numpy.ndarray`, :code:`None`
        Optional specification of covariance matrix (in log space).
    propnan : :class:`float`, [0, 1)
        Proportion of missing values in the output dataset.
    missing : :class:`str`, :code:`None`
        If not :code:`None`, a string in :code:`{"MCAR", "MAR", "MNAR"}`.

            * If :code:`missing = "MCAR"``, data will be missing at random.
            * If :code:`missing = "MAR"``, data will be missing with some relationship to other parameters.
            * If :code:`missing = "MNAR"``, data will be thresholded at some lower bound.

    Returns
    --------
    :class:`numpy.ndarray`
        Simulated dataset with missing values.

    Todo
    ------
        * Update the `:code:`missing = "MAR"`` example to be more realistic/variable.
    """
    if mean is None:
        mean = np.random.randn(D - 1)
    else:
        D = mean.size
        mean = ilr(mean.reshape(1, D)).flatten()
        mean += np.random.randn(*mean.shape) * 0.01  # minor noise

    if cov is None:
        cov = random_cov_matrix(D - 1)
    else:
        assert cov.shape == (D - 1, D - 1)
        # if the covariance matrix isn't for the logspace data, we'd have to convert it

    data = inverse_ilr(np.random.multivariate_normal(mean, cov, size=size))

    if missing is not None:
        if missing == "MCAR":
            nnan = int(propnan * size)
            for _ in range(nnan):
                for i in range(data.shape[1] - 1):
                    data[np.random.randint(size), i + 1] = np.nan
        elif missing == "MAR":
            thresholds = np.percentile(data, propnan * 100, axis=0)

        elif missing == "MNAR":
            thresholds = np.percentile(data, propnan * 100, axis=0)
            data[:, 1:] = np.where(
                data[:, 1:] < np.tile(thresholds[1:], size).reshape(size, D - 1),
                np.nan,
                data[:, 1:],
            )
        else:
            msg = "Provide a value for missing in {}".format(
                set(["MCAR", "MAR", "MNAR"])
            )
            raise NotImplementedError(msg)

    return data


def test_df(cols=["SiO2", "CaO", "MgO", "FeO", "TiO2"], index_length=10, **kwargs):
    """
    Creates a pandas.DataFrame with random data.
    """
    return pd.DataFrame(
        columns=cols, data=random_composition(size=index_length, D=len(cols), **kwargs)
    )


def test_ser(index=["SiO2", "CaO", "MgO", "FeO", "TiO2"], **kwargs):
    """
    Creates a pandas.Series with random data.
    """
    return pd.Series(
        random_composition(size=1, D=len(index), **kwargs).flatten(), index=index
    )
