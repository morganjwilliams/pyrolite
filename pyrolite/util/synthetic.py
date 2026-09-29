"""
Utility functions for creating synthetic (geochemical) data.
"""

import numpy as np
import pandas as pd

from ..comp.codata import ILR, inverse_ILR
from ..geochem.ind import REE, get_ionic_radii
from ..geochem.norm import get_reference_composition
from ..util.lambdas.eval import get_function_components
from .log import Handle

logger = Handle(__name__)


def random_cov_matrix(
    dim: int,
    sigmas: list[float] | np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
    validate: bool = False,
    seed: int | None = None,
) -> np.ndarray[tuple[int, int]]:
    """
    Generate a random covariance matrix which is symmetric positive-semidefinite.

    Parameters
    -----------
    dim : int
        Dimensionality of the covariance matrix.
    sigmas : numpy.ndarray
        Optionally specified sigmas for the variables, 1D.
    validate : bool
        Whether to validate output.

    Returns
    --------
    numpy.ndarray
        Covariance matrix of shape `(dim, dim)`.

    Todo
    -----
    * Implement a characteristic scale for the covariance matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    # create a matrix of correlation coefficients
    corr = (np.random.rand(dim, dim) - 0.5) * 2  # values between -1 and 1
    corr[np.tril_indices(dim)] = corr.T[np.tril_indices(dim)]  # lower=upper
    corr[np.arange(dim), np.arange(dim)] = 1.0

    sigmas: np.ndarray[tuple[int, int], np.dtype[np.floating]] = (
        np.ones(dim, dtype=np.float32) if sigmas is None else np.array(sigmas)
    ).reshape(1, dim)

    cov = sigmas.T @ sigmas  # multiply by ~ variance
    cov *= corr
    cov = np.sign(cov) * np.sqrt(np.abs(cov) / dim)
    cov = cov @ cov.T

    if validate:
        try:
            assert (cov == cov.T).all()
            # eig = np.linalg.eigvalsh(cov)
            for i in range(dim):
                assert np.linalg.det(cov[0:i, 0:i]) > 0.0  # sylvesters criterion
        except AssertionError:  # not symmetrical covariance matrix
            cov = random_cov_matrix(dim, validate=validate)
    return cov


def random_composition(
    size: int = 1000,
    D: int = 4,
    mean: list | np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
    cov: np.ndarray[tuple[int, int], np.dtype[np.floating]] | None = None,
    propnan: float = 0.1,
    missing_columns: int | tuple | None = None,
    missing: str | None = None,
    seed: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Generate a simulated random unimodal compositional dataset,
    optionally with missing data.

    Parameters
    -----------
    size : int
        Size of the dataset.
    D : int
        Dimensionality of the dataset.
    mean : numpy.ndarray
        Optional specification of mean composition.
    cov : numpy.ndarray
        Optional specification of covariance matrix (in log space).
    propnan : float
        Proportion of missing values in the output dataset.
    missing_columns : int | tuple
        Specification of columns to be missing. If an integer is specified,
        interpreted to be the number of columns containin missing data (at a proportion
        defined by `propnan`). If a tuple or list, the specific columns to contain
        missing data.
    missing : str
        Missingness pattern.
        If not `None`, one of `"MCAR", "MAR", "MNAR"`.

        * If `missing = "MCAR"`, data will be missing at random.
        * If `missing = "MAR"`, data will be missing with some relationship to other parameters.
        * If `missing = "MNAR"`, data will be thresholded at some lower bound.

    seed : int
        Random seed to use, optionally specified.

    Returns
    --------
    numpy.ndarray
        Simulated dataset with missing values.

    Todo
    ------
    * Add feature to translate rough covariance in D to logcovariance in D-1
    * Update the ``missing = "MAR"`` example to be more realistic/variable.
    """
    data = None
    # dimensions
    if mean is None and cov is None:
        pass
    elif mean is None and cov is not None:
        D = cov.shape[0] + 1
    elif cov is None:
        mean = np.array(mean)
        D = mean.size
    else:  # both defined
        mean, cov = np.array(mean), np.array(cov)
        assert mean.size == cov.shape[0] + 1
        D = mean.size
        mean = mean.reshape(1, -1)

    if seed is not None:
        np.random.seed(seed)
    # mean
    if mean is None:
        if D > 1:
            mean = np.random.randn(D - 1).reshape(1, -1)
        else:  # D == 1, return a 1D series
            data = np.exp(np.random.randn(size).reshape(size, D))
            data /= np.nanmax(data)
            return data
    else:
        mean = ILR(mean.reshape(1, D)).reshape(
            1, -1
        )  # ILR of a (1, D) mean to (1, D-1)

    # covariance
    if cov is None:
        if D != 1:
            cov = random_cov_matrix(
                D - 1, sigmas=np.abs(mean) * 0.1, seed=seed
            )  # 10% sigmas
        else:
            cov = np.array([[1]])

    assert cov.shape in [(D - 1, D - 1), (1, 1)]

    if size == 1:  # single sample
        data = inverse_ILR(mean).reshape(size, D)

    # if the covariance matrix isn't for the logspace data, we'd have to convert it
    if data is None:
        data = inverse_ILR(
            np.random.multivariate_normal(mean.reshape(D - 1), cov, size=size)
        ).reshape(size, D)

    if missing_columns is None:
        nancols = (
            np.random.choice(
                range(data.shape[1] - 1), size=int(data.shape[1] - 1), replace=False
            )
            + 1
        )
    elif isinstance(missing_columns, int):  # number of columns specified
        nancols = (
            np.random.choice(
                range(data.shape[1] - 1), size=missing_columns, replace=False
            )
            + 1
        )
    else:  # tuples, list etc
        nancols = missing_columns

    if missing is not None:
        if missing == "MCAR":
            for i in nancols:
                data[np.random.randint(size, size=int(propnan * size)), i] = np.nan
        elif missing == "MAR":
            thresholds = np.percentile(data[:, nancols], propnan * 100, axis=0)[
                np.newaxis, :
            ]
            # should update this such that data are proportional to other variables
            # potentially just by rearranging the where statement
            data[:, nancols] = np.where(
                data[:, nancols]
                < np.tile(thresholds, size).reshape(size, len(nancols)),
                np.nan,
                data[:, nancols],
            )
        elif missing == "MNAR":
            thresholds = np.percentile(data[:, nancols], propnan * 100, axis=0)[
                np.newaxis, :
            ]
            data[:, nancols] = np.where(
                data[:, nancols]
                < np.tile(thresholds, size).reshape(size, len(nancols)),
                np.nan,
                data[:, nancols],
            )
        else:
            msg = "Provide a value for missing in {}".format({"MCAR", "MAR", "MNAR"})
            raise NotImplementedError(msg)

    return data


def normal_frame(
    columns: list | None = None,
    size: int = 10,
    mean: list | np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    r"""
    Creates a pandas.DataFrame with samples from a single multivariate-normal
    distributed composition.

    Parameters
    ----------
    columns : list
        List of columns to use for the dataframe. These won't have any direct impact
        on the data returned, and are only for labelling.
    size : int
        Index length for the dataframe.
    mean : numpy.ndarray | list
        Optional specification of mean composition.

    Returns
    --------
    pandas.DataFrame

    Notes
    -----
    See also: :func:`~pyrolite.util.synthetic.random_composition`.
    """
    if columns is None:
        columns = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
    return pd.DataFrame(
        columns=columns,
        data=random_composition(size=size, D=len(columns), mean=mean, **kwargs),
    )


def normal_series(
    index: list | pd.Index | None = None,
    mean: list | np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
    **kwargs,
) -> pd.Series:
    """
    Creates a pandas.Series with a single sample from a single multivariate-normal
    distributed composition.

    Parameters
    ------------
    index : list
        List of indexes for the series. These won't have any direct impact
        on the data returned, and are only for labelling.
    mean : numpy.ndarray, `None`
        Optional specification of mean composition.

    Returns
    --------
    pandas.Series

    Notes
    ------
    See also: :func:`~pyrolite.util.synthetic.random_composition`.
    """
    if index is None:
        index = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
    return pd.Series(
        random_composition(size=1, D=len(index), mean=mean, **kwargs).flatten(),
        index=index,
    )


def example_spider_data(
    start: str = "EMORB_SM89",
    norm_to: str | None = "PM_PON",
    size: int = 120,
    noise_level: float = 0.5,
    offsets: dict[str, float] | None = None,
    units: str = "ppm",
) -> pd.DataFrame:
    """
    Generate some random data for demonstrating spider plots.

    By default, this generates a composition based around EMORB, normalised
    to Primitive Mantle.

    Parameters
    -----------
    start : str
        Composition to start with.
    norm_to : str
        Composition to normalise to. Can optionally specify `None`.
    size : int
        Number of observations to include (index length).
    noise_level : float
        Log-units of noise (1sigma).
    offsets : dict
        Dictionary of offsets in log-units (in log units).
    units : str
        Units to use before conversion. Should have no effect other than reducing
        calculation times if `norm_to` is `None`.

    Returns
    --------
    df : pandas.DataFrame
        Dataframe of example synthetic data.
    """

    ref: pd.Series = get_reference_composition(start)
    ref.set_units(units)
    df: pd.Series | pd.DataFrame = ref.comp.pyrochem.compositional
    if norm_to is not None:
        df = df.pyrochem.normalize_to(norm_to, units=units)
    start: pd.Series | pd.DataFrame = np.log(df)
    nindex = df.columns.size if isinstance(df, pd.DataFrame) else df.index.size

    y: np.ndarray[tuple[int, int], np.dtype] = np.tile(start.values, size).reshape(
        size, nindex
    )

    y += np.random.normal(0, noise_level / 2.0, size=(size, nindex))  # noise
    y += np.random.normal(0, noise_level, size=(1, size)).T  # random pattern offset

    syn_df = pd.DataFrame(
        y, columns=df.columns if isinstance(df, pd.DataFrame) else df.index
    )
    if offsets is not None:
        for element, offset in offsets.items():
            syn_df[element] += offset  # significant offset for e.g. Eu anomaly
    syn_df = np.exp(syn_df)
    return syn_df


def example_patterns_from_parameters(
    fit_parameters: np.ndarray,
    radii: np.ndarray | None = None,
    n: int = 100,
    proportional_noise: float = 0.15,
    includes_tetrads: bool = False,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """ """
    fit_parameters = np.tile(fit_parameters, n).reshape(n, -1)
    if radii is None:
        radii = get_ionic_radii(REE(), coordination=8, charge=3)

    _, _, components = get_function_components(radii, fit_tetrads=includes_tetrads)
    pattern_df = pd.DataFrame(
        np.exp(fit_parameters @ np.array(components)), columns=columns
    )
    # add some random correlated proportional noise
    sz = len(radii)
    cov = np.zeros((sz, sz))
    for offset in np.arange(-sz + 1, sz):
        vals = np.ones(sz - np.abs(offset)) * np.abs(sz - np.abs(offset)) / sz
        cov += np.diag(vals**2, offset)
    noise = 1 + proportional_noise * np.random.multivariate_normal(
        np.zeros(sz), cov, size=pattern_df.shape[0]
    )
    pattern_df *= noise
    return pattern_df
