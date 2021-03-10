"""
Utility functions for creating synthetic (geochemical) data.
"""
import numpy as np
import pandas as pd
from ..comp.codata import ILR, inverse_ILR
from ..geochem.norm import get_reference_composition
from ..geochem.ind import get_ionic_radii, REE
from ..util.lambdas.eval import get_function_components
from .meta import get_additional_params
from .log import Handle

logger = Handle(__name__)

def random_cov_matrix(dim, sigmas=None, validate=False, seed=None):
    """
    Generate a random covariance matrix which is symmetric positive-semidefinite.

    Parameters
    -----------
    dim : :class:`int`
        Dimensionality of the covariance matrix.
    sigmas : :class:`numpy.ndarray`
        Optionally specified sigmas for the variables.
    validate : :class:`bool`
        Whether to validate output.

    Returns
    --------
    :class:`numpy.ndarray`
        Covariance matrix of shape :code:`(dim, dim)`.

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

    if sigmas is None:
        sigmas = np.ones(dim).reshape(1, dim)
    else:
        sigmas = np.array(sigmas)
        sigmas = sigmas.reshape(1, dim)

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
    size=1000,
    D=4,
    mean=None,
    cov=None,
    propnan=0.1,
    missing_columns=None,
    missing=None,
    seed=None,
):
    """
    Generate a simulated random unimodal compositional dataset,
    optionally with missing data.

    Parameters
    -----------
    size : :class:`int`
        Size of the dataset.
    D : :class:`int`
        Dimensionality of the dataset.
    mean : :class:`numpy.ndarray`, :code:`None`
        Optional specification of mean composition.
    cov : :class:`numpy.ndarray`, :code:`None`
        Optional specification of covariance matrix (in log space).
    propnan : :class:`float`, [0, 1)
        Proportion of missing values in the output dataset.
    missing_columns : :class:`int` | :class:`tuple`
        Specification of columns to be missing. If an integer is specified,
        interpreted to be the number of columns containin missing data (at a proportion
        defined by `propnan`). If a tuple or list, the specific columns to contain
        missing data.
    missing : :class:`str`, :code:`None`
        Missingness pattern.
        If not :code:`None`, one of :code:`"MCAR", "MAR", "MNAR"`.

            * If :code:`missing = "MCAR"`, data will be missing at random.
            * If :code:`missing = "MAR"`, data will be missing with some relationship to other parameters.
            * If :code:`missing = "MNAR"`, data will be thresholded at some lower bound.

    Returns
    --------
    :class:`numpy.ndarray`
        Simulated dataset with missing values.

    Todo
    ------
        * Add feature to translate rough covariance in D to logcovariance in D-1
        * Update the `:code:`missing = "MAR"`` example to be more realistic/variable.
    """
    data = None
    # dimensions
    if mean is None and cov is None:
        pass
    elif mean is None:
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
            msg = "Provide a value for missing in {}".format(
                set(["MCAR", "MAR", "MNAR"])
            )
            raise NotImplementedError(msg)

    return data


def normal_frame(
    columns=["SiO2", "CaO", "MgO", "FeO", "TiO2"], size=10, mean=None, **kwargs
):
    r"""
    Creates a :class:`pandas.DataFrame` with samples from a single multivariate-normal
    distributed composition.

    Parameters
    ----------
    columns : :class:`list`
        List of columns to use for the dataframe. These won't have any direct impact
        on the data returned, and are only for labelling.
    size : :class:`int`
        Index length for the dataframe.
    mean : :class:`numpy.ndarray`, :code:`None`
        Optional specification of mean composition.
    {otherparams}

    Returns
    --------
    :class:`pandas.DataFrame`
    """
    return pd.DataFrame(
        columns=columns,
        data=random_composition(size=size, D=len(columns), mean=mean, **kwargs),
    )


def normal_series(index=["SiO2", "CaO", "MgO", "FeO", "TiO2"], mean=None, **kwargs):
    """
    Creates a :class:`pandas.Series` with a single sample from a single multivariate-normal
    distributed composition.

    Parameters
    ------------
    index : :class:`list`
        List of indexes for the series. These won't have any direct impact
        on the data returned, and are only for labelling.
    mean : :class:`numpy.ndarray`, :code:`None`
        Optional specification of mean composition.
    {otherparams}

    Returns
    --------
    :class:`pandas.Series`
    """
    return pd.Series(
        random_composition(size=1, D=len(index), mean=mean, **kwargs).flatten(),
        index=index,
    )


def example_spider_data(
    start="EMORB_SM89",
    norm_to="PM_PON",
    size=120,
    noise_level=0.5,
    offsets=None,
    units="ppm",
):
    """
    Generate some random data for demonstrating spider plots.

    By default, this generates a composition based around EMORB, normalised
    to Primitive Mantle.

    Parameters
    -----------
    start : :class:`str`
        Composition to start with.
    norm_to : :class:`str`
        Composition to normalise to. Can optionally specify :code:`None`.
    size : :class:`int`
        Number of observations to include (index length).
    noise_level : :class:`float`
        Log-units of noise (1sigma).
    offsets : :class:`dict`
        Dictionary of offsets in log-units (in log units).
    units : :class:`str`
        Units to use before conversion. Should have no effect other than reducing
        calculation times if `norm_to` is :code:`None`.

    Returns
    --------
    df : :class:`pandas.DataFrame`
        Dataframe of example synthetic data.
    """

    ref = get_reference_composition(start)
    ref.set_units(units)
    df = ref.comp.pyrochem.compositional
    if norm_to is not None:
        df = df.pyrochem.normalize_to(norm_to, units=units)
    start = df.applymap(np.log)
    nindex = df.columns.size

    y = np.tile(start.values, size).reshape(size, nindex)
    y += np.random.normal(0, noise_level / 2.0, size=(size, nindex))  # noise
    y += np.random.normal(0, noise_level, size=(1, size)).T  # random pattern offset

    syn_df = pd.DataFrame(y, columns=df.columns)
    if offsets is not None:
        for element, offset in offsets.items():
            syn_df[element] += offset  # significant offset for e.g. Eu anomaly
    syn_df = syn_df.applymap(np.exp)
    return syn_df


def example_patterns_from_parameters(
    fit_parameters,
    radii=None,
    n=100,
    proportional_noise=0.15,
    includes_tetrads=False,
    columns=None,
):

    """
    """
    fit_parameters = np.tile(fit_parameters, n).reshape(n, -1)
    if radii is None:
        radii = get_ionic_radii(REE(), coordination=8, charge=3)

    names, _, components = get_function_components(radii, fit_tetrads=includes_tetrads)
    pattern_df = pd.DataFrame(
        np.exp(fit_parameters @ np.array(components)), columns=columns
    )
    # add some random correlated proportional noise
    sz = len(radii)
    cov = np.zeros((sz, sz))
    for offset in np.arange(-sz + 1, sz):
        vals = np.ones(sz - np.abs(offset)) * np.abs((sz - np.abs(offset))) / sz
        cov += np.diag(vals ** 2, offset)
    noise = 1 + proportional_noise * np.random.multivariate_normal(
        np.zeros(sz), cov, size=pattern_df.shape[0]
    )
    pattern_df *= noise
    return pattern_df


_add_additional_parameters = True

normal_frame.__doc__ = normal_frame.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            random_composition, header="Other Parameters", indent=8, subsections=True
        ),
    ][_add_additional_parameters]
)

normal_series.__doc__ = normal_series.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            random_composition, header="Other Parameters", indent=8, subsections=True
        ),
    ][_add_additional_parameters]
)
