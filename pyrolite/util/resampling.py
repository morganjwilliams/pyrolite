"""
Utilities for (weighted) bootstrap resampling applied to geoscientific point-data.
"""
import numpy as np
import pandas as pd
from .meta import subkwargs
from .spatial import great_circle_distance, _get_sqare_grid_segment_indicies
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def _segmented_univariate_distance_matrix(A, B, metric, dtype="float32", segs=10):
    max_size = np.max([a.shape[0] for a in [A, B]])
    dist = np.zeros((max_size, max_size), dtype=dtype)  # full matrix
    for ix_s, ix_e, iy_s, iy_e in _get_sqare_grid_segment_indicies(max_size, segs):
        dist[ix_s:ix_e, iy_s:iy_e] = metric(
            A[ix_s:ix_e][:, np.newaxis], B[iy_s:iy_e][np.newaxis, :],
        )
    return dist


def get_distance_matrix(a, b=None, metric=None):
    """
    Get a distance matrix for a single column or array of values.

    Parameters
    -----------
    a, b : :class:`float` | :class:`numpy.ndarray`
        Points or arrays to calculate distance between. If only one array is
        specified, a full distance matrix (i.e. calculate a point-to-point distance
        for every combination of points) will be returned.
    metric
        Callable function f(a, b) from which to derive a distance metric.

    Returns
    -------
    :class:`numpy.ndarray`
        2D distance matrix.
    """
    if metric is None:
        metric = lambda a, b: np.abs(a - b)

    a = np.atleast_1d(np.array(a).astype(np.float))
    matrix = False
    if b is not None:
        b = np.atleast_1d(np.array(b).astype(np.float))
    else:
        matrix = True
        b = a.copy()
    return _segmented_univariate_distance_matrix(a, b, metric)


def get_spatiotemporal_resampling_weights(
    df,
    spatial_norm=1.8,
    temporal_norm=38,
    latlong_names=["Latitude", "Longitude"],
    age_name="Age",
    max_memory_fraction=0.25,
    **kwargs
):
    """
    Takes a dataframe with lat, long and age and returns a sampling weight for each
    sample which is essentailly the inverse of the mean distance to other samples.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe to calculate weights for.
    spatial_norm : :class:`float`
        Normalising constant for spatial measures (1.8 arc degrees).
    temporal_norm : :class:`float`
        Normalising constant for temporal measures (38 Mya).
    latlong_names : :class:`list`
        List of column names referring to latitude and longitude.
    age_name : :class:`str`
        Column name corresponding to geological age or time.
    max_memory_fraction : :class:`float`
        Constraint to switch to calculating mean distances where :code:`matrix=True`
        and the distance matrix requires greater than a specified fraction of total
        avaialbe physical memory. This is passed on to
        :func:`~pyrolite.util.spatial.great_circle_distance`.

    Returns
    -------
    :class:`pandas.Series`

    Notes
    ------
    This function is equivalent to Eq(1) from Keller and Schone:

    \\
    W_i \propto 1 \Big / \sum_{j=1}^{n} \Big ( \frac{1}{((z_i - z_j)/a)^2 + 1} + \frac{1}{((t_i - t_j)/b)^2 + 1} \Big )
    \\
    """

    weights = pd.Series(index=df.index, dtype="float")
    z = great_circle_distance(
        df[[*latlong_names]],
        absolute=False,
        max_memory_fraction=max_memory_fraction,
        **subkwargs(kwargs, great_circle_distance)
    )  # angular distances
    t = get_distance_matrix(df[age_name])
    # where the distances are zero, these weights will go to inf
    # instead we replace with the smallest non-zero distance/largest non-inf
    # inverse weight
    norm_inverse_distances = 1.0 / ((z / spatial_norm) ** 2 + 1)
    norm_inverse_distances[~np.isfinite(norm_inverse_distances)] = 1

    norm_inverse_time = 1.0 / ((t / temporal_norm) ** 2 + 1)
    norm_inverse_time[~np.isfinite(norm_inverse_time)] = 1

    return 1.0 / np.sum(norm_inverse_distances + norm_inverse_time, axis=0)


def add_age_noise(
    df,
    min_sigma=50,
    noise_level=1.0,
    age_name="Age",
    min_age_name="MinAge",
    max_age_name="MaxAge",
):
    # try and get age uncertainty

    # otherwise get age min age max
    # get age uncertainties
    age_certainty = np.abs(df[max_age_name] - df[min_age_name]) / 2  # half bin width
    age_certainty[~np.isfinite(age_certainty) | age_certainty < min_sigma] = min_sigma
    # add noise to ages
    age_noise = np.random.randn(df.index.size) * noise_level * age_certainty.values
    df[age_name] += age_noise
    return df


def bootstrap_resample(
    df,
    weights=None,
    niter=10000,
    categories=None,
    transform=np.log,
    add_gaussian_parameter_noise=True,
    add_gaussian_age_noise=True,
    metrics=["mean", "var"],
    noise_level=0.5,
    age_name="Age",
    latlong_names=["Latitude", "Longitude"],
    **kwargs
):
    """
    Resample and aggregate metrics from a dataframe, optionally aggregating by a given
    set of categories. Formulated specifically for dealing with resampling to address
    uneven sampling density in space and particularly geological time.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe to resample.
    weights : :class:`numpy.ndarray` | :class:`pandas.Series`
        Array of weights for resampling, if precomputed.
    niter : :class:`int`
        Number of resampling iterations. This will be the minimum index size of the output
        metric dataframes.
    categories : :class:`list` | :class:`numpy.ndarray` | :class:`pandas.Series`
        List of sample categories to group the ouputs by, which has the same size as the
        dataframe index.
    transform
        Callable function to transform input data prior to aggregation functions. Note
        that the outputs will need to be inverse-transformed.
    add_gaussian_parameter_noise : :class:`bool`
        Whether to add gaussian noise to the input dataset parameters.
    add_gaussian_age_noise : :class:`bool`
        Whether to add gassian noise to the input dataset ages, where present.
    metrics : :class:`list`
        List of metrics to use for dataframe aggregation.
    noise_level : :class:`float`
        Multiplier for the random gaussian noise added to the dataset and ages.
    age_name : :class:`str`
        Column name for geological age.
    latlong_names : :class:`list`
        Column names for latitude and longitude, or equvalent orthogonal spherical
        spatial measures.

    Returns
    --------
    :class:`dict`
        Dictionary of aggregated Dataframe(s) indexed by statistical metrics. If
        categories are specified, the dataframe(s) will have a hierarchical index of
        :code:`categories, iteration`.
    """
    if weights is None:
        weights = get_spatiotemporal_resampling_weights(
            df,
            age_name=age_name,
            latlong_names=latlong_names,
            **subkwargs(kwargs, get_spatiotemporal_resampling_weights)
        )

    # get the subset of parameters to be resampled, removing spatial and age names
    # and only taking numeric data
    subset = [
        c
        for c in df.columns
        if c not in [[i for i in df.columns if age_name in i], *latlong_names]
        and np.issubdtype(df.dtypes[c], np.number)
    ]

    def _metric_name(metric):
        return repr(metric).replace("'", "")

    metric_data = {_metric_name(metric): [] for metric in metrics}
    for repeat in range(niter):
        smpl = df.sample(weights=weights, frac=1, replace=True)

        if add_gaussian_age_noise:
            smpl = add_age_noise(
                smpl,
                min_sigma=50,
                age_name=age_name,
                noise_level=noise_level,
                **subkwargs(kwargs, add_age_noise)
            )

        if transform is not None:
            smpl[subset] = smpl[subset].apply(transform, axis="index")

        # add random noise within uncertainty bounds - this is essentially smoothing

        # try to get uncertainties for the data, otherwise use standard deviations?
        if add_gaussian_parameter_noise:
            smpl[subset] += (
                smpl[subset].std().values[np.newaxis, :]
                * np.random.randn(*smpl[subset].shape)
                * noise_level
            )
            # smpl[subset] = smpl[subset].pyrocomp.renormalise()

        if categories is not None:
            for metric in metrics:
                metric_data[_metric_name(metric)].append(
                    smpl[subset].groupby(categories).agg(metric)
                )

        else:
            for metric in metrics:
                metric_data[_metric_name(metric)].append(smpl[subset].agg(metric))

    if categories is not None:
        return {
            metric: pd.concat(data, keys=range(niter), names=["Iteration"])
            .swaplevel(0, 1)
            .sort_index()
            for metric, data in metric_data.items()
        }

    else:
        return {metric: pd.DataFrame(data) for metric, data in metric_data.items()}
