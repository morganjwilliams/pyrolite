import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from ..log import Handle

logger = Handle(__file__)


def get_PCA_component_labels(
    pca_object, input_columns, max_components=4, fmt_string="PCA_{number}({label})"
):
    """
    Generate labels for PCA components based on the magnitude and sign of
    the contributing features.

    Parameters
    ----------
    pca_object : :class:`sklearn.decomposition.PCA`
        Fitted PCA object.
    input_columns : :class:`list`
        List of columns which are input into the PCA decompositon (these are
        not preserved by default by the object).
    max_components : :class:`int`
        Maximum components to include in each label.
    fmt_string : :class:`str`
        Formatting string for labels, optionally accepting keyword-based labels
        for 'number' and 'label' (e.g. :code:`'PCA_{number}({label})'`).

    Returns
    -------
    :class:`list`
        List of labels for the PCA components.
    """
    try:
        assert isinstance(pca_object, PCA)
    except AssertionError:
        raise NotImplementedError(
            "Object supplied needs to be an instance of sklearn.decompositon.PCA."
        )
    check_is_fitted(pca_object)

    labels = [
        "".join(
            [
                "{}{}".format(["-", "+"][int(np.sign(v) > 0)], el)
                for el, v in row.iloc[np.argsort(np.abs(row))[::-1][:max_components]]
                .to_dict()
                .items()
            ]
        )
        for idx, row in pd.DataFrame(
            pca_object.components_, columns=input_columns
        ).iterrows()
    ]
    if fmt_string is not None:
        labels = [
            fmt_string.format(number=ix + 1, label=label)
            for ix, label in enumerate(labels)
        ]
    return labels
