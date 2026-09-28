from ..log import Handle

logger = Handle(__file__)
try:
    from sklearn.decomposition import PCA
    from sklearn.utils.validation import check_is_fitted

    HAVE_SKL = True
except ImportError:
    HAVE_SKL = False

if HAVE_SKL:
    import numpy as np
    import pandas as pd

    def get_PCA_component_labels(
        pca_object: PCA,
        input_columns: list[str],
        max_components: int = 4,
        fmt_string: str = "PCA_{number}({label})",
    ):
        """
        Generate labels for PCA components based on the magnitude and sign of
        the contributing features.

        Parameters
        ----------
        pca_object : sklearn.decomposition.PCA
            Fitted PCA object.
        input_columns : list
            List of columns which are input into the PCA decompositon (these are
            not preserved by default by the object).
        max_components : int
            Maximum components to include in each label.
        fmt_string : str
            Formatting string for labels, optionally accepting keyword-based labels
            for 'number' and 'label' (e.g. `'PCA_{number}({label})'`).

        Returns
        -------
        list
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
                    for el, v in row.iloc[
                        np.argsort(np.abs(row))[::-1][:max_components]
                    ]
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
