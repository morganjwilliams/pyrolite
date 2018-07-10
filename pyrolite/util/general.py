import numpy as np


def flatten_dict(d, climb=False):
    """
    Flattens a nested dictionary.

    Parameters
    ----------
    climb: True | False
        Whether to keep trunk or leaf-values, for items with the same key.
    """
    def _items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    sk = subkey # key + "." + subkey
                    sv = subvalue
                    yield  sk, sv
            else:
                yield key, value


    if climb:
        # We prioritise trunk values
        items = list(_items())
    else:
        # We prioritise the leaf values
        items = reversed(list(_items()))
    # By reversing the order, items lower in the dictionary are presereved
    return dict(items)


def swap_item(list: list, pull: str, push: str):
    return [push if i == pull else i for i in list]


def on_finite(arr, f):
    """
    Calls a function on an array ignoring np.nan and +/- np.inf.
    """
    ma = np.isfinite(arr)
    return f(arr[ma])
