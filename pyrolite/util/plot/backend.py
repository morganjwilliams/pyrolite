import warnings

import pandas as pd

from ..log import Handle
from ...plot import pyroplot_matplotlib

logger = Handle(__name__)

BACKEND = "matplotlib"


def _get_backend():
    return BACKEND


_backend_accessors = {"matplotlib": pyroplot_matplotlib}
try:
    import plotly.graph_objects as go

    HAVE_PLOTLY = True

    from .plotly import pyroplot_plotly

    _backend_accessors["plotly"] = pyroplot_plotly

except ImportError:
    HAVE_PLOTLY = False


def set_plotting_backend(backend=None, revert=False):
    if backend is not None:
        if backend in ["matplotlib", "plotly"]:
            global BACKEND
            if BACKEND != backend:
                logger.debug(
                    "{} plotting backend to {}".format(
                        "Reverting" if revert else "Setting", backend
                    )
                )
                BACKEND = backend
        else:
            raise NotImplementedError("Backend {} not available.")


class pyroplot(object):
    def __init__(self, obj):
        """
        Custom dataframe accessor for pyrolite plotting.

        Notes
        -----
            This accessor enables the coexistence of array-based plotting functions and
            methods for pandas objects. This enables some separation of concerns.
        """
        self._obj = obj

    def scatter(self, *args, backend=None, **kwargs):
        with Backend(backend=backend):
            return _backend_accessors[_get_backend()](self._obj).scatter(
                *args, **kwargs
            )

    def spider(self, *args, backend=None, **kwargs):
        with Backend(backend=backend):
            return _backend_accessors[_get_backend()](self._obj).spider(*args, **kwargs)


with warnings.catch_warnings():
    # can get invalid values which raise RuntimeWarnings
    # consider changing to np.errstate
    warnings.simplefilter("ignore", category=UserWarning)
    pd.api.extensions.register_dataframe_accessor("pyroplot")(
        pyroplot  # _backend_accessors.get(self.backend)
    )


class Backend:
    """
    Backend context manager for pyrolite plotting.
    """

    def __init__(self, backend=None):
        self.start = _get_backend()
        self.backend = backend
        set_plotting_backend(self.backend)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # only reset when changed in the first place?
        if self.backend is not None:
            set_plotting_backend(self.start, revert=True)
