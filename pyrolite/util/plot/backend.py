import warnings

import pandas as pd

from ..log import Handle
from ...plot import pyroplot_matplotlib

logger = Handle(__name__)

BACKEND = "matplotlib"

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
            logger.debug(
                "{} plotting backend to {}".format(
                    "Reverting" if revert else "Setting", backend
                )
            )
            BACKEND = backend
        else:
            raise NotImplementedError("Backend {} not available.")


class Backend:
    """
    Backend context manager for pyrolite plotting.
    """

    def __init__(self, backend=None):
        self.start = BACKEND
        self.backend = backend
        set_plotting_backend(self.backend)
        with warnings.catch_warnings():
            # can get invalid values which raise RuntimeWarnings
            # consider changing to np.errstate
            warnings.simplefilter("ignore", category=UserWarning)
            pd.api.extensions.register_dataframe_accessor("pyroplot")(
                _backend_accessors.get(self.backend)
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # only reset when changed in the first place?
        if self.backend is not None:
            set_plotting_backend(self.start, revert=True)
