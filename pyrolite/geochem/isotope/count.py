import numpy as np
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

def deadtime_correction(data, deadtime):
    """
    Apply a deadtime correction to count data.

    Parameters
    -------------
    data : :class:`numpy.ndarray`
        Array of count data.
    deadtime : :class:`float`
        Deadtime in nanoseconds.

    Returns
    --------
    :class:`numpy.ndarray`
        Corrected count data.
    """
    dt = deadtime / 10 ** 9  # nanoseconds
    # Need to check for overflow and divide by zero errors
    if np.mean(data) < 10000000:
        data = data * np.exp(
            data * dt * np.exp(data * dt * np.exp(data * dt * np.exp(data * dt)))
        )
    else:
        data = data / (1 - data * dt)

    return data
