import matplotlib.pyplot as plt


#@update_docstring_references
def TAS(ax=None, relim=True, color="k", **kwargs):
    """
    Adds the TAS diagram [#ref_1]_ to an axes.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template onto.

    References
    -----------
    .. [#ref_1] Pearce J. A. (2008) Geochemical fingerprinting of oceanic basalts
                with applications to ophiolite classification and the search for
                Archean oceanic crust. Lithos 100, 14–48.
                doi: {pearce2008}

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
    """
