import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ternary
from pyrolite.geochem import common_elements


def spiderplot(df, ax=None, components:list=None, plot=True, fill=False, **kwargs):
    """
    Plots spidergrams for trace elements data.
    By using separate lines and scatterplots, values between two null-valued
    items are still presented. Might be able to speed up the lines
    with a matplotlib.collections.LineCollection

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    components: list, None
        Elements or compositional components to plot.
    plot: boolean, True
        Whether to plot lines and markers.
    fill:
        Whether to add a patch representing the full range.
    style:
        Styling keyword arguments to pass to matplotlib.
    """

    try:
        assert plot or fill
    except:
        raise AssertionError('Please select to either plot values or fill between ranges.')
    sty = {}
    # Some default values
    sty['marker'] = kwargs.get('marker') or 'D'
    sty['color'] = kwargs.get('color') or kwargs.get('c') or None
    sty['alpha'] = kwargs.get('alpha') or kwargs.get('a') or 1.
    if sty['color'] is None:
        del sty['color']

    components = components or [el for el in common_elements(output='str')
                                if el in df.columns]
    assert len(components) != 0
    c_indexes = np.arange(len(components))

    ax = ax or plt.subplots(1, figsize=(len(components)*0.25, 4))[1]

    if plot:
        ls = ax.plot(c_indexes,
                     df[components].T.values.astype(np.float),
                     **sty)

        sty['s'] = kwargs.get('markersize') or kwargs.get('s') or 5.
        if sty.get('color') is None:
            sty['color'] = ls[0].get_color()
        sc = ax.scatter(np.tile(c_indexes, (df[components].index.size,1)).T,
                        df[components].T.values.astype(np.float), **sty)

    for s_item in ['marker', 's']:
        if s_item in sty:
            del sty[s_item]

    if fill:
        mins, maxs = df[components].min(axis=0), df[components].max(axis=0)
        ax.fill_between(c_indexes, mins, maxs, **sty)

    ax.set_xticks(c_indexes)
    ax.set_xticklabels(components, rotation=60)
    ax.set_yscale('log')
    ax.set_xlabel('Element')

    unused_keys = [i for i in kwargs if i not in list(sty.keys()) + \
                  ['alpha', 'a', 'c', 'color', 'marker']]
    if len(unused_keys):
        warnings.warn(f'Styling not yet implemented for:{unused_keys}')


def ternaryplot(df, ax=None, components=None, **kwargs):
    """
    Plots scatter ternary diagrams, using a wrapper around the
    python-ternary library (gh.com/marcharper/python-ternary).

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    components: list, None
        Elements or compositional components to plot.
    """

    try:
        if not len(df.columns)==3:
            assert len(components)==3
        components = components or df.columns.values
    except:
        raise AssertionError('Please either suggest three elements or a 3-element dataframe.')

    # Some default values
    scale = kwargs.get('scale') or 100.
    figsize = kwargs.get('size') or 8.
    gridsize = kwargs.get('gridsize') or 10.
    fontsize = kwargs.get('fontsize') or 12.

    sty = {}
    sty['marker'] = kwargs.get('marker') or 'D'
    sty['color'] = kwargs.get('color') or kwargs.get('c') or '0.5'
    sty['label'] = kwargs.get('label') or None
    sty['alpha'] = kwargs.get('alpha') or kwargs.get('a') or 1.

    ax = ax or plt.subplots(1, figsize=(figsize, figsize* 3**0.5 * 0.5))[1]
    d1 = ax.__dict__.copy()

     # Checking if there's already a ternary axis
    tax = getattr(ax, 'tax', None) or ternary.figure(ax=ax, scale=scale)[1]

    # Set attribute for future reference
    ax.tax = tax
    points = df.loc[:, components].div(df.loc[:, components].sum(axis=1), axis=0).values * scale
    if points.any():
        tax.scatter(points, **sty)

    if sty['label'] is not None:
        tax.legend(frameon=False,)

    # Check if there's already labels
    if not len(tax._labels.keys()):
        tax.left_axis_label(components[2], fontsize=fontsize)
        tax.bottom_axis_label(components[0], fontsize=fontsize)
        tax.right_axis_label(components[1], fontsize=fontsize)

        tax.gridlines(multiple=gridsize, color='k', alpha=0.5)
        tax.ticks(axis='lbr', linewidth=1, multiple=gridsize)
        tax.boundary(linewidth=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return tax
