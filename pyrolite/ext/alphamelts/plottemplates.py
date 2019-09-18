"""
Templates for creating plots from alphaMELTS tables.

Todo
------

    * accept an 'ax' parameter
    * accept 1-n plots (currently has to be multiple.)
    * consistent styles for specific phases
    * split out some of the code for setting up plots
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from ...util.plot import __DEFAULT_DISC_COLORMAP__, proxy_line
from ...util.text import titlecase
from ...geochem.ind import common_oxides

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def phase_linestyle(phasename):
    """
    Method for generating linestyles for delineating sequential phase names
    (e.g. olivine_0, olivine_1) based on their names.

    Parameters
    -----------
    phasename : :class:`str`
        Phase name for which to generate a line style.

    Returns
    ---------
    linestyle : :class:`str`
        Line style for the phase name.
    """
    if "_" in phasename:
        return ["-", "--", ":", "-."][int(phasename[-1])]
    else:
        return "-"


def plot_phasetable(
    summary,
    table="phasevol",
    xvar="Temperature",
    plotswide=1,
    figsize=None,
    yscale="linear",
):
    """
    Plot a particular phase table per-experiment across indiviudal axes.

    Parameters
    -----------
    summary : :class:`dict`
        Dictionary of experiment result outputs indexed by title.
    table : :class:`str`
        Which table to plot from the experiment output.
    xvar : :class:`str`
        Variable to use for the independent axis.
    plotswide : :class:`int`
        With of the figure, as a number of axes.
    figsize : :class:`tuple`
        Size of the figure, optional.
    yscale : :class:`str`
        Scale to use for the y-axis.

    Returns
    ---------
    :class:`matplotlib.figure.Figure`
        Figure on which the results are plotted.
    """
    all_phases = set()
    for s in summary.values():
        all_phases = all_phases | s["phases"]

    colors = {k: v for k, v in zip(all_phases, range(len(all_phases)))}

    nkeys = len(summary.keys())
    plotshigh = nkeys // plotswide
    if nkeys % plotswide:
        plotshigh += 1
    fig, ax = plt.subplots(
        plotshigh, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    if nkeys <= 1:  # single plot
        ax = np.array([ax])

    if plotswide > 1 and plotshigh > 1:  # array of axes
        for axix in ax:
            axix[0].set_ylabel(titlecase(table.lower().replace("phase", "")) + " (%)")
    else:
        ax[0].set_ylabel(titlecase(table.lower().replace("phase", "")) + " (%)")

    ax = ax.flat

    for ix, (t, d) in enumerate(summary.items()):
        phases = d["phases"]
        output = d["output"]
        phasecols = getattr(output, table).columns[
            [any([p in c for p in phases]) for c in getattr(output, table).columns]
        ]
        phasecols = [i for i in phasecols if i != xvar]
        for p in phasecols:
            phase = [p, p[: p.find("_")]][p.find("_") >= 0]
            config = dict(
                color=__DEFAULT_DISC_COLORMAP__(colors[phase]),
                label=p,
                ls=phase_linestyle(p),
            )
            getattr(output, table).loc[:, [p, xvar]].plot(x=xvar, ax=ax[ix], **config)
        ax[ix].legend(
            loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False, facecolor=None
        )
        ax[ix].set_title(d["output"].title)
    ax[0].set_yscale(yscale)
    plt.tight_layout()
    return fig


def plot_comptable(
    summary,
    table="liquidcomp",
    xvar="Temperature",
    plotswide=1,
    figsize=None,
    yscale="linear",
):
    """
    Plot a particular compostiion table per-experiment across indiviudal axes.

    Parameters
    -----------
    summary : :class:`dict`
        Dictionary of experiment result outputs indexed by title.
    table : :class:`str`
        Which table to plot from the experiment output.
    xvar : :class:`str`
        Variable to use for the independent axis.
    plotswide : :class:`int`
        With of the figure, as a number of axes.
    figsize : :class:`tuple`
        Size of the figure, optional.
    yscale : :class:`str`
        Scale to use for the y-axis.

    Returns
    ---------
    :class:`matplotlib.figure.Figure`
        Figure on which the results are plotted.
    """
    all_components = set()
    for s in summary.values():
        all_components = all_components | (
            set(getattr(s["output"], table).columns) & common_oxides(as_set=True)
        )

    colors = {k: v for k, v in zip(all_components, range(len(all_components)))}

    nkeys = len(summary.keys())
    plotshigh = nkeys // plotswide
    if nkeys % plotswide:
        plotshigh += 1
    fig, ax = plt.subplots(
        plotshigh, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    if nkeys <= 1:  # single plot
        ax = np.array([ax])

    if plotswide > 1 and plotshigh > 1:  # array of axes
        for axix in ax:
            axix[0].set_ylabel("Mass (%)")
    else:
        ax[0].set_ylabel("Mass (%)")

    ax = ax.flat

    if nkeys % plotswide:
        for a in ax[nkeys:]:
            a.axis("off")

    for ix, (t, d) in enumerate(summary.items()):
        phases, output = d["phases"], d["output"]
        comptable = getattr(output, table)
        components = [
            i
            for i in comptable.columns
            if i in common_oxides(as_set=True) and i != xvar
        ]
        for c in components:
            config = dict(color=__DEFAULT_DISC_COLORMAP__(colors[c]))
            comptable.loc[:, [c, xvar]].plot(x=xvar, ax=ax[ix], **config, label=c)
        ax[ix].legend(
            loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False, facecolor=None
        )
        ax[ix].set_title(d["output"].title)
    ax[0].set_yscale(yscale)
    plt.tight_layout()
    return fig


def plot_phase_composition(
    summary,
    phase="olivine",
    xvar="Temperature",
    plotswide=1,
    figsize=None,
    yscale="linear",
):
    """
    Plot a particular phase per-experiment across indiviudal axes.

    Parameters
    -----------
    summary : :class:`dict`
        Dictionary of experiment result outputs indexed by title.
    phase : :class:`str`
        Which phase to plot from the experiment output.
    xvar : :class:`str`
        Variable to use for the independent axis.
    plotswide : :class:`int`
        With of the figure, as a number of axes.
    figsize : :class:`tuple`
        Size of the figure, optional.
    yscale : :class:`str`
        Scale to use for the y-axis.

    Returns
    ---------
    :class:`matplotlib.figure.Figure`
        Figure on which the results are plotted.
    """
    all_phases = set()
    all_components = set()
    for s in summary.values():
        all_phases = all_phases | {
            p for p in s["output"].phasenames if phase.lower() in p.lower()
        }
    for s in summary.values():
        for p in all_phases:
            if p in s["output"].phasenames:
                all_components = all_components | (
                    set(s["output"].phases[p].columns) & common_oxides(as_set=True)
                )
    colors = {k: v for k, v in zip(all_components, range(len(all_components)))}

    nkeys = len(summary.keys())
    plotshigh = nkeys // plotswide
    if nkeys % plotswide:
        plotshigh += 1
    fig, ax = plt.subplots(
        plotshigh, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    if nkeys <= 1:  # single plot
        ax = np.array([ax])

    if plotswide > 1 and plotshigh > 1:  # array of axes
        for axix in ax:
            axix[0].set_ylabel("Mass (%)")
    else:
        ax[0].set_ylabel("Mass (%)")

    ax = ax.flat

    for ix, (t, d) in enumerate(summary.items()):
        phases = [p for p in d["output"].phasenames if p in all_phases]
        components = all_components - set([xvar])
        output = d["output"]
        for p in phases:
            for c in components:
                phase = [p, p[: p.find("_")]][p.find("_") >= 0]
                label = c
                if len(phases) > 1:
                    label = label + p
                config = dict(
                    color=__DEFAULT_DISC_COLORMAP__(colors[c]),
                    label=label,
                    ls=phase_linestyle(p),
                )
                config = dict(color=__DEFAULT_DISC_COLORMAP__(colors[c]), label=label)

                if s["output"].phases[p].loc[:, [c, xvar]].size:
                    try:
                        s["output"].phases[p].loc[:, [c, xvar]].plot(
                            x=xvar, ax=ax[ix], **config
                        )
                    except:
                        pass
        ax[ix].legend(
            loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False, facecolor=None
        )
        ax[ix].set_title(d["output"].title)
    ax[0].set_yscale(yscale)
    plt.tight_layout()
    return fig


def table_by_phase(
    summary,
    table="phasevol",
    xvar="Temperature",
    plotswide=2,
    figsize=None,
    yscale="linear",
):
    """
    Plot a particular table per-phase across indiviudal axes.

    Parameters
    -----------
    summary : :class:`dict`
        Dictionary of experiment result outputs indexed by title.
    table : :class:`str`
        Which table to access from the experiment output.
    xvar : :class:`str`
        Variable to use for the independent axis.
    plotswide : :class:`int`
        With of the figure, as a number of axes.
    figsize : :class:`tuple`
        Size of the figure, optional.
    yscale : :class:`str`
        Scale to use for the y-axis.

    Returns
    ---------
    :class:`matplotlib.figure.Figure`
        Figure on which the results are plotted.
    """
    phases = set()
    for k, v in summary.items():
        phases |= v["phases"]
    colors = {k: v for k, v in zip(phases, range(len(phases)))}
    plotshigh = len(phases) // plotswide + [0, 1][len(phases) % plotswide > 0]
    figsize = figsize or (4 * plotswide, 3 * plotshigh)
    fig, ax = plt.subplots(
        plotshigh, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    ax = ax.flat
    ax[0].set_ylabel(titlecase(table.lower().replace("phase", "")) + " (%)")
    ax[0].set_yscale(yscale)

    for ix, p in enumerate(phases):
        proxies = {}
        # ax[ix].set_title(p)
        for k, v in summary.items():
            output = v["output"]
            outphases = output.phases
            outtbl = getattr(output, table)

            c = [i for i in outtbl.columns if p in i]
            config = dict(
                color=__DEFAULT_DISC_COLORMAP__(colors[p]),
                # alpha=1 / np.log(outtbl.index.size),
            )
            if c:
                for _p in c:
                    _pconfig = {**config, "ls": phase_linestyle(_p)}
                    proxies[_p] = proxy_line(**_pconfig)
                    outtbl.loc[:, [xvar, _p]].plot(
                        x=xvar, ax=ax[ix], legend=False, **_pconfig
                    )
        ax[ix].legend(
            list(proxies.values()),
            list(proxies.keys()),
            frameon=False,
            facecolor=None,
            # bbox_to_anchor=(1.0, 1.0),
            loc="best",
        )

    fig.tight_layout()

    return fig
