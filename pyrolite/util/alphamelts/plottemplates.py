import logging

import matplotlib.pyplot as plt
from pyrolite.util.plot import __DEFAULT_DISC_COLORMAP__
from pyrolite.util.text import titlecase
from pyrolite.geochem.ind import __common_oxides__

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def plot_phasetable(
    summary,
    table="phasevolume",
    xvar="Temperature",
    figsize=(12, 20),
    plotswide=2,
    yscale="linear",
):
    all_phases = set()
    for s in summary.values():
        all_phases = all_phases | s["phases"]

    colors = {k: v for k, v in zip(all_phases, range(len(all_phases)))}

    nkeys = len(summary.keys())
    plotshigh = nkeys // plotswide
    if nkeys % plotswide:
        plotshigh += 1
    fig, ax = plt.subplots(
        nkeys // plotswide, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    for axix in ax:
        axix[0].set_ylabel(titlecase(table.lower().replace("phase", "") + " %"))

    ax = ax.flat

    for ix, (t, d) in enumerate(summary.items()):
        phases = d["phases"]
        output = d["output"]
        phasecols = getattr(output, table).columns[
            [any([p in c for p in phases]) for c in getattr(output, table).columns]
        ]
        phasecols = [i for i in phasecols if i != xvar]
        plotted = []
        for p in phasecols:
            phase = [p, p[: p.find("_")]][p.find("_") >= 0]
            config = dict(color=__DEFAULT_DISC_COLORMAP__(colors[phase]), label=p)
            if phase in plotted:  # eg. clinopyroxene 2
                config = {**config, "ls": ["--", ":", "-."][plotted.count(phase) - 1]}
            getattr(output, table).loc[:, [p, xvar]].plot(x=xvar, ax=ax[ix], **config)
            plotted.append(phase)
        ax[ix].legend(
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, facecolor=None
        )
        ax[ix].set_title(d["output"].title)
    ax[0].set_yscale(yscale)
    plt.tight_layout()
    return fig


def plot_comptable(
    summary,
    table="liquidcomp",
    xvar="Temperature",
    figsize=(12, 20),
    plotswide=2,
    yscale="linear",
):
    all_components = set()
    for s in summary.values():
        all_components = all_components | (
            set(getattr(s["output"], table).columns) & __common_oxides__
        )

    colors = {k: v for k, v in zip(all_components, range(len(all_components)))}

    nkeys = len(summary.keys())
    plotshigh = nkeys // plotswide
    if nkeys % plotswide:
        plotshigh += 1
    fig, ax = plt.subplots(
        nkeys // plotswide, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    for axix in ax:
        axix[0].set_ylabel("Wt%")
    ax = ax.flat

    if nkeys % plotswide:
        for a in ax[nkeys:]:
            a.axis("off")

    for ix, (t, d) in enumerate(summary.items()):
        phases, output = d["phases"], d["output"]
        comptable = getattr(output, table)
        components = [
            i for i in comptable.columns if i in __common_oxides__ and i != xvar
        ]
        for c in components:
            config = dict(color=__DEFAULT_DISC_COLORMAP__(colors[c]))
            comptable.loc[:, [c, xvar]].plot(x=xvar, ax=ax[ix], **config, label=c)
        ax[ix].legend(
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, facecolor=None
        )
        ax[ix].set_title(d["output"].title)
    ax[0].set_yscale(yscale)
    plt.tight_layout()
    return fig


def plot_phase_composition(
    summary,
    phase="olivine",
    xvar="Temperature",
    figsize=(12, 20),
    plotswide=2,
    yscale="linear",
):
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
                    set(s["output"].phases[p].columns) & __common_oxides__
                )
    colors = {k: v for k, v in zip(all_components, range(len(all_components)))}

    nkeys = len(summary.keys())
    plotshigh = nkeys // plotswide
    if nkeys % plotswide:
        plotshigh += 1
    fig, ax = plt.subplots(
        nkeys // plotswide, plotswide, figsize=figsize, sharex=True, sharey=True
    )
    for axix in ax:
        axix[0].set_ylabel("Wt %")

    ax = ax.flat

    for ix, (t, d) in enumerate(summary.items()):
        phases = [p for p in d["output"].phasenames if p in all_phases]
        components = all_components - set([xvar])
        output = d["output"]
        for p in phases:
            plotted = []
            for c in components:
                phase = [p, p[: p.find("_")]][p.find("_") >= 0]
                label = c
                if len(phases) > 1:
                    label = label + p
                config = dict(color=__DEFAULT_DISC_COLORMAP__(colors[c]), label=label)

                if phase in plotted:  # eg. clinopyroxene 2
                    config = {
                        **config,
                        "ls": ["--", ":", "-."][plotted.count(phase) - 1],
                    }
                if s["output"].phases[p].loc[:, [c, xvar]].size:
                    try:
                        s["output"].phases[p].loc[:, [c, xvar]].plot(
                            x=xvar, ax=ax[ix], **config
                        )
                    except:
                        pass
            plotted.append(phase)
        ax[ix].legend(
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, facecolor=None
        )
        ax[ix].set_title(d["output"].title)
    ax[0].set_yscale(yscale)
    plt.tight_layout()
    return fig
