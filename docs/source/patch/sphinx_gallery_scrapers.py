import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from pyrolite.util.plot import save_figure

from sphinx_gallery import scrapers

def alt_matplotlib_scraper(block, block_vars, gallery_conf, **kwargs):
    """Patched matplotlib scraper which won't close figures.

    Parameters
    ----------
    block : tuple
        A tuple containing the (label, content, line_number) of the block.
    block_vars : dict
        Dict of block variables.
    gallery_conf : dict
        Contains the configuration of Sphinx-Gallery
    **kwargs : dict
        Additional keyword arguments to pass to
        :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.
        The ``format`` kwarg in particular is used to set the file extension
        of the output file (currently only 'png' and 'svg' are supported).

    Returns
    -------
    rst : str
        The ReSTructuredText that will be rendered to HTML containing
        the images. This is often produced by :func:`~sphinx_galleryscrapers.figure_rst`.
    """
    lbl, cnt, ln = block
    image_path_iterator = block_vars["image_path_iterator"]

    image_paths = []
    figs = [m.canvas.figure for m in _pylab_helpers.Gcf.get_all_fig_managers()]
    fltr = "[^a-zA-Z0-9]+fig,\s??|[^a-zA-Z0-9]+fig\s??|[^a-zA-Z0-9]+figure\s?|plt.show()|plt.gcf()"
    if figs and re.search(fltr, cnt):  # where figure or plt.show is called
        for fig, image_path in zip([figs[-1]], image_path_iterator):
            save_figure(
                fig,
                save_at=Path(image_path).parent,
                name=Path(image_path).stem,
                save_fmts=["png"],
                **kwargs.copy()
            )
            image_paths.append(image_path)

    return scrapers.figure_rst(image_paths, gallery_conf["src_dir"])

scrapers._scraper_dict.update(dict(altmatplot=alt_matplotlib_scraper))
