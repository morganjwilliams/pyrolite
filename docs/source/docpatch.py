from sphinx_gallery import gen_rst
from sphinx_gallery import binder
from sphinx_gallery.gen_rst import *
from sphinx_gallery.binder import *
from sphinx_gallery import scrapers
from sphinx_gallery.scrapers import *
import matplotlib
import matplotlib.pyplot as plt
import logging
from matplotlib import _pylab_helpers


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
        the images. This is often produced by :func:`figure_rst`.
    """
    lbl, cnt, ln = block
    image_path_iterator = block_vars["image_path_iterator"]

    image_paths = []

    for fig, image_path in zip(
        [m.canvas.figure for m in _pylab_helpers.Gcf.get_all_fig_managers()],
        image_path_iterator,
    ):
        #image_path = "%s-%s" % (os.path.splitext(image_path)[0], ln)
        if "format" in kwargs:
            image_path = "%s.%s" % (os.path.splitext(image_path)[0], kwargs["format"])
        # Set the fig_num figure as the current figure as we can't
        # save a figure that's not the current figure.
        to_rgba = matplotlib.colors.colorConverter.to_rgba
        # shallow copy should be fine here, just want to avoid changing
        # "kwargs" for subsequent figures processed by the loop

        fig.savefig(image_path, **kwargs.copy())
        image_paths.append(image_path)
    return figure_rst(image_paths, gallery_conf["src_dir"])


def alt_gen_binder_rst(
    fpath, binder_conf, gallery_conf, img="https://mybinder.org/badge_logo.svg"
):
    """Generate the RST + link for the Binder badge.
    Parameters
    ----------
    fpath: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict or None
        If a dictionary it must have the following keys:
        'binderhub_url'
            The URL of the BinderHub instance that's running a Binder service.
        'org'
            The GitHub organization to which the documentation will be pushed.
        'repo'
            The GitHub repository to which the documentation will be pushed.
        'branch'
            The Git branch on which the documentation exists (e.g., gh-pages).
        'dependencies'
            A list of paths to dependency files that match the Binderspec.
    Returns
    -------
    rst : str
        The reStructuredText for the Binder badge that links to this file.
    """
    binder_conf = check_binder_conf(binder_conf)
    binder_url = gen_binder_url(fpath, binder_conf, gallery_conf)
    rst = (
        "\n" ".. image:: {0}\n" "    :target: {1}\n" "    :alt: Launch Binder\n"
    ).format(img, binder_url)
    return rst


def _save_rst_example(
    example_rst, example_file, time_elapsed, memory_used, gallery_conf
):
    """Patch for sphinx_gallery's save_rst_example
    Parameters
    ----------
    example_rst : str
        rst containing the executed file content
    example_file : str
        Filename with full path of python example file in documentation folder
    time_elapsed : float
        Time elapsed in seconds while executing file
    memory_used : float
        Additional memory used during the run.
    gallery_conf : dict
        Sphinx-Gallery configuration dictionary
    """

    ref_fname = os.path.relpath(example_file, gallery_conf["src_dir"])
    ref_fname = ref_fname.replace(os.path.sep, "_")

    binder_conf = check_binder_conf(gallery_conf.get("binder"))

    binder_text = (
        " or run this example in your browser via Binder" if len(binder_conf) else ""
    )

    # the note section is excluded here
    note_rst = (
        ".. note::\n"
        "    :class: sphx-glr-download-link-note\n\n"
        "    Click :ref:`here <sphx_glr_download_{0}>` "
        "to download the full example code{1}\n"
    ).format(ref_fname, binder_text)

    title_rst = (
        ".. rst-class:: sphx-glr-example-title\n\n" ".. _sphx_glr_{0}:\n\n"
    ).format(ref_fname)

    example_rst = title_rst + example_rst
    if time_elapsed >= gallery_conf["min_reported_time"]:
        time_m, time_s = divmod(time_elapsed, 60)
        example_rst += TIMING_CONTENT.format(time_m, time_s)

    if gallery_conf["show_memory"]:
        example_rst += "**Estimated memory usage:** {0: .0f} MB\n\n".format(memory_used)

    # Generate a binder URL if specified
    binder_badge_rst = ""
    if len(binder_conf) > 0:
        binder_badge_rst += gen_binder_rst(example_file, binder_conf, gallery_conf)

    fname = os.path.basename(example_file)
    example_rst += CODE_DOWNLOAD.format(
        fname, gen_rst.replace_py_ipynb(fname), binder_badge_rst, ref_fname
    )
    example_rst += SPHX_GLR_SIG

    write_file_new = re.sub(r"\.py$", ".rst.new", example_file)
    with codecs.open(write_file_new, "w", encoding="utf-8") as f:
        f.write(example_rst)
    # in case it wasn't in our pattern, only replace the file if it's
    # still stale.
    gen_rst._replace_md5(write_file_new)


scrapers._scraper_dict = dict(
    matplotlib=matplotlib_scraper,
    mayavi=mayavi_scraper,
    altmatplot=alt_matplotlib_scraper,
)
# binder.gen_binder_rst = alt_gen_binder_rst
gen_rst.save_rst_example = _save_rst_example
