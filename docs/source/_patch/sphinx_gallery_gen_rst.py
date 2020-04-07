import re
import os
import sys
import copy
import ast
import codecs
from functools import partial
from io import StringIO
import logging
from sphinx_gallery import gen_rst
from sphinx_gallery import scrapers
from sphinx_gallery.gen_rst import (
    _replace_md5,
    _exec_once,
    replace_py_ipynb,
    gen_binder_rst,
    check_binder_conf,
    TIMING_CONTENT,
    CODE_DOWNLOAD,
    SPHX_GLR_SIG
)
import matplotlib.pyplot as plt

_si = """
.. image:: /%s
    :width: 80 %
    :align: center
"""
html_header = """.. only:: builder_html

.. raw:: html

    {0}\n        <br />\n        <br />"""


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
        " or run this example in your browser via Binder." if len(binder_conf) else ""
    )

    # the note section is excluded here
    note_rst = (
        ".. only:: html\n\n"
        "   .. note::\n"
        "       :class: sphx-glr-download-link-note\n\n"
        "       Click :ref:`here <sphx_glr_download_{0}>` "
        "   to download the full example code{1}\n"
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
        fname, replace_py_ipynb(fname), binder_badge_rst, ref_fname
    )
    example_rst += SPHX_GLR_SIG

    write_file_new = re.sub(r"\.py$", ".rst.new", example_file)
    with codecs.open(write_file_new, "w", encoding="utf-8") as f:
        f.write(example_rst)
    # in case it wasn't in our pattern, only replace the file if it's
    # still stale.
    _replace_md5(write_file_new)
    plt.close("all")


# patches for sphinx_gallery 0.5
gen_rst.SINGLE_IMAGE = _si
gen_rst.save_rst_example = _save_rst_example
