import re
import os
import sys
import copy
import ast
from functools import partial
from io import StringIO
import logging
from pathlib import Path
from sphinx_gallery import gen_rst
from sphinx_gallery import binder
from sphinx_gallery.gen_rst import *
from sphinx_gallery.binder import *
from sphinx_gallery import scrapers
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from pyrolite.util.plot import save_figure

_si = """
.. image:: /%s
    :width: 80 %
    :align: center
"""
html_header = """.. only:: builder_html

.. raw:: html

    {0}\n        <br />\n        <br />"""


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
    fltr = "[^a-zA-Z0-9]+fig\s??|[^a-zA-Z0-9]+figure\s?|plt.show()|plt.gcf()"
    if figs and re.search(fltr, cnt):  # where figure or plt.show is called
        for fig, image_path in zip([figs[-1]], image_path_iterator):
            to_rgba = matplotlib.colors.colorConverter.to_rgba
            save_figure(
                fig,
                save_at=Path(image_path).parent,
                name=Path(image_path).stem,
                save_fmts=["png"],
                **kwargs.copy()
            )
            image_paths.append(image_path)

    return scrapers.figure_rst(image_paths, gallery_conf["src_dir"])


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
    plt.close("all")


def execute_code_block(compiler, block, example_globals, script_vars, gallery_conf):
    """Executes the code block of the example file"""
    blabel, bcontent, lineno = block
    # If example is not suitable to run, skip executing its blocks
    if not script_vars["execute_script"] or blabel == "text":
        return ""

    cwd = os.getcwd()
    # Redirect output to stdout and
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    src_file = script_vars["src_file"]

    # First cd in the original example dir, so that any file
    # created by the example get created in this directory

    captured_std = StringIO()
    os.chdir(os.path.dirname(src_file))

    sys_path = copy.deepcopy(sys.path)
    sys.path.append(os.getcwd())
    sys.stdout = sys.stderr = LoggingTee(captured_std, logger, src_file)

    try:
        dont_inherit = 1
        if sys.version_info >= (3, 8):
            ast_Module = partial(ast.Module, type_ignores=[])
        else:
            ast_Module = ast.Module
        code_ast = ast_Module([bcontent])
        code_ast = compile(
            bcontent, src_file, "exec", ast.PyCF_ONLY_AST | compiler.flags, dont_inherit
        )
        ast.increment_lineno(code_ast, lineno - 1)
        # capture output if last line is expression
        is_last_expr = False
        if len(code_ast.body) and isinstance(code_ast.body[-1], ast.Expr):
            is_last_expr = True
            last_val = code_ast.body.pop().value
            # exec body minus last expression
            _, mem_body = gen_rst._memory_usage(
                gen_rst._exec_once(
                    compiler(code_ast, src_file, "exec"), example_globals
                ),
                gallery_conf,
            )
            # exec last expression, made into assignment
            body = [
                ast.Assign(
                    targets=[ast.Name(id="___", ctx=ast.Store())], value=last_val
                )
            ]
            last_val_ast = ast_Module(body=body)
            ast.fix_missing_locations(last_val_ast)
            _, mem_last = gen_rst._memory_usage(
                gen_rst._exec_once(
                    compiler(last_val_ast, src_file, "exec"), example_globals
                ),
                gallery_conf,
            )
            # capture the assigned variable
            ___ = example_globals["___"]
            mem_max = max(mem_body, mem_last)
        else:
            _, mem_max = gen_rst._memory_usage(
                gen_rst._exec_once(
                    compiler(code_ast, src_file, "exec"), example_globals
                ),
                gallery_conf,
            )
        script_vars["memory_delta"].append(mem_max)

    except Exception:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        except_rst = gen_rst.handle_exception(
            sys.exc_info(), src_file, script_vars, gallery_conf
        )
        code_output = u"\n{0}\n\n\n\n".format(except_rst)
        # still call this even though we won't use the images so that
        # figures are closed
        scrapers.save_figures(block, script_vars, gallery_conf)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, orig_stderr = orig_stdout, orig_stderr
        sys.path = sys_path
        os.chdir(cwd)

        last_repr = None
        repr_meth = None
        if gallery_conf["capture_repr"] != () and is_last_expr:
            for meth in gallery_conf["capture_repr"]:
                try:
                    last_repr = getattr(___, meth)()
                    # for case when last statement is print()
                    if last_repr == "None":
                        repr_meth = None
                    else:
                        repr_meth = meth
                except Exception:
                    pass
                else:
                    if isinstance(last_repr, str):
                        break
        captured_std = captured_std.getvalue().expandtabs()
        # normal string output
        if repr_meth in ["__repr__", "__str__"] and last_repr:
            captured_std = u"{0}\n{1}".format(captured_std, last_repr)
        if captured_std and not captured_std.isspace():
            captured_std = CODE_OUTPUT.format(indent(captured_std, u" " * 4))
        else:
            captured_std = ""
        images_rst = scrapers.save_figures(block, script_vars, gallery_conf)
        # give html output its own header
        if repr_meth == "_repr_html_":
            captured_html = html_header.format(indent(last_repr, u" " * 8))
        else:
            captured_html = ""
        code_output = u"\n{0}\n\n{1}\n{2}\n\n".format(
            images_rst, captured_std, captured_html
        )
    finally:
        os.chdir(cwd)
        sys.path = sys_path
        sys.stdout, sys.stderr = orig_stdout, orig_stderr

    return code_output


scrapers._scraper_dict.update(dict(altmatplot=alt_matplotlib_scraper))

gen_rst.SINGLE_IMAGE = _si

gen_rst.execute_code_block = execute_code_block
# binder.gen_binder_rst = alt_gen_binder_rst
