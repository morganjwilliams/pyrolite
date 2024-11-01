#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pyrolite documentation build configuration file, created by
# sphinx-quickstart on Tue Sep 18 13:48:13 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import re
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore", "Unknown section")

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../."))
sys.path.insert(0, os.path.abspath("../.."))
# pip install git+https://github.com/rtfd/recommonmark.git@master
import recommonmark
from recommonmark.transform import AutoStructify

import pyrolite

version = re.findall(r"^[\d]*.[\d]*.[\d]*", pyrolite.__version__)[0]
release = pyrolite.__version__.replace(".dirty", "")

"""
from mock import Mock as MagicMock
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['numpy', 'scipy', 'scipy.linalg', 'scipy.stats']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
"""

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "recommonmark",
    "sphinx.ext.viewcode",  # generates sourcecode on docs site, with reverse links to docs
    "sphinx_gallery.gen_gallery",  # sphinx gallery
    # "jupyterlite_sphinx",
]

autosummary_generate = True

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]
# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pyrolite"
copyright = "2018-%s, Morgan Williams" % date.today().year

author = "Morgan Williams"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

autodoc_member_order = "bysource"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": None,
    "vcs_pageview_mode": "edit",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# from pandas-dev theme
html_css_files = ["css/custom.css"]
# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "globaltoc.html",
        "sourcelink.html",
        # "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}
html_logo = "./_static/icon_small.png"
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "morganjwilliams",  # Username
    "github_repo": "pyrolite",  # Repo name
    "github_version": "develop",  # Version
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pyrolitedoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pyrolite.tex", "pyrolite Documentation", "Morgan Williams", "manual")
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pyrolite", "pyrolite Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pyrolite",
        "pyrolite Documentation",
        author,
        "pyrolite",
        "A set of tools for getting the most from your geochemical data.",
        "Science/Research",
    )
]

# -- intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pathlib": ("https://pathlib.readthedocs.io/en/pep428/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "periodictable": ("https://periodictable.readthedocs.io/en/latest/", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "sphinx_gallery": ("https://sphinx-gallery.github.io/stable/", None),
    "mpltern": ("https://mpltern.readthedocs.io/en/latest/", None),
    "pyrolite_meltsutil": (
        "https://pyrolite-meltsutil.readthedocs.io/en/develop/",
        None,
    ),
}

# sphinx_gallery config
from sphinx_gallery.sorting import ExplicitOrder


def reset_mpl(gallery_conf, fname):
    import matplotlib.style

    # this should already be exported, so can be used
    matplotlib.style.use("pyrolite")


sphinx_gallery_conf = {
    "examples_dirs": [
        "gallery/examples/",
        "gallery/tutorials/",
        "gallery/data/",
    ],  # path to sources
    "gallery_dirs": ["examples", "tutorials", "data"],  # output paths
    "subsection_order": ExplicitOrder(
        [
            "gallery/examples/plotting",
            "gallery/examples/geochem",
            "gallery/examples/comp",
            "gallery/examples/util",
            "gallery/tutorials/",
            "gallery/data",
        ]
    ),
    "show_signature": False,
    "capture_repr": ("_repr_html_", "__repr__", "__str__"),
    "backreferences_dir": "_backreferences",
    "doc_module": ("pyrolite"),
    "filename_pattern": r"\.py",
    "default_thumb_file": str(Path("./_static/icon_small.png").resolve()),
    "remove_config_comments": True,
    "download_all_examples": False,
    "reference_url": {"pyrolite": None},
    "image_scrapers": ("altmatplot"),
    "binder": {
        # Required keys
        "org": "morganjwilliams",
        "repo": "pyrolite",
        "branch": "develop",  # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        "binderhub_url": "https://mybinder.org",  # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
        "dependencies": ["../../binder/environment.yml", "../../binder/postBuild"],
        # Optional keys
        # "filepath_prefix": "/docs/notebooks/",  # A prefix to prepend to any filepaths in Binder links.
        "notebooks_dir": "docs/source/",
        "use_jupyter_lab": True,
    },
    # "jupyterlite": {"use_jupyter_lab": True},
    "first_notebook_cell": "%matplotlib inline\n",
    "reset_modules": (reset_mpl),
    "nested_sections": False,
}
# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

from _patch._sphinx_gallery_patch import *  # patch for sphinx_gallery pages

github_doc_root = "https://github.com/morganjwilliams/pyrolite/tree/develop/docs/"

# metadata
# ordered reference composition list
from pyrolite.geochem.norm import all_reference_compositions

refs = all_reference_compositions()
reservoirs = set(
    [refs[n].reservoir for n in refs.keys() if refs[n].reservoir is not None]
)
comps = []
for r in reservoirs:
    comps += [n for n in refs if refs[n].reservoir == r]

refcomps = (
    "    <dl>"
    + "\n    ".join(
        [
            "<dt>{}</dt><dd>{}</dd>".format(
                n,
                " ".join(
                    [str(refs[n])]
                    + (
                        ["<br><b>Citation</b>: " + refs[n].citation]
                        if refs[n].citation
                        else []
                    )
                    + (
                        [
                            "<br><b>doi</b>: <a href='https://dx.doi.org/{}'>{}</a>".format(
                                refs[n].doi, refs[n].doi
                            )
                        ]
                        if refs[n].doi
                        else []
                    )
                ),
            )
            for n in comps
        ]
    )
    + "</dl>"
)
print(refcomps)
rst_prolog = """
.. |br| raw:: html

   <br />

.. |year| raw:: html

    {year}

.. |version| raw:: html

    {version}

.. |refcomps| raw:: html

    {rc}

.. |doibadages| raw:: html

    <a style="border-width:0" href="https://doi.org/10.21105/joss.02314">
    <img src="https://joss.theoj.org/papers/10.21105/joss.02314/status.svg" alt="DOI" >
    </a>

    <a href="https://zenodo.org/badge/latestdoi/137172322">
    <img src="https://zenodo.org/badge/137172322.svg" alt="Archive">
    </a>

""".format(rc=refcomps, year=str(date.today().year), version=version)

rst_prolog += """

.. raw:: html

  <a href="https://github.com/morganjwilliams/pyrolite"
     class="github-corner"
     aria-label="View source on GitHub">
  <svg width="80"
       height="80"
       viewBox="0 0 250 250"
       style="fill:#151513; color:#fff; position: absolute; top: 0; border: 0; right: 0;"
       aria-hidden="true">
       <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
       <path d="M128.3,109.0
                C113.8,99.7 119.0,89.6 119.0,89.6
                C122.0,82.7 120.5,78.6 120.5,78.6
                C119.2,72.0 123.4,76.3 123.4,76.3
                C127.3,80.9 125.5,87.3 125.5,87.3
                C122.9,97.6 130.6,101.9 134.4,103.2"
       fill="currentColor"
       style="transform-origin: 130px 106px;"
       class="octo-arm"></path>
       <path d="M115.0,115.0
                C114.9,115.1 118.7,116.5 119.8,115.4
                L133.7,101.6
                C136.9,99.2 139.9,98.4 142.2,98.6
                C133.8,88.0 127.5,74.4 143.8,58.0
                C148.5,53.4 154.0,51.2 159.7,51.0
                C160.3,49.4 163.2,43.6 171.4,40.1
                C171.4,40.1 176.1,42.5 178.8,56.2
                C183.1,58.6 187.2,61.8 190.9,65.4
                C194.5,69.0 197.7,73.2 200.1,77.6
                C213.8,80.2 216.3,84.9 216.3,84.9
                C212.7,93.1 206.9,96.0 205.4,96.6
                C205.1,102.4 203.0,107.8 198.3,112.5
                C181.9,128.9 168.3,122.5 157.7,114.1
                C157.9,116.9 156.7,120.9 152.7,124.9
                L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z"
       fill="currentColor"
       class="octo-body"></path>
     </svg>
   </a>
   <style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>
"""

from docutils import nodes


def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Role for matplotlib's rcparams, which are referred to in the documentation via links.
    """
    rendered = nodes.Text('rcParams["{}"]'.format(text))
    refuri = "https://matplotlib.org/api/matplotlib_configuration_api.html#matplotlib.rcParams"
    ref = nodes.reference(rawtext, rendered, refuri=refuri)
    return [nodes.literal("", "", ref)], []


def setup(app):
    app.add_role("rc", rcparam_role)
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: github_doc_root + url,
            "auto_toc_tree_section": "Contents",
        },
        True,
    )
    app.add_transform(AutoStructify)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
