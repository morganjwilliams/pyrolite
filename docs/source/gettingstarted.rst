Getting Started
----------------

.. note:: This page is under construction. Feel free to send through suggestions or
          questions.

Getting Set Up
~~~~~~~~~~~~~~~

Before you can get up and running with pyrolite, you'll need to have a distribution of
Python installed. If you're joining the scientific Python world, chances are that the
`Anaconda distributions <https://www.anaconda.com/distribution/#download-section>`__
are a good match for you [*]_. Check PyPI for the most up to date information regarding
:mod:`pyrolite` compatibility, but as of writing this guide, :mod:`pyrolite` is
known to work well with Python 3.5, 3.6 and 3.7.

When it comes to how you edit files and interact with Python, there are now many
different choices (especially for editors), and choosing something which allows
you to work the way you wish to work is the key aspect. While you can certainly edit
python files (:code:`.py`) in any text editor, choosing an editor designed for the task
makes life easier. Choosing one is subjective - know that many exist, and perhaps try a
few. Integrated Development Environments (IDEs) often allow you to
quickly edit and run code within the same window (e.g.
`Spyder <https://www.spyder-ide.org/>`__, which is typically included in the default
Anaconda distribution). Through notebooks and related ideas the
`Jupyter <https://jupyter.org/>`__ ecosystem has broadened how people are interacting
with code across multiple languages, including Python. For reference,
:mod:`pyrolite` has been developed principally in `Atom <https://atom.io>`__,
leveraging the `Hydrogen <https://atom.io/packages/hydrogen>`__ package to provide
an interactive coding environment using Jupyter.

Finally, consider getting up to speed with simple Git practises for your projects
and code such that you can keep versioned histories of your analyses, and have a look
at hosted repository services (e.g. `GitHub <https://github.com/>`__,
`GitLab <https://gitlab.com>`__). These hosted repositories together with integrated
services are often worth taking advantage of (e.g. hosting
material and analyses from papers, posters or presentation, and linking this through
to `Zenodo <https://jupyter.org/>`__ to get an archived version with a DOI).

.. [*] If you're strapped for space, or are bloat-averse, you could also consider using
      `Anaconda's miniconda distributions <https://docs.conda.io/en/latest/miniconda.html>`__.


Installing pyrolite
~~~~~~~~~~~~~~~~~~~~~

There's a separate page dedicated to `pyrolite installations <installation.html>`__,
but for most purposes, the best way to install pyrolite is through opening a terminal
(an Anaconda terminal, if that's the distribution you're using) and type:

.. code-block:: bash

  pip install pyrolite

To keep pyrolite up to date (new versions are released often), periodically run the
following to update your local version:

.. code-block:: bash

  pip install --upgrade pyrolite


Writing Some Code
~~~~~~~~~~~~~~~~~~~

If you're new to Python [*]_, or just new to :mod:`pyrolite`, checking out some of the
examples is a good first step. You should be able to copy-paste or download
and run each of these examples as written on your own system, or alternatively you
can run them interatively in your browser thanks to
`Binder <https://mybinder.readthedocs.io/en/latest/>`__ and
`sphinx-gallery <https://github.com/sphinx-gallery/sphinx-gallery>`__
(check for the links towards the bottom of each page). Have a play with these, and
adapt them to your own purposes.

.. [*] If you're completely new to Python, check out some of the many free online
       courses to get up to scratch with basic Python concepts, data structures
       and get in a bit of practice writing code (e.g. the basic Python course on
       `Codecademy <https://www.codecademy.com/>`__). Knowing your way around some
       of these things before you dive into applying them can help make it a much
       more surmountable challenge. Remember that the pyrolite community is also
       around to help out if you get stuck, and we all started from a similar place!
       There are no 'stupid questions', so feel free to ping us on
       `Gitter <https://gitter.im/pyrolite/community>`__ with any questions
       or aspects that are proving particularly challenging.


Importing Data
~~~~~~~~~~~~~~~~

A large part of the pyrolite API is based around :mod:`pandas` DataFrames.
One of the first hurdles for new users is importing their own data tables.
To make this as simple as possible, it's best to organise - or 'tidy' - your data
tables [*]_. Minimise unnecessary whitespace, and
where possible make sure your table columns are the first row of your table.
In most cases, where these data are in the form of text or Excel files,
the typical steps for data import are similar. A few simple examples are given
below.

To import a table from a .csv file:

.. code-block:: python

   from pathlib import Path
   import pandas as pd

   filepath = Path('./mydata.csv')
   df = pd.read_csv(filepath)


In the case of an excel table:

.. code-block:: python

  filepath = Path('./mydata.xlsx')
  df = pd.read_excel(filepath)


There is also a pyrolite function which abstracts away these differences by making a
few assumptions, and enables you to import the table from either a csv or excel file:

.. code-block:: python

  from pyrolite.util.pd import read_table
  df = read_table(filepath)


.. [*] Where each variable is a column, and each observation is a row. If you're
       unfamiliar with the 'Tidy Data' concept, check out [Wickham2014]_.

.. [Wickham2014] Wickham, H., 2014. Tidy Data.
                 Journal of Statistical Software 59, 1–23.
                 `doi: doi.org/10.18637/jss.v059.i10 <https://doi.org/10.18637/jss.v059.i10>`__

`Gitter Community <https://gitter.im/pyrolite/community>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Gitter Community has been set up to serve as a landing page for conversations,
questions and support regarding the pyrolite python package and related activities.
Here we hope to capture questions and bugs from the community such that these can be
addressed quickly, and we can ensure pyrolite and its documentation are as useful as
possible to the community. Please feel free to use this space to:

    * Ask questions or seek help about getting started with
      pyrolite or particular pyrolite features
    * Get tips for troubleshooting
    * Discuss the general development of pyrolite
    * Ask about contributing to pyrolite

Items which are related to specific aspects of pyrolite development
(requesting a feature, or reporting an identified bug) are best coordinated through
`GitHub <https://github.com/morganjwilliams/pyrolite>`__,
but feel free to touch base here first.
See below and the `contribution <./dev/contributing.html>`__
and `development <./dev/development.html>`__ guides for  more information.

Note that users and contributors in online spaces (including Gitter) are expected to
adhere to the `Code of Conduct <conduct.html>`__ of this project (and any other
guidelines of the relevant platform).

Bugs, Debugging & Bug Reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section provides some guidance for what to do when things don't work as expected,
and some tips for debugging issues and common errors associated with
pyrolite. Note that the scope of these suggestions is necessarily limited, and specific
issues which relate more to :mod:`matplotlib.pyplot`, :mod:`pandas`, and :mod:`numpy`
are often well documented elsewhere online.

* Checked the documentation, had a look for FAQ and examples here and still stuck?

  The maintainers are happy to answer questions and help you solve small bugs.
  It's useful to know where people get stuck so we can modify things where useful,
  and this is an easy way to contribute to the project. Consider posting a question on
  `Gitter <https://gitter.im/pyrolite/community>`__, and if you think it's something
  others may run into or could be a problem related to use of another package,
  perhaps also consider posting a question on
  `stackoverflow <https://stackoverflow.com/>`__ for visibility.

* Think it's a bug or problem with pyrolite specifically?

  Some guidelines for reporting issues and bugs are given in the
  `contributing guide <./dev/contributing.html#bug-reports>`__.

.. seealso::

    `Examples <./examples/index.html>`__,
    `API <./api/API.html>`__,
    `Changelog <./dev/changelog.html>`__,
    `Code of Conduct <./dev/conduct.html>`__
