Citation
==========

|doibadage|

.. note:: :mod:`pyrolite` began as a personal project, but as the project develops
          there are likely to be other contributors. Check the
          `Contributors list <./contributors.html>`__ and add major contributors as
          authors.

If you use :mod:`pyrolite` extensively for your research, citation of the software
would be particularly appreciated. It helps quantify the impact of the project
(assisting those contributing through paid and volunteer work), and is one way to get
the message out and help build the pyrolite community.

As no overview papers have yet been published,
the best option is to reference the `Zenodo <https://zenodo.org>`__ archive DOI. Ideally
reference a specific version if you know which one you used
(:code:`import pyrolite; pyrolite.__version__`) to make your work more replicable.

While the exact format of your citation will vary
with wherever you're publishing, it should take the general form:

  Williams, M. J. (|year|). pyrolite, Zenodo, `doi:10.5281/zenodo.2545106 <https://dx.doi.org/doi:10.5281/zenodo.2545106>`__

Or, if you wish to cite a specific version:

  Williams, M. J. (|year|). pyrolite v |version|, Zenodo, `doi:10.5281/zenodo.2545106 <https://dx.doi.org/doi:10.5281/zenodo.2545106>`__

If you're after a BibTeX citation for :mod:`pyrolite`, I've added one below.

.. parsed-literal::

    @misc{pyrolite,
      author       = {Morgan Williams},
      title        = {pyrolite |version|},
      year         = |year|,
      doi          = {10.5281/zenodo.2545106},
      url          = {https://doi.org/10.5281/zenodo.2545106}
    }
