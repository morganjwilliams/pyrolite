---
title: 'pyrolite: Python for geochemistry'
tags:
  - Python
  - geochemistry
  - geology
  - mineralogy
  - petrology
authors: # (Multiple affiliations must be quoted)
  - name: Morgan J. Williams
    orcid: 0000-0003-4764-9555
    affiliation: "1"
  - name: Louise Schoneveld
    orcid: 0000-0002-9324-1676
    affiliation: "1"
  - name: Yajing Mao
    orcid: 0000-0002-2725-2158
    affiliation: "2"
  - name: Jens Klump
    orcid: 0000-0001-5911-6022
    affiliation : "1"
  - name: Justin Gosses
    orcid: 0000-0002-5351-7295
    affiliation: "3"
  - name: Hayden Dalton
    orcid: 0000-0003-2114-9894
    affiliation: "4"
  - name: Adam Bath
    orcid: 0000-0003-0882-0807
    affiliation: "1"
  - name: Steve Barnes
    orcid: 0000-0002-4912-9177
    affiliation: "1"
affiliations:
 - name: CSIRO Mineral Resources
   index: 1
 - name: Institute of Geology and Geophysics, Chinese Academy of Geosciences
   index: 2
 - name: SAIC
   index: 3
 - name: School of Earth Sciences, University of Melbourne
   index: 4
date: 14 January 2020
bibliography: paper.bib
---

<!-- 250-1000 words -->

# Summary


<Geochem Intro>

``pyrolite`` is a Python package for working with multivariate geochemical data. Geochemical data is compositional (i.e. sums to 100%), and as such requires non-standard statistical treatment [@Aitchison1984]. Further, the need to visualise geochemical data has historically limited the use of multivariate measures in geochemical research. While effectively visualising geochemical data remains a challenge, ``pyrolite`` enables users to make better use of their data dimensionality, calculate more accurate statistical measures, and provides access to visualisation methods useful for working with steadily growing volumes of geochemical data.

``pyrolite`` provides tools for munging, transforming and visualising geochemical data from common tabular formats. It enables you to recalculate and rescale whole-rock and mineral compositions, perform compositional statistics and create appropriate visualisations and also includes numerous specific utilities (e.g. a geological timescale).

``pyrolite`` has an API which follows and builds upon a number of existing packages and where relevant exposes their API, particularly for ``matplotlib`` [@Hunter2007] and ``pandas`` [@McKinney2010]. This enables geochemists new to Python to hit the ground running, and encourages development of
transferable digital skills.

## Submodules

pyrolite.geochem

* Transforming geochemical data
* Reference compositions for normalisation

pyrolite.mineral

* Mineral endmember recalculation
* Lattice strain calculations [@Blundy2003]
* Rock-forming mineral database [@Deer2013]

pyrolite.comp

* Compositional data transformations

pyrolite.plot

* Ternary, spider, density diagrams and more.
* Templated plots, e.g. the Total-Alkali Silica diagram [@LeBas1992], Pearce diagrams [@Pearce2008]

pyrolite.util

* Utilities for ``scikit-learn`` [@Pedregosa2011], plotting, web interfaces, synthetic data,
  missing data, geological timescale & more

## Extensions

``pyrolite-meltsutil`` provides utilities for working with ``alphaMELTS`` [@Smith2005] and it's outputs, and is targeted towards performing large numbers of related experiments.

# Acknowledgements

# References
