Submitting Author: Name (@morganjwilliams)  
Package Name: pyrolite
One-Line Description of Package: A set of tools for getting the most from your geochemical data.
Repository Link:  https://github.com/morganjwilliams/pyrolite
Version submitted:  0.2.5
Editor: TBD  
Reviewer 1: TBD  
Reviewer 2: TBD  
Archive: TBD  
Version accepted: TBD
---

## Description

- Include a brief paragraph describing what your package does:

pyrolite provides tools for munging, transforming and visualising geochemical data from common tabular formats. It enables you to recalculate and rescale whole-rock and mineral compositions, perform compositional statistics and create appropriate visualisations and also includes numerous specific utilities (e.g. a geological timescale).

## Scope
- Please indicate which [category or categories](https://www.pyopensci.org/dev_guide/peer_review/aims_scope.html) this package falls under:
    - [ ] Data retrieval
    - [ ] Data extraction
    - [x] Data munging
    - [ ] Data deposition
    - [x] Reproducibility
    - [ ] Geospatial
    - [x] Education
    - [x] Data visualization*


- Explain how the and why the package falls under these categories (briefly, 1-2 sentences):

pyrolite leverages Pandas to enable import, munging and transformation of geochemical data from standard tabular formats, and matplotlib to facilitate common (and some less common) geochemical visualisations. One of the principal project aims is assisting to improve the reproducibility of geochemical research (especially for data-processing steps which often are overlooked or undocumented).

With regards to education, pyrolite is well suited to being incorporated into university-level geochemistry and petrology classes which wish to teach a little Python. The documentation is continually evolving, and more examples and tutorials will gradually be added. It isn't a principal aim of the project, however.

-   Who is the target audience and what are scientific applications of this package?

pyrolite is targeted towards geochemists and geoscientists who use geochemical data (chemistry, mineralogy and relevant properties), especially those using lithogeochemistry. pyrolite has been developed principally to enable more reproducible data import, munging, transformation and visualization for geochemical data. In addition to this, pyrolite:

  * Encourages better practices throughout these processes, including the use of compositional statistics (i.e. log-transforms).

  * Implements some common geochemical models and methods to make these easily accessible and reusable (e.g. lattice strain models, orthogonal polynomial decomposition of Rare Earth Element patterns - 'lambdas').

  * Contains a small database of rock forming minerals for normative calculations and looking up mineral formulae.

Extensions beyond the core package are also being developed for specific applications or interfaces (e.g. to alphaMELTS).

-   Are there other Python packages that accomplish the same thing? If so, how does yours differ?

There is at least one other Python package which has some minor overlap for visualisations (GeoPyTool, which has a GUI-focused interface), but generally there are few open source Python packages for geochemistry (especially on PyPI). pyrolite provides some broader functionality (for both plotting and handling geochemistry) and is designed to be used from an editor or terminal and encourage geoscientists to further develop transferable Python skills. Where practical, the APIs for the tools on which it is built are exposed (e.g. pandas, matplotlib and sklearn).

-   If you made a pre-submission enquiry, please paste the link to the corresponding issue, forum post, or other discussion, or `@tag` the editor you contacted:

[#17](https://github.com/pyOpenSci/software-review/issues/17)

## Technical checks

For details about the pyOpenSci packaging requirements, see our [packaging guide](https://www.pyopensci.org/dev_guide/packaging/packaging_guide.html). Confirm each of the following by checking the box.  This package:

- [x] does not violate the Terms of Service of any service it interacts with.
- [x] has an [OSI approved license](https://opensource.org/licenses)
- [ ] contains a README with instructions for installing the development version.
- [ ] includes documentation with examples for all functions.
- [ ] contains a vignette with examples of its essential functions and uses.
- [x] has a test suite.
- [x] has continuous integration, such as Travis CI, AppVeyor, CircleCI, and/or others.

## Publication options

- [x] Do you wish to automatically submit to the [Journal of Open Source Software](http://joss.theoj.org/)? If so:

<details>
 <summary>JOSS Checks</summary>  

- [x] The package has an **obvious research application** according to JOSS's definition in their [submission requirements](https://joss.readthedocs.io/en/latest/submitting.html#submission-requirements). Be aware that completing the pyOpenSci review process **does not** guarantee acceptance to JOSS. Be sure to read their submission requirements (linked above) if you are interested in submitting to JOSS.
- [x] The package is not a "minor utility" as defined by JOSS's [submission requirements](https://joss.readthedocs.io/en/latest/submitting.html#submission-requirements): "Minor ‘utility’ packages, including ‘thin’ API clients, are not acceptable." pyOpenSci welcomes these packages under "Data Retrieval", but JOSS has slightly different criteria.
- [x] The package contains a `paper.md` matching [JOSS's requirements](https://joss.readthedocs.io/en/latest/submitting.html#what-should-my-paper-contain) with a high-level description in the package root or in `inst/`.
- [x] The package is deposited in a long-term repository with the DOI: [10.5281/zenodo.2545106](https://doi.org/10.5281/zenodo.2545106)

*Note: Do not submit your package separately to JOSS*

</details>

## Are you OK with Reviewers Submitting Issues and/or pull requests to your Repo Directly?
This option will allow reviewers to open smaller issues that can then be linked to PR's rather than submitting a more dense text based review. It will also allow you to demonstrate addressing the issue via PR links.

- [x] Yes I am OK with reviewers submitting requested changes as issues to my repo. Reviewers will then link to the issues in their submitted review.

## Code of conduct

- [x] I agree to abide by [pyOpenSci's Code of Conduct](https://www.pyopensci.org/dev_guide/peer_review/coc.html) during the review process and in maintaining my package should it be accepted.


**P.S.** *Have feedback/comments about our review process? Leave a comment [here](https://github.com/pyOpenSci/governance/issues/8)*

## Editor and Review Templates

[Editor and review templates can be found here](https://www.pyopensci.org/dev_guide/appendices/templates.html)
