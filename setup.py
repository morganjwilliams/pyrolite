import versioneer
from setuptools import find_packages, setup

tests_require = ["pytest", "pytest-runner", "pytest-cov", "coverage", "coveralls"]
docs_require = [
    "sphinx_rtd_theme",
    "docutils<0.17",
    "sphinx>=4",
    "sphinx-autodoc-annotation",
    "sphinx_gallery>=0.6.0",
    "recommonmark",
]
dev_require = ["pytest", "versioneer", "black", "twine"] + tests_require + docs_require
db_require = ["pyodbc", "psycopg2"]
skl_require = ["scikit-learn"]
stats_require = ["statsmodels", "scikit-learn"]
spatial_require = ["owslib", "geojson"]  # this needs pyproj -> C compiler

with open("README.md", "r") as src:
    LONG_DESCRIPTION = src.read()

setup(
    name="pyrolite",
    description="Tools for geochemical data analysis.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    url="https://github.com/morganjwilliams/pyrolite",
    project_urls={
        "Documentation": "https://pyrolite.readthedocs.io/",
        "Code": "https://github.com/morganjwilliams/pyrolite",
        "Issue tracker": "https://github.com/morganjwilliams/pyrolite/issues",
    },
    author="Morgan Williams",
    author_email="morgan.williams@csiro.au",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Framework :: Matplotlib",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["geochemistry", "compositional data", "visualisation", "petrology"],
    packages=find_packages(exclude=["test*"]),
    install_requires=[
        "numpy",
        "numpydoc",
        "tinydb>4.1",  # >4.1 required for read-only access mode for JSON storage
        "typing-extensions",  # required for newer tinydb versions?
        "psutil",
        "periodictable",
        "matplotlib", 
        "mpltern>=0.4.0",
        "scipy>=1.2",  # uses scipy.optimize.Bounds, added around 1.2
        "mpmath",
        "sympy>=1.7",
        "pandas>=1.0",  # dataframe acccessors, attrs attribute
        "xlrd",  # reading excel from pandas
        "openpyxl",  # writing excel from pandas
        "joblib",
        "requests",  # used by alphaMELTS utilities,  util.web
    ],
    extras_require={
        "dev": dev_require,
        "docs": docs_require,
        "skl": skl_require,
        "spatial": spatial_require,
        "db": db_require,
        "stats": stats_require,
    },
    tests_require=tests_require,
    test_suite="test",
    include_package_data=True,
    license="CSIRO Modifed MIT/BSD",
    cmdclass=versioneer.get_cmdclass(),
)
