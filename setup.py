from setuptools import setup, find_packages
import versioneer

setup(name='pyrolite',
      description="Tools for geochemical data analysis.",
      long_description=open('README.md').read(),
      version=versioneer.get_version(),
      url='https://github.com/morgan.j.williams/pyrolite',
      author='Morgan Williams',
      author_email='morgan.williams@csiro.au',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
      packages=find_packages(exclude=['test*']),
      install_requires=['pathlib',
                        'numpy',
                        'scipy',
                        'scikit-learn',
                        'pandas',
                        'matplotlib',
                        'periodictable',
                        'xlrd',
                        'mpmath',
                        'python-ternary'
                        ],

      extras_require={'dev': ['versioneer',
                              'nbstripout',
                              'nbdime',
                              'twine']},

      tests_require=['pytest',
                     'pytest-runner',
                     'pytest-cov',
                     'coverage',
                     'coveralls'],

      test_suite="test",
      package_data={'pyrolite': ['data/*']},
      license='CSIRO Modifed MIT/BSD',
      cmdclass=versioneer.get_cmdclass()
)
