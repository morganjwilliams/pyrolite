from setuptools import setup
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
      packages=['pyrolite'],
      install_requires=['pathlib',
                        'numpy',
                        'scipy',
                        'scikit-learn',
                        'pandas',
                        'matplotlib',
                        'periodictable',
                        'xlrd',
                        'regex'
                        ],
      license='MIT',
      cmdclass=versioneer.get_cmdclass()
)
