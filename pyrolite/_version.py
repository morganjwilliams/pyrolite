from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("pyrolite")
except PackageNotFoundError:
    # package is not installed
    __version__ : str = ''
