from pyrolite.ext.alphamelts.download import install_melts
from pyrolite.util.meta import pyrolite_datafolder, stream_log

# Here we can do a conditonal install - only downloading alphamelts if it doesnt exist
if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
    stream_log("pyrolite.util.alphamelts", level="INFO")  # logger for output info
    install_melts(local=True)  # install a copy of melts to pyrolite data folder
