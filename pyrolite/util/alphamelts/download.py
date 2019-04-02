import os, sys, platform
import subprocess, shutil
import io
import requests
import zipfile
from pathlib import Path
from pyrolite.util.general import (
    copy_file,
    extract_zip,
    remove_tempdir,
    internet_connection,
    check_perl,
)
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def download_melts(directory, version="1.9"):
    """
    Download and extract melts zip file to a given directory.

    TODO:
    #2. Check install folder doens't have current installation
    #3. If it does, and update is True - overwrite

    Parameters
    ----------
    directory : str | pathlib.Path
        Directory into which to extract melts.
    """
    try:
        assert internet_connection()
        system = platform.system()
        release = platform.release()
        platver = platform.version()
        bits, linkage = platform.architecture()
        bits = bits[:2]
        mver = "-".join(version.split("."))
        zipsource = "https://magmasource.caltech.edu/alphamelts/zipfiles/"
        if system == "Linux":
            if ("Microsoft" in release) or ("Microsoft" in platver):
                url = zipsource + "wsl_alphamelts_{}.zip".format(mver)
            else:
                url = zipsource + "linux_alphamelts_{}.zip".format(mver)

        elif system == "Darwin":
            url = zipsource + "macosx_alphamelts_{}.zip".format(mver)
        elif system == "Windows":
            url = zipsource + "windows_alphamelts_{}.zip".format(mver)
            install_file = "alphamelts_win{}.exe".format(bits)
        else:
            raise NotImplementedError("System unknown: {}".format(system))

        # Set install directory for .bat files
        directory = Path(directory)
        if directory:
            install_dir = directory
        else:
            install_dir = "."

        if not install_dir.exists():
            install_dir.mkdir(parents=True)

        r = requests.get(url, stream=True)
        if r.ok:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            extract_zip(z, install_dir)
    except AssertionError:
        raise AssertionError("Need an internet connection to download.")


def install_melts(
    install_dir,
    link_dir=None,
    eg_dir=None,
    native=True,
    temp_dir=Path("~").expanduser() / "temp" / "temp_melts",
    keep_tempdir=False,
    with_readline=True,
):
    """
    Parameters
    ----------
    install_dir : s:class:`str` | :class:`pathlib.Path`
        Directory into which to install melts executable.
    link_dir : :class:`str` | :class:`pathlib.Path`
        Directory into which to deposit melts links.
    eg_dir : :class:`str` | :class:`pathlib.Path`
        Directory into which to deposit melts examples.
    native : :class:`bool`, :code:`True`
        Whether to install using perl scripts (windows).
    temp_dir : :class:`str` | :class:`pathlib.Path`, :code:`$USER$/temp/temp_melts`
        Temporary directory for melts file download and install.
    keep_tempdir : :class:`bool`, :code:`False`
        Whether to cache tempoary files and preserve the temporary directory.
    with_readline : :class:`bool`, :code:`True`
        Whether to also attempt to install with_readline.
    """
    system = platform.system()
    release = platform.release()
    version = platform.version()
    bits, linkage = platform.architecture()
    bits = bits[:2]

    temp_dir = Path(temp_dir)

    if (temp_dir / "install.command").exists():
        pass
    else:
        print("Downloading Melts")
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True)
        download_melts(temp_dir)

    install_dir = Path(install_dir)

    if not install_dir.exists():
        install_dir.mkdir(parents=True)

    if link_dir is not None:
        link_dir = Path(link_dir)
    else:
        link_dir = install_dir / "links"

    if not link_dir.exists():
        link_dir.mkdir(parents=True)

    if eg_dir is not None:
        eg_dir = Path(eg_dir)
    else:
        eg_dir = install_dir / "examples"

    if not eg_dir.exists():
        eg_dir.mkdir(parents=True)

    print("Installing to {} from {}".format(install_dir, temp_dir))
    try:
        if check_perl() and (not native):
            """
            Note: setting an install folder other than the download folder
            seems to fail here.
            Melts gets confused with the directory structure...
            and creates .bat files which point to the wrong place
            """
            install_source = os.path.join(str(temp_dir), "install.command")
            args = ["perl", install_source]

            # [C:\Users\<>\Documents\bin]
            # [./\examples]
            # use default settings file
            # continue
            # return to finish
            inputs = ["", str(link_dir), str(eg_dir), "", "y", "", ""]
            p = subprocess.run(
                args, input=("\n".join(inputs)).encode("UTF-8"), stdout=subprocess.PIPE
            )

            for line in p.stdout.decode("UTF-8").split("\r\n"):
                print(line)
            assert p.returncode == 0

            # copy files from tempdir to install_dir
            regs = []  #'command', 'command_auto_file', 'path', 'perl']
            comms = ["column_pick", "file_format", "run_alphamelts"]
            for (prefixes, ext) in [(regs, ".reg"), (comms, ".command")]:
                for prefix in prefixes:
                    temp_regpath = (temp_dir / prefix).with_suffix(ext)
                    install_regpath = install_dir / temp_regpath.name

                    shutil.copy(str(temp_regpath), str(install_regpath))
        elif native:

            # need to split into platforms
            egs = []
            for g in ["*.melts", "*.txt", "*.m "]:
                egs += list(temp_dir.glob(g))
            comms = ["column_pick", "file_format", "run_alphamelts"]
            comms = [(temp_dir / i).with_suffix(".command") for i in comms]

            files_to_copy = []

            # getting the executable file
            if system == "Windows":
                alphafile = temp_dir / "alphamelts_win{}.exe".format(bits)
            elif system == "Linux":
                if ("Microsoft" in release) or ("Microsoft" in version):
                    alphafile = temp_dir / "alphamelts_wsl"
                    # with_readline
                else:
                    alphafile = temp_dir / "alphamelts_linux{}".format(bits)
                    # with_readline
            elif system == "Darwin":
                alphafile = temp_dir / "alphamelts_macosx{}".format(bits)
                # with_readline

            # getting files to copy

            files_to_copy += [
                (eg_dir, egs),
                (install_dir, comms),
                (install_dir, [alphafile]),
            ]

            if system == "Windows":
                bats = comms + [temp_dir / "alphamelts"]
                bats = [i.with_suffix(".bat") for i in bats]
                batdata = {}

                for cf in comms:
                    batdata[cf.stem] = """@echo off\n"{}" %*""".format(
                        install_dir / cf.name
                    )
                batdata["alphamelts"] = '''@echo off\n"{}"'''.format(
                    install_dir / alphafile.name
                )
                for b in bats:
                    with open(str(b), "w") as fout:
                        fout.write(batdata[b.stem])  # dummy bats

                files_to_copy += [(link_dir, bats)]

                # regs = ['command', 'command_auto_file', 'path', 'perl']

            for (target, files) in files_to_copy:
                for fn in files:
                    copy_file(temp_dir / fn.name, target / fn.name)
    except AssertionError:
        raise AssertionError
    finally:
        if not keep_tempdir:
            remove_tempdir(temp_dir)
