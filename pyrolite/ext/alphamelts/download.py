"""
alphaMELTS download and installsation utilities.
"""
import os, sys, platform
import subprocess, shutil
import io
import requests
import zipfile
from pathlib import Path
from ...util.general import copy_file, extract_zip, remove_tempdir, check_perl
from ...util.meta import pyrolite_datafolder
from ...util.web import internet_connection
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def download_melts(directory, version=None):
    """
    Download and extract melts zip file to a given directory.

    Parameters
    ----------
    directory :  :class:`str` |  :class:`pathlib.Path`
        Directory into which to extract melts.
    version : :class:`str`
        Which alphamelts version to use. Defaults to latest stable version.

    Todo
    -----
        * Check version, enable update-style overwrite
    """
    try:
        assert internet_connection()
        system = platform.system()
        release = platform.release()
        platver = platform.version()
        bits, linkage = platform.architecture()
        bits = bits[:2]
        if version is None:
            version = 1.9  # eventually should update this programatically!
        mver = "-".join(str(version).split("."))
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
    install_dir=None,
    link_dir=None,
    eg_dir=None,
    native=True,
    temp_dir=Path("~").expanduser() / "temp" / "temp_melts",
    keep_tempdir=False,
    with_readline=True,
    local=False,
    version=None,
):
    """
    Parameters
    ----------
    install_dir : :class:`str` | :class:`pathlib.Path`
        Directory into which to install melts executable.
    link_dir : :class:`str` | :class:`pathlib.Path`
        Directory into which to deposit melts links.
    eg_dir : :class:`str` | :class:`pathlib.Path`
        Directory into which to deposit melts examples.
    native : :class:`bool`, :code:`True`
        Whether to install using python (:code:`True`) or the perl scripts (windows).
    temp_dir : :class:`str` | :class:`pathlib.Path`, :code:`$USER$/temp/temp_melts`
        Temporary directory for melts file download and install.
    keep_tempdir : :class:`bool`, :code:`False`
        Whether to cache tempoary files and preserve the temporary directory.
    with_readline : :class:`bool`, :code:`True`
        Whether to also attempt to install with_readline.
    local : :class:`bool`
        Whether to install a version of melts into an auxiliary pyrolite data folder.
        This will override
    """
    system = platform.system()
    platrel = platform.release()
    platver = platform.version()
    bits, linkage = platform.architecture()
    bits = bits[:2]

    temp_dir = Path(temp_dir)

    if (
        temp_dir / "install.command"
    ).exists() and version is None:  # already downloaded for some reason
        pass
    else:
        logger.info("Downloading Melts")
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True)
        download_melts(temp_dir, version=version)

    if local:
        install_dir = pyrolite_datafolder(subfolder="alphamelts") / "localinstall"
    else:
        install_dir = Path(install_dir)

    if not install_dir.exists():
        install_dir.mkdir(parents=True)

    if (link_dir is not None) and (not local):
        link_dir = Path(link_dir)
    else:
        link_dir = install_dir / "links"

    if not link_dir.exists():
        link_dir.mkdir(parents=True)

    if (eg_dir is not None) and (not local):
        eg_dir = Path(eg_dir)
    else:
        eg_dir = install_dir / "examples"

    if not eg_dir.exists():
        eg_dir.mkdir(parents=True)

    logger.info("Installing to {} from {}".format(install_dir, temp_dir))
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

            logger.info("\n" + p.stdout.decode("UTF-8"))
            assert p.returncode == 0

            # copy files from tempdir to install_dir
            regs = []  #'command', 'command_auto_file', 'path', 'perl']
            comms = ["install", "column_pick", "file_format", "run_alphamelts"]
            for (prefixes, ext) in [(regs, ".reg"), (comms, ".command")]:
                for prefix in prefixes:
                    temp_regpath = (temp_dir / prefix).with_suffix(ext)
                    install_regpath = install_dir / temp_regpath.name

                    shutil.copy(str(temp_regpath), str(install_regpath))
        elif native:
            # need to split into platforms
            egs = []
            for g in ["*.melts", "*.txt", "*.m ", "*.pdf"]:
                egs += list(temp_dir.glob(g))
            comms = [
                "column_pick.command",
                "file_format.command",
                "run_alphamelts.command",
            ]

            # getting the executable file
            if system == "Windows":
                alphafile = "alphamelts_win{}.exe".format(bits)
                alias = alphafile.replace(".exe", "")
            elif system == "Linux":
                if ("Microsoft" in platrel) or ("Microsoft" in platver):
                    alphafile = "alphamelts_wsl"
                    # with_readline
                else:
                    alphafile = "alphamelts_linux{}".format(bits)
                    # with_readline
            elif system == "Darwin":
                alphafile = "alphamelts_macosx{}".format(bits)
                # with_readline

            # copy executable; alias is used on windows to make sure run_command links to the exe
            alphaalias = [alphafile, "alphamelts"][system == "Windows"]
            copy_file(temp_dir / alphafile, install_dir / alphaalias, permissions=0o777)
            # copy examples
            for (target, files) in [(eg_dir, egs)]:
                for fn in files:
                    copy_file(temp_dir / fn.name, target / fn.name)

            # copycommand files
            for (target, files) in [(install_dir, [(temp_dir / i) for i in comms])]:
                for fn in files:  # executable files will need permissions
                    copy_file(temp_dir / fn.name, target / fn.name, permissions=0o777)

            # create links to the install directory
            linksrc = [(install_dir / i) for i in comms] + [install_dir / alphaalias]
            linkdest = [
                link_dir / "alphamelts"
                if (("alphamelts" in i.name) and (not "run" in i.name))
                else link_dir / i.name
                for i in linksrc
            ]
            if system == "Windows":  # create batch files to act as symlinks
                for src, dst in zip(linksrc, linkdest):
                    logger.debug("Creating batch file: {} <- {}".format(src, dst))
                    with open(str(dst.with_suffix(".bat")), "w") as fout:
                        fout.write("""@echo off\nperl "{}" %*""".format(src))
            else:  # create symlinks for command files and the exectuable
                for src, dst in zip(linksrc, linkdest):
                    logger.debug("Creating symlink: {} <- {}".format(src, dst))
                    if dst.exists():
                        os.remove(str(dst))  # remove old symlinks if present
                    os.symlink(str(src), str(dst))

    except AssertionError:
        raise AssertionError
    finally:
        if not keep_tempdir:
            remove_tempdir(temp_dir)
