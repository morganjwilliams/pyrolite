"""
This file contains functions for automated execution, plotting and reporting from
alphamelts 1.9.
"""
import logging
import os, sys, platform
from pathlib import Path
import subprocess
import shlex
from ..general import copy_file

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def make_meltsfolder(meltsfile, title, dir=None, env="./alphamelts_default_env.txt"):
    """
    Create a folder for a given meltsfile, including the default environment file.

    Parameters
    -----------
    meltsfile : :class:`str`
        String containing meltsfile info.
    title : :class:`str`
        Title of the experiment
    dir : :class:`str` | :class:`pathlib.Path`
        Path to the base directory to create melts folders in.
    env : :class:`str` | :class:`pathlib.Path`
        Path to a specific environment file to use as the default environment for the
        experiment. Pass this to alphamelts with
        :code:`run_alphamelts.command -f <envfile>`.
    """
    if dir is None:
        dir = Path("./")
    else:
        dir = Path(dir)
    filepath = dir / title / (title + ".melts")
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    with open(filepath, "w") as f:
        f.write(meltsfile)

    env = Path(env)
    copy_file(env, filepath.parent / env.name)
