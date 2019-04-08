"""
This file contains functions for automated execution, plotting and reporting from
alphamelts 1.9.
"""
import logging
import time
import os, sys, platform
from pathlib import Path
import subprocess
import threading
import queue
import shlex
from ..general import copy_file

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def enqueue_output(out, queue):
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


def make_meltsfolder(meltsfile, title, dir=None, env="./alphamelts_default_env.txt"):
    """
    Create a folder for a given meltsfile, including the default environment file.
    From this folder, pass these to alphamelts with
    :code:`run_alphamelts.command -m <meltsfile> -f <envfile>`.

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
        experiment.

    Returns
    --------
    :class:`pathlib.Path`
        Path to melts folder.
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

    return filepath.parent # return the folder name


class MeltsExperiment(object):
    """
    Melts Experiment Object

    Currently in-development for automation of calling melts across a grid of
    parameters.

    Todo
    ----
        * Automated creation of folders for experiment results
        * Being able to run melts
        * Compressed export/save function
    """

    def __init__(self):
        pass

    def run(self, meltsfile=None, env=None):
        """
        Call 'run_alphamelts.command'.
        """
        args = ["run_alphamelts.command"]
        meltsargs = []
        if meltsfile is not None:
            args += ["-m", str(meltsfile)]
            meltsargs += ["1", str(meltsfile)]  # enter meltsfile
        if env is not None:
            args += ["-f", str(env)]
        if system == "Windows":
            pass
        elif system == "Linux":
            if ("Microsoft" in release) or ("Microsoft" in version):
                pass
            else:
                pass
        elif system == "Darwin":
            pass

        meltsargs += ["3", "1"]  # try superliquidus start
        subprocess.check_call(args)


class MeltsProcess(object):
    def __init__(
        self,
        executable="run_alphamelts.command",
        env="alphamelts_default_env.txt",
        meltsfile=None,
        fromdir=r"./",
    ):
        """
        Parameters
        ----------
        executable : :class:`str`
            Executable to run. In this case defaults to the `run_alphamelts.command `
            script.
        env : :class:`str` | :class:`pathlib.Path`
            Environment file to use.
        meltsfile : :class:`str` | :class:`pathlib.Path`
            Path to meltsfile to use for calculations.
        fromdir : :class:`str` | :class:`pathlib.Path`
            Directory to use as the working directory for the execution.
        """
        self.env = None
        self.meltsfile = None
        self.fromdir = None
        if fromdir is not None:
            logger.debug("Setting working directory: {}".format(fromdir))
            fromdir = Path(fromdir)
            assert fromdir.exists() and fromdir.is_dir()
            self.fromdir = Path(fromdir)

        self.executable = [executable]  # executable file

        self.init_args = []  # initial arguments to pass to the exec before returning
        if meltsfile is not None:
            logger.debug("Setting meltsfile: {}".format(meltsfile))
            self.meltsfile = Path(meltsfile)
            self.executable += ["-m"]
            self.executable += [str(meltsfile)]
            self.init_args += ["1", str(meltsfile)]  # enter meltsfile
        if env is not None:
            logger.debug("Setting environment file: {}".format(env))
            self.env = Path(env)
            self.executable += ["-f", str(env)]

        self.start()
        time.sleep(0.5)
        for a in self.init_args:
            logger.debug("Passing Inital Variable: " + a)
            self.write(a)

    def log_output(self):
        logger.info("\n" + self.read())

    def start(self):
        logger.info("Starting Melts Process with: " + " ".join(self.executable))
        config = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            cwd=self.fromdir
        )
        self.process = subprocess.Popen(self.executable, **config)
        self.q = queue.Queue()
        self.T = threading.Thread(
            target=enqueue_output, args=(self.process.stdout, self.q)
        )
        self.T.daemon = True  # kill when process dies
        self.T.start()  # start the output thread
        return self.process

    def read(self):
        lines = []
        while not self.q.empty():
            lines.append(self.q.get_nowait().decode())
        return "".join(lines)

    def wait(self, step=0.5):
        """
        Wait until process.stdout stops.
        """
        while True:
            size =  self.q.qsize()
            time.sleep(step)
            if size == self.q.qsize():
                break

    def write(self, *messages, wait=False, log=False):
        for message in messages:
            msg = "{}\n".format(str(message).strip()).encode("utf-8")
            logger.info("\n" + str(message))
            self.process.stdin.write(msg)
            self.process.stdin.flush()
            if wait:
                self.wait()
            if log:
                self.process.log_output()

    def terminate(self):
        self.write("0")
        try:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=0.2)
        except ProcessLookupError:
            logger.debug('Process Terminated Successfully')
