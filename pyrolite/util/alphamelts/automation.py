"""
This file contains functions for automated execution, plotting and reporting from
alphamelts 1.9.
"""
import os, sys, platform
import logging
import time
from pathlib import Path
import subprocess
import threading
import queue
import shlex
from ..general import copy_file
from ..meta import pyrolite_datafolder
from .tables import MeltsOutput

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def get_experiments_summary(dir, **kwargs):
    """
    Aggregate alphaMELTS experiment results across folders.

    Parameters
    -----------
    dir : :class:`str` | :class:`pathlib.Path` | :class:`list`
        Directory to aggregate folders from, or list of folders.

    Returns
    --------
    :class:`dict`
    """
    if isinstance(dir, list):
        target_folders = dir
    else:
        dir = Path(dir)
        target_folders = [p for p in dir.iterdir() if p.is_dir()]
    summary = {}
    for ix, t in enumerate(target_folders):
        output = MeltsOutput(t, **kwargs)
        summary[output.title] = {}
        summary[output.title]["phases"] = {
            i[: i.find("_")] if i.find("_") > 0 else i for i in output.phasenames
        }
        summary[output.title]["output"] = output
    return summary


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

    return filepath.parent  # return the folder name


class MeltsExperiment(object):
    """
    Melts Experiment Object. Currently in-development for automation of calling melts
    across a grid of parameters.

    Todo
    ----
        * Automated creation of folders for experiment results (see :func:`make_meltsfolder`)
        * Being able to run melts in an automated way (see :class:`MeltsProcess`)
        * Compressed export/save function
    """

    def __init__(self, dir="./"):
        self.dir = dir
        self.log = []

    def run(self, meltsfile=None, env=None):
        """
        Call 'run_alphamelts.command'.
        """
        mp = MeltsProcess(
            env=env,
            meltsfile=meltsfile,
            fromdir=self.dir,
            log=lambda x: self.log.append(x),
        )


def enqueue_output(out, queue):
    """
    Send output to a queue.

    Parameters
    -----------
    out
        Readable output object.
    queue : :class:`queue.Queue`
        Queue to send ouptut to.
    """
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


class MeltsProcess(object):
    def __init__(
        self,
        executable=None,
        env="alphamelts_default_env.txt",
        meltsfile=None,
        fromdir=r"./",
        log=print,
    ):
        """
        Parameters
        ----------
        executable : :class:`str` | :class:`pathlib.Path`
            Executable to run. Enter path to the the `run_alphamelts.command `
            script. Falls back to local installation if no exectuable is specified
            and a local instllation exists.
        env : :class:`str` | :class:`pathlib.Path`
            Environment file to use.
        meltsfile : :class:`str` | :class:`pathlib.Path`
            Path to meltsfile to use for calculations.
        fromdir : :class:`str` | :class:`pathlib.Path`
            Directory to use as the working directory for the execution.
        log : :class:`callable`
            Function for logging output.

        Todo
        -----
            * Recognise errors from stdout
            * Input validation (graph of available options vs menu level)
            * Logging of failed runs
            * Facilitation of interactive mode upon error
            * Error recovery methods (e.g. change the temperature)
        """
        self.env = None
        self.meltsfile = None
        self.fromdir = None
        self.log = log
        if fromdir is not None:
            self.log("Setting working directory: {}".format(fromdir))
            fromdir = Path(fromdir)
            assert fromdir.exists() and fromdir.is_dir()
            self.fromdir = Path(fromdir)

        if executable is None:
            # check for local install
            if platform.system() == "Windows":
                local_run = (
                    pyrolite_datafolder(subfolder="alphamelts")
                    / "localinstall"
                    / "links"
                    / "run_alphamelts.bat"
                )

            else:
                local_run = (
                    pyrolite_datafolder(subfolder="alphamelts")
                    / "localinstall"
                    / "run_alphamelts.command"
                )
            if local_run.exists() and local_run.is_file():
                executable = local_run
                self.log("Using local executable meltsfile: {}".format(executable.name))

        assert (
            executable is not None
        ), "Need to specify an installable or perform a local installation of alphamelts."
        self.executable = [str(executable)]  # executable file

        self.init_args = []  # initial arguments to pass to the exec before returning
        if meltsfile is not None:
            self.log("Setting meltsfile: {}".format(meltsfile))
            self.meltsfile = Path(meltsfile)
            self.executable += ["-m"]
            self.executable += [str(meltsfile)]
            self.init_args += ["1", str(meltsfile)]  # enter meltsfile
        if env is not None:
            self.log("Setting environment file: {}".format(env))
            self.env = Path(env)
            self.executable += ["-f", str(env)]

        self.start()
        time.sleep(0.5)
        for a in self.init_args:
            self.log("Passing Inital Variable: " + a)
            self.write(a)

    def log_output(self):
        """
        Log output to the configured logger.
        """
        self.log("\n" + self.read())

    def start(self):
        """
        Start the process.

        Returns
        --------
        :class:`subprocess.Popen`
            Melts process object.
        """
        self.log("Starting Melts Process with: " + " ".join(self.executable))
        config = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.fromdir,
            close_fds=(os.name == "posix"),
        )
        self.process = subprocess.Popen(self.executable, **config)
        self.q = queue.Queue()

        self.T = threading.Thread(
            target=enqueue_output, args=(self.process.stdout, self.q)
        )

        self.T.daemon = True  # kill when process dies
        self.T.start()  # start the output thread

        self.errq = queue.Queue()
        self.errT = threading.Thread(  # separate thread for error reporting
            target=enqueue_output, args=(self.process.stderr, self.errq)
        )
        self.errT.daemon = True  # kill when process dies
        self.errT.start()  # start the err output thread
        return self.process

    def read(self):
        """
        Read from the output queue.

        Returns
        ---------
        :class:`str`
            Concatenated output from the output queue.
        """
        lines = []
        while not self.q.empty():
            lines.append(self.q.get_nowait().decode())
        return "".join(lines)

    def wait(self, step=0.5):
        """
        Wait until addtions to process.stdout stop.

        Parameters
        -----------
        step : :class:`float`
            Step in seconds at which to check the stdout queue.
        """
        while True:
            size = self.q.qsize()
            time.sleep(step)
            if size == self.q.qsize():
                break

    def write(self, *messages, wait=False, log=False):
        """
        Send commands to the process.

        Parameters
        -----------
        messages
            Sequence of messages/commands to send.
        wait : :class:`bool`
            Whether to wait for process.stdout to finish.
        log : :class:`bool`
            Whether to log output to the logger.
        """
        for message in messages:
            msg = "{}\n".format(str(message).strip()).encode("utf-8")
            self.process.stdin.write(msg)
            self.process.stdin.flush()
            if wait:
                self.wait()
            if log:
                self.log(message)
                self.log_output()

    def terminate(self):
        """
        Terminate the process.

        Notes
        -------
            * Will likely terminate as expected using the command '0' to exit.
            * Otherwise will attempt to cleanup the process.
        """
        self.write("0")
        time.sleep(0.5)
        try:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=0.2)
        except ProcessLookupError:
            logger.debug("Process Terminated Successfully")
