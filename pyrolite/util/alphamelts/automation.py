"""
This file contains functions for automated execution, plotting and reporting from
alphamelts 1.9.
"""
import os, sys, platform
import stat
import psutil
import logging
import time
from pathlib import Path
import subprocess
import threading
import queue
import shlex
from ..general import get_process_tree
from ..meta import pyrolite_datafolder
from .tables import MeltsOutput
from .parse import read_envfile, read_meltsfile
from .env import MELTS_Env


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


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

    Todo
    ------
        * Options for naming environment files
    """
    if dir is None:
        dir = Path("./")
    else:
        dir = Path(dir)
    title = str(title) # need to pathify this!
    experiment_folder = dir / title
    if not experiment_folder.exists():
        experiment_folder.mkdir(parents=True)

    meltsfile, mpath = read_meltsfile(meltsfile)
    with open(str(experiment_folder / (title + ".melts")), "w") as f:
        f.write(meltsfile)

    env, epath = read_envfile(env, unset_variables=False)
    with open(str(experiment_folder / "environment.txt"), "w") as f:
        f.write(env)

    return experiment_folder  # return the folder name


def _enqueue_output(out, queue):
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

        Notes
        ------
            * Need to specify an exectuable or perform a local installation of alphamelts.
            * Need to get full paths for melts files, directories etc
        """
        self.env = None
        self.meltsfile = None
        self.fromdir = None  # default to None, runs from cwd
        self.log = log
        if fromdir is not None:
            self.log("Setting working directory: {}".format(fromdir))
            fromdir = Path(fromdir)
            try:
                assert fromdir.exists() and fromdir.is_dir()
            except AssertionError:
                fromdir.mkdir(parents=True)
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

            executable = local_run
            self.log("Using local executable meltsfile: {}".format(executable.name))

        executable = Path(executable)
        assert executable.exists() and executable.is_file()
        self.exname = str(executable.name)

        st = os.stat(executable)
        assert bool(stat.S_IXUSR), "User needs execution permission."
        self.executable = str(executable)
        self.run = [self.executable]  # executable file

        self.init_args = []  # initial arguments to pass to the exec before returning
        if meltsfile is not None:
            self.log("Setting meltsfile: {}".format(meltsfile))
            self.meltsfile = Path(meltsfile)
            self.run += ["-m", str(self.meltsfile)]
            self.init_args += ["1", str(self.meltsfile)]  # enter meltsfile
        if env is not None:
            self.log("Setting environment file: {}".format(env))
            self.env = Path(env)
            self.run += ["-f", str(env)]

        self.start()
        time.sleep(0.5)
        self.log("Passing Inital Variables: " + " ".join(self.init_args))
        self.write(self.init_args)

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
        self.log(
            "Starting Melts Process with: " + " ".join([self.exname] + self.run[1:])
        )
        config = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.fromdir,
            close_fds=(os.name == "posix"),
        )
        self.process = subprocess.Popen(self.run, **config)
        logger.debug("Process Started with ID {}".format(self.process.pid))
        # Queues and Logging
        self.q = queue.Queue()
        self.T = threading.Thread(
            target=_enqueue_output, args=(self.process.stdout, self.q)
        )
        self.T.daemon = True  # kill when process dies
        self.T.start()  # start the output thread

        self.errq = queue.Queue()
        self.errT = threading.Thread(  # separate thread for error reporting
            target=_enqueue_output, args=(self.process.stderr, self.errq)
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

    def write(self, messages, wait=False, log=False):
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
            msg = (str(message).strip()+str(os.linesep)).encode("utf-8")
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
        alphamelts_ex = [
            p for p in get_process_tree(self.process.pid) if "alpha" in p.name()
        ]
        self.write("0")
        time.sleep(0.5)
        try:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=0.2)
        except ProcessLookupError:
            logger.debug("Process Terminated Successfully")

        for p in alphamelts_ex:  # kill the children executables
            try:
                # kill the alphamelts executable which can hang
                logger.debug("Terminating {}".format(p.name()))
                p.kill()
            except psutil.NoSuchProcess:
                pass


class MeltsExperiment(object):
    """
    Melts Experiment Object. For a single call to melts, with one set of outputs.
    Autmatically creates the experiment folder, meltsfile and environment file, runs
    alphaMELTS and collects the results.

    Todo
    ----
        * Automated creation of folders for experiment results (see :func:`make_meltsfolder`)
        * Being able to run melts in an automated way (see :class:`MeltsProcess`)
        * Compressed export/save function
        * Post-processing functions for i) validation and ii) plotting
    """

    def __init__(self, title="MeltsExperiment", dir="./", meltsfile=None, env=None):
        self.title = title
        self.dir = dir
        self.log = []

        if meltsfile is not None:
            self.set_meltsfile(meltsfile)
        if env is not None:
            self.set_envfile(env)
        else:
            self.set_envfile(MELTS_Env())

        self._make_folder()

    def set_meltsfile(self, meltsfile, **kwargs):
        """
        Set the meltsfile for the experiment.

        Parameters
        ------------
        meltsfile : :class:`pandas.Series` | :class:`str` | :class:`pathlib.Path`
            Either a path to a valid melts file, a :class:`pandas.Series`, or a
            multiline string representation of a melts file object.
        """
        self.meltsfile, self.meltsfilepath = read_meltsfile(meltsfile)

    def set_envfile(self, env):
        """
        Set the environment for the experiment.

        Parameters
        ------------
        env : :class:`str` | :class:`pathlib.Path`
            Either a path to a valid environment file, a :class:`pandas.Series`, or a
            multiline string representation of a environment file object.
        """
        self.envfile, self.envfilepath = read_envfile(env)

    def _make_folder(self):
        """
        Create the experiment folder.
        """
        self.folder = make_meltsfolder(
            meltsfile=self.meltsfile, title=self.title, dir=self.dir, env=self.envfile
        )
        self.meltsfilepath = self.folder / (self.title + ".melts")
        self.envfilepath = self.folder / "environment.txt"

    def run(self, log=False, superliquidus_start=True):
        """
        Call 'run_alphamelts.command'.
        """
        mp = MeltsProcess(
            meltsfile=(self.title + ".melts"),
            env="environment.txt",
            fromdir=self.folder,
        )
        mp.write([3, [0, 1][superliquidus_start], 4], wait=True, log=log)
        mp.terminate()

    def cleanup(self):
        pass


class MeltsBatch(object):
    """
    Batch of :class:`MeltsExperiment`, which may represent evaluation over a grid of
    parameters or configurations.

    Todo
    ------
        * Can start with a single composition or multiple compositions in a dataframe
        * Enable grid search for individual parameters
        * Improved output logging/reporting
    """

    def __init__(self):
        pass
