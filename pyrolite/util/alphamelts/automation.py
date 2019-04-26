"""
This file contains functions for automated execution, plotting and reporting from
alphamelts 1.9.
"""
import os, sys, platform
import psutil
import logging
import time
from pathlib import Path
import subprocess
import threading
import queue
import shlex
from ..general import copy_file, get_process_tree
from ..meta import pyrolite_datafolder
from .tables import MeltsOutput
from .meltsfile import to_meltsfile
from .env import MELTS_Env

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
    title = str(title)
    experiment_folder = dir / title
    if not experiment_folder.exists():
        experiment_folder.mkdir(parents=True)

    meltstarget = experiment_folder / (title + ".melts")
    if isinstance(meltsfile, Path):
        _src = meltsfile
        copy_file(_src, meltstarget)
    elif isinstance(meltsfile, str):
        if "\n" in meltsfile:  # multiline string
            with open(meltstarget, "w") as f:
                f.write(meltsfile)
        else:  # path, copy file
            _src = Path(meltsfile)
            copy_file(_src, meltstarget)

    if isinstance(env, Path):
        copy_file(env, experiment_folder / env.name)
    elif isinstance(env, str):
        if "\n" in env:  # multiline string
            with open(experiment_folder / env.name, "w") as f:
                f.write(env)
        else:  # path, copy file
            copy_file(Path(env), experiment_folder / Path(env).name)

    return experiment_folder  # return the folder name


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


def _file_from_obj(fileobj):
    """
    Read in file data either from a file path or a string.

    Parameters
    ------------
    fileobj : :class:`str` | :class:`pathlib.Path`
        Either a path to a valid file, or a multiline string representation of a
        file object.

    Returns
    --------
    file : :class:`str`
        Multiline string representation of a file.
    path
        Path to the original file, if it exists.

    Notes
    ------
        This function deconvolutes the possible ways in which one can pass either
        a file, or reference to a file.

    Todo
    ----
        * Could be passed an open file object
    """
    path, file = None
    if isinstance(fileobj, Path):
        path = fileobj
    elif isinstance(fileobj, str):
        if len(re.split("[\r\n]", fileobj)) > 1:  # multiline string passed as a file
            file = fileobj
        else:  # path passed as a string
            path = fileobj
    else:
        pass
    if (path is not None) and (file is None):
        file = open(path).read()

    assert file is not None  # can't not have a meltsfile
    return file, path


def _read_meltsfile(meltsfile):
    """
    Read in a melts file from a :class:`~pandas.Series`, :class:`~pathlib.Path` or
    string.

    Parameters
    ------------
    meltsfile : :class:`pandas.Series` | :class:`str` | :class:`pathlib.Path`
        Either a path to a valid melts file, a :class:`pandas.Series`, or a
        multiline string representation of a melts file object.

    Returns
    --------
    file : :class:`str`
        Multiline string representation of a meltsfile.
    path
        Path to the original file, if it exists.

    Notes
    ------
        This function deconvolutes the possible ways in which one can pass either
        a meltsfile, or reference to a meltsfile.
    """
    path, file = None, None
    if isinstance(meltsfile, pd.Series):
        file = to_meltsfile(meltsfile, **kwargs)
    else:
        file, path = _file_from_obj(meltsfile)
    return file, path


def _read_envfile(envfile):
    """
    Read in a environment file from a  :class:`~pyrolite.util.alphamelts.env.MELTS_Env`,
    :class:`~pathlib.Path` or string.

    Parameters
    ------------
    envfile : :class:`~pyrolite.util.alphamelts.env.MELTS_Env` | :class:`str` | :class:`pathlib.Path`
        Either a path to a valid environment file, a :class:`pandas.Series`, or a
        multiline string representation of a environment file object.

    Returns
    --------
    file : :class:`str`
        Multiline string representation of an environment file.
    path
        Path to the original file, if it exists.
    """
    path, file = None, None
    if isinstance(envfile, MELTS_Env):
        file = MELTS_Env.to_envfile(**kwargs)
    else:
        file, path = _file_from_obj(envfile)
    return file, path


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
            if local_run.exists() and local_run.is_file():
                executable = local_run
                self.log("Using local executable meltsfile: {}".format(executable.name))

        assert (
            executable is not None
        ), "Need to specify an installable or perform a local installation of alphamelts."

        if isinstance(executable, Path):
            self.exname = str(executable.name)
        else:
            self.exname = str(executable)
        self.executable = str(executable)
        self.run = [str(self.executable)]  # executable file

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
        for a in self.init_args:
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

    def __init__(self, dir="./", meltsfile=None, env=None, exec=None):
        self.dir = dir
        self.log = []
        self.env = MELTS_Env()
        if not env is None:
            pass  # parse env

    def set_meltsfile(self, meltsfile, **kwargs):
        """
        Set the meltsfile for the experiment.

        Parameters
        ------------
        meltsfile : :class:`pandas.Series` | :class:`str` | :class:`pathlib.Path`
            Either a path to a valid melts file, a :class:`pandas.Series`, or a
            multiline string representation of a melts file object.
        """
        self.meltsfile, self.meltsfilepath = _read_meltsfile(meltsfile)

    def set_envfile(self, env):
        """
        Set the environment for the experiment.

        Parameters
        ------------
        env : :class:`str` | :class:`pathlib.Path`
            Either a path to a valid environment file, a :class:`pandas.Series`, or a
            multiline string representation of a environment file object.
        """
        self.envfilepath = env

    def _make_folder(self, startdir=None):
        """
        Create the experiment folder.
        """
        self.folder = make_meltsfolder(
            meltsfile=self.meltsfile,
            title=self.title,
            dir=startdir,
            env=self.envfilepath,
        )

    def run(self, meltsfile=None, env=None, log=False, superliquidus_start=True):
        """
        Call 'run_alphamelts.command'.
        """
        mp = MeltsProcess(
            meltsfile=meltsfile or self.meltsfilepath,
            env=env or self.envfilepath,
            fromdir=self.folder,
            log=lambda x: self.log.append(x),
        )
        mp.write(3, [0, 1][superliquidus_start], 4, wait=True, log=log)
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
