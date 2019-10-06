"""
This file contains functions for automated execution, plotting and reporting from
alphamelts 1.9.
"""
import os, sys, platform
import time, datetime
from tqdm import tqdm
import stat
import psutil
import logging
import time
from pathlib import Path
import subprocess
import threading
import queue
import shlex
from ...util.multip import combine_choices
from ...util.general import get_process_tree
from ...util.meta import pyrolite_datafolder, ToLogger
from ...geochem.ind import common_elements, common_oxides
from .tables import MeltsOutput
from .parse import read_envfile, read_meltsfile
from .meltsfile import to_meltsfile
from .env import MELTS_Env


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__abbrv__ = {"fractionate solids": "frac", "isobaric": "isobar"}

__chem__ = common_elements(as_set=True) | common_oxides(as_set=True)


def exp_name(exp):
    """
    Derive an experiment name from an experiment configuration dictionary.

    Parameters
    ------------
    exp : :class:`dict`
        Dictionary of parameters and their specific values to derive an experiment name
        from.

    Todo
    ------

        This is a subset of potential parameters, need to expand to ensure uniqueness of naming.
    """

    mode = "".join([__abbrv__.get(m, m) for m in exp["modes"]])

    fo2 = exp.get("Log fO2 Path", "")
    fo2d = exp.get("Log fO2 Delta", "")
    p0, p1, t0, t1 = "", "", "", ""
    if exp.get("Initial Pressure", None) is not None:
        p0 = "{:d}".format(int(exp["Initial Pressure"] / 1000))
    if exp.get("Final Pressure", None) is not None:
        p1 = "-{:d}".format(int(exp["Final Pressure"] / 1000))
    if exp.get("Initial Temperature", None) is not None:
        t0 = "{:d}".format(int(exp["Initial Temperature"]))
    if exp.get("Final Temperature", None) is not None:
        t1 = "-{:d}".format(int(exp["Final Temperature"]))
    chem = "-".join(
        ["{}@{}".format(k, v) for k, v in exp.get("modifychem", {}).items()]
    )
    suppress = "-".join(["no_{}".format(v) for v in exp.get("Suppress", {})])
    return "{}{}{}kbar{}{}C{}{}{}{}".format(
        mode, p0, p1, t0, t1, fo2d, fo2, chem, suppress
    )


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
    title = str(title)  # need to pathify this!
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
        log=logger.debug,
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
                    / "links"
                    / "run_alphamelts.command"
                )

            executable = local_run
            self.log(
                "Using local executable: {} @ {}".format(
                    executable.name, executable.parent
                )
            )

        executable = Path(executable)
        self.exname = str(executable.name)
        self.executable = str(executable)
        st = os.stat(self.executable)
        assert bool(stat.S_IXUSR), "User needs execution permission."
        self.run = []

        self.run.append(self.executable)  # executable file

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

    @property
    def callstring(self):
        """Get the call string such that analyses can be reproduced manually."""
        return " ".join(["cd", str(self.fromdir), "&&"] + self.run)

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
            cwd=str(self.fromdir),
            close_fds=(os.name == "posix"),
        )
        self.process = subprocess.Popen(self.run, **config)
        logger.debug("Process Started with ID {}".format(self.process.pid))
        logger.debug("Reproduce using: {}".format(self.callstring))
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

    def wait(self, step=1.0):
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

    def write(self, messages, wait=True, log=False):
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
            msg = (str(message).strip() + str(os.linesep)).encode("utf-8")
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
        self.alphamelts_ex = []
        try:
            for p in get_process_tree(self.process.pid):
                if "alpha" in p.name():
                    self.alphamelts_ex.append(p)
            self.write("0")
            time.sleep(0.5)
        except (ProcessLookupError, psutil.NoSuchProcess):
            logger.warning("Process terminated unexpectedly.")

        try:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=0.2)
        except ProcessLookupError:
            logger.debug("Process terminated successfully.")

        self.cleanup()

    def cleanup(self):
        for p in self.alphamelts_ex:  # kill the children executables
            try:
                # kill the alphamelts executable which can hang
                logger.debug("Terminating {}".format(p.name()))
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
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
        self.mp = MeltsProcess(
            meltsfile=str(self.title) + ".melts",
            env="environment.txt",
            fromdir=str(self.folder),
        )
        self.mp.write([3, [0, 1][superliquidus_start], 4], wait=True, log=log)
        self.mp.terminate()

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
        * Calculate relative number of calculations to be performed for the est duration

            This is currently about correct for an isobaric calcuation at 10 degree
            temperature steps over few hundred degrees - but won't work for different
            T steps.
    """

    def __init__(
        self,
        comp_df,
        fromdir=Path("./"),
        default_config={},
        grid={},
        env=None,
        logger=logger,
    ):
        """

        Parameters
        -----------
        comp_df : :class:`pandas.DataFrame`
            Dataframe of compositions.
        default_config : :class:`dict`
            Dictionary of default parameters.
        grid : class:`dict`
            Dictionary of parameters to systematically vary.
        """
        self.logger = logger
        self.dir = fromdir
        self.default = default_config
        self.grid = [{}]
        self.grid += [i for i in combine_choices(grid) if i not in self.grid]
        self.env = env or MELTS_Env()
        self.compositions = comp_df
        exps = [{**self.default, **ex} for ex in self.grid]
        self.experiments = [
            (n, e) for (e, n) in zip(exps, [exp_name(ex) for ex in exps])
        ]

    def run(self, overwrite=False, exclude=[], superliquidus_start=True):
        self.started = time.time()
        experiments = self.experiments
        if not overwrite:
            experiments = [
                (n, e) for (n, e) in experiments if not (self.dir / n).exists()
            ]

        self.logger.info(
            "Starting {} Calculations for {} Compositions.".format(
                len(experiments), self.compositions.index.size
            )
        )
        self.logger.info(
            "Estimated Time: {}".format(
                datetime.timedelta(
                    seconds=len(experiments) * self.compositions.index.size * 6
                )  # 6s/run
            )
        )
        paths = []
        for name, exp in tqdm(experiments, file=ToLogger(self.logger), mininterval=2):
            edf = self.compositions.copy()
            if "Title" not in edf.columns:
                edf["Title"] = [
                    "{}-{}".format(name, ix) for ix in edf.index.values.astype(str)
                ]
            if "modifychem" in exp:
                for k, v in exp[
                    "modifychem"
                ].items():  # this isn't quite corect , you'd need to modify everything else too
                    edf[k] = v
                cc = [i for i in edf.columns if i in __chem__]
                edf.loc[:, cc] = renormalise(edf.loc[:, cc])
            P = exp["Initial Pressure"]
            T0, T1 = exp["Initial Temperature"], exp["Final Temperature"]
            edf["Initial Pressure"] = P
            edf["Initial Temperature"] = T0
            edf["Final Temperature"] = T1
            for par in ["Log fO2 Path", "Log fO2 Delta", "Limit coexisting"]:
                if par in exp:
                    edf[par] = exp[par]

            if "Suppress" in exp:
                edf["Suppress"] = [exp["Suppress"] for i in range(edf.index.size)]

            exp_exclude = exclude
            if "exclude" in exp:
                exp_exclude += exp["exclude"]

            expdir = self.dir / name  # experiment dir

            paths.append(expdir)

            for ix in edf.index:
                meltsfile = to_meltsfile(
                    edf.loc[ix, :], modes=exp["modes"], exclude=exp_exclude
                )
                M = MeltsExperiment(
                    meltsfile=meltsfile, title=edf.Title[ix], env=self.env, dir=expdir
                )
                M.run(superliquidus_start=superliquidus_start)
        self.duration = datetime.timedelta(seconds=time.time() - self.started)
        self.logger.info("Calculations Complete after {}".format(self.duration))
        self.paths = paths

    def cleanup(self):
        pass
