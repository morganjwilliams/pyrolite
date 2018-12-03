from selenium.webdriver import Firefox, FirefoxProfile
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys as KEYS
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
import os, sys, time
import logging
from shutil import rmtree
from tempfile import mkdtemp
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from pyrolite.util.general import copy_file, stream_log, Timewith
from pyrolite.util.spatial import spatiotemporal_split

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()
ch = logging.StreamHandler()
fmt = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)
logger.setLevel(getattr(logging, "INFO"))


def check_alert(driver, tout=2):
    try:
        WebDriverWait(driver, tout).until(EC.alert_is_present())
        return True
    except TimeoutException as e:
        return False


def process_downloads(
    output_dir,
    from_tempdir=None,
    clear_tempdir=True,
    rename=None,
    timeout=100,
    expect_files=1,
):
    """
    Wait for downloads to finish with a specified timeout.

    Args
    ----
    directory : str
        The path to the folder where the files will be downloaded.
    timeout : int
        How many seconds to wait until timing out.

    """
    if not (Path(output_dir).exists() and Path(output_dir).is_dir()):
        output_dir.mkdir(parents=True)

    if from_tempdir:
        dir = from_tempdir
        assert Path(from_tempdir).exists() and Path(from_tempdir).is_dir()
    else:
        dir = output_dir

    seconds = 0
    dl_wait = True
    while dl_wait and seconds < timeout:
        time.sleep(1)
        dl_wait = False
        files = os.listdir(dir)
        if not files:
            dl_wait = True
        elif len(files) < expect_files:
            dl_wait = True
        else:
            for fname in files:
                if fname.endswith(".part"):
                    dl_wait = True

        seconds += 1

    for ix, f in enumerate(files):
        fpath = Path(dir) / f
        if rename is None:
            name = f
        else:
            if isinstance(rename, str):
                assert len(files) == 1
                name = rename
            elif isinstance(rename, list):
                name = rename[ix]
            else:
                name = f

        copy_file(fpath, (Path(output_dir) / name).with_suffix(fpath.suffix))

    if from_tempdir and clear_tempdir:
        for f in Path(from_tempdir).iterdir():
            if f.is_file():
                os.remove(f)
            elif f.is_dir():
                rmtree(f)


def waitfordownloadstart(dir, untilfiles=1, step=0.5, timeout=None):
    msg = "Waiting for download to start."
    logger.info(msg)

    if timeout is None:
        while len(os.listdir(dir)) < untilfiles:
            time.sleep(step)
    else:
        elapsed = 0.0
        while (len(os.listdir(dir)) < untilfiles) and (elapsed <= timeout):
            time.sleep(step)
            elapsed += step


def waitfordownloadend(dir, step=0.5):
    _ts = np.array([os.stat(os.path.join(dir, f)).st_mtime for f in os.listdir(dir)])
    stable = False

    with Timewith("Download") as T:
        while not stable:
            time.sleep(step)
            ts = np.array(
                [os.stat(os.path.join(dir, f)).st_mtime for f in os.listdir(dir)]
            )

            if all(ts == _ts):
                stable = True
            else:
                _ts = ts


class EarthChemDriver(object):
    def __init__(
        self,
        tout=60,
        tempdir=mkdtemp(),
        savedir=os.getcwd(),
        autosave=[
            "text/csv",
            "text/html",
            "text/plain",
            "application/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
            "application/download",
            "application/octet-stream",
        ],
    ):
        self.fp = FirefoxProfile()
        self.fp.set_preference("browser.download.folderList", 2)
        self.fp.set_preference("browser.download.manager.showWhenStarting", False)
        self.fp.set_preference("browser.download.dir", tempdir)
        self.fp.set_preference(
            "browser.helperApps.neverAsk.saveToDisk", ", ".join(autosave)
        )
        self.drv = Firefox(
            executable_path="C:/geckodriver/geckodriver.exe", firefox_profile=self.fp
        )
        self.dbs = [
            "navdat",
            "petdb",
            "seddb",
            "georoc",
            "usgs",
            "metpetdb",
            "earthchem",
            "darwin",
        ]
        self.tempdir = tempdir
        self.savedir = savedir
        self.tout = tout
        self.homepage = "http://ecp.iedadata.org/"
        self.config = {}
        self._configs = []
        self.home()

    def __enter__(self):
        return self

    def home(self):
        self.drv.get(self.homepage)
        assert "EarthChem" in self.drv.title
        self.built = True

    def close(self):
        self.drv.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def select_geologic_province(self, feature_name=None):
        pass

    def select_ocean_feature_name(self, feature_name=None):
        logger.info("Selecting Ocean Feature: {}".format(feature_name))

        # Enter the Ocean Feature Select Dialouge
        _of_select = '//tbody//tr//td//input[contains(@onclick, "setoceanfeature")]'
        WebDriverWait(self.drv, self.tout).until(
            EC.element_to_be_clickable((By.XPATH, _of_select))
        )
        ofselector = self.drv.find_element_by_xpath(_of_select)
        ofselector.send_keys(KEYS.RETURN)

        WebDriverWait(self.drv, self.tout).until(
            EC.presence_of_element_located((By.ID, "myselect"))
        )

        sel = Select(self.drv.find_element_by_id("myselect"))
        options = {option.text.upper().strip(): option for option in sel.options}
        assert feature_name.upper().strip() in options.keys()
        options[feature_name.upper().strip()].click()
        # Wait until submit present and submit
        WebDriverWait(self.drv, self.tout).until(
            EC.presence_of_element_located((By.NAME, "submit"))
        )
        submit = self.drv.find_elements_by_name("submit")[-1]
        submit.send_keys(KEYS.RETURN)

        # Wait until the ocean feature select is again clickable
        WebDriverWait(self.drv, self.tout).until(
            EC.element_to_be_clickable((By.XPATH, _of_select))
        )

        self.config["oceanfeature"] = feature_name

    def select_bounds(self, bounds=[[-180, 0], [-180, 45], [-90, 45], [-90, 0]]):
        poly = "; ".join(["{:01.6f} {:01.6f}".format(*xy) for xy in bounds])
        logger.info("Selecting Polygon: {}".format(poly))
        _loc_select = '//tbody//tr//td//input[contains(@onclick, "setlocation")]'
        WebDriverWait(self.drv, self.tout).until(
            EC.element_to_be_clickable((By.XPATH, _loc_select))
        )
        locationselector = self.drv.find_element_by_xpath(_loc_select)
        locationselector.send_keys(KEYS.RETURN)

        WebDriverWait(self.drv, self.tout).until(
            EC.presence_of_element_located((By.NAME, "latitude_longitude_data"))
        )
        location_text = self.drv.find_element_by_name("latitude_longitude_data")
        location_text.send_keys(poly)
        location_submit = self.drv.find_elements_by_name("submit")[-1]
        location_submit.send_keys(KEYS.RETURN)

        WebDriverWait(self.drv, self.tout).until(
            EC.element_to_be_clickable((By.XPATH, _loc_select))
        )

        self.config["bounds"] = bounds

    def select_DB(self, opts=[]):
        logger.info("Selecting DBs: {}".format(", ".join(opts)))
        if opts:
            opts = [i for i in opts if i in self.dbs]
        if not opts:
            opts = self.dbs

        for o in self.dbs:
            select_ = '//input[@name="{}" and @type="checkbox"]'.format(o)
            WebDriverWait(self.drv, self.tout).until(
                EC.element_to_be_clickable((By.XPATH, select_))
            )
            checkbox = self.drv.find_element_by_xpath(select_)
            if checkbox.is_selected():
                if not o in opts:
                    checkbox.click()
            else:
                if o in opts:
                    checkbox.click()

        self.config["db"] = opts

    def select_material(self, opt="BULK"):
        pass

    def continue_to_data_selection(self):
        # TODO: have fallbacks for no data available
        select_ = '//input[@name="clicky" and @type="submit"]'
        WebDriverWait(self.drv, self.tout).until(
            EC.element_to_be_clickable((By.XPATH, select_))
        )
        button = self.drv.find_element_by_xpath(select_)
        button.click()

        WebDriverWait(self.drv, self.tout).until(
            EC.presence_of_element_located((By.XPATH, '//h2["DATA ACCESS"]'))
        )
        msg = "Continuing to Data Selection"
        logger.info(msg)
        self._configs.append(self.config.copy())

    def get_chemical_data(
        self,
        searchopt="anyopt",  # anyopt | exactopt | alldata
        dispmode="text",  # text | html | xlsx
        rowtype="method",  # method | sample |
        itemselect="Show Standard Output Items",  # Show Items that Exist in Current Query
        showmethods=False,
        showunits=False,
        **kwargs
    ):
        self.current_files = len(os.listdir(self.tempdir))
        self.config["output"] = {}
        msg = "Getting Chemical Data"
        logger.info(msg)
        select_ = '//input[@name="advanced" and @type="submit" and @value="Get Chemical Data"]'
        WebDriverWait(self.drv, self.tout).until(
            EC.element_to_be_clickable((By.XPATH, select_))
        )
        button = self.drv.find_element_by_xpath(select_)
        button.click()

        for opt, value in [
            ("searchopt", searchopt),
            ("dispmode", dispmode),
            ("rowtype", rowtype),
        ]:
            select_ = '//input[@type="radio" and @name="{}" and @value="{}"]'.format(
                opt, value
            )
            radio = self.drv.find_element_by_xpath(select_)
            radio.click()
            self.config["output"][opt] = value

        for opt, value in [("showmethods", showmethods), ("showunits", showunits)]:
            select_ = '//input[@type="checkbox" and @name="{}"]'.format(opt)
            checkbox = self.drv.find_element_by_xpath(select_)
            if (checkbox.is_selected() and not value) or (
                (not checkbox.is_selected()) and value
            ):
                checkbox.click()
            self.config["output"][opt] = value

        for buttonselect in [
            '//input[@type="button" and @value="{}"]'.format(itemselect),
            '//input[@type="submit" and @value="Go to Data"]',
        ]:
            button = self.drv.find_element_by_xpath(buttonselect)
            button.click()

        if check_alert(self.drv):
            alert = self.drv.switch_to.alert()
            alert.accept()
            self.drv.switchTo().window(MainWindow)

    def wait_and_process_download(
        self,
        names=None,
        output_dir=Path(os.getcwd()) / "data",
        clear_tempdir=True,
        timeout=None,
    ):
        # wait for file to download
        waitfordownloadstart(
            self.tempdir, untilfiles=self.current_files + 1, timeout=timeout
        )
        waitfordownloadend(self.tempdir, step=5)
        process_downloads(
            output_dir,
            rename=names,
            from_tempdir=self.tempdir,
            clear_tempdir=clear_tempdir,
        )


# %% --------

ECD = EarthChemDriver(tout=30)


def get_ocean_features(ecd, features):
    for s in features:
        # avoid redownloading
        if not (Path("./") / "data" / "ocean_features" / "{}.xlsx".format(s)).exists():
            ecd.select_ocean_feature_name(feature_name=s)
            ecd.select_DB([])  # all databases
            ecd.continue_to_data_selection()
            ecd.get_chemical_data(dispmode="xlsx")
            ecd.wait_and_process_download(
                names=s, output_dir=Path("./") / "data" / "ocean_features", timeout=60
            )
        ecd.home()


def get_grid(ecd):
    for s in spatiotemporal_split(segments=2, lat=(-10, 10), long=(-10, 10)):
        e, w, n, s = s["east"], s["west"], s["north"], s["south"]
        bounds = [[w, s], [w, n], [e, n], [e, s]]

        ecd.select_bounds(bounds=bounds)
        ecd.select_DB(["petdb"])  # all databases
        ecd.continue_to_data_selection()
        ecd.get_chemical_data(dispmode="xlsx")
        ecd.wait_and_process_download(names=s)
        ecd.home()


OF = open("ocean_features.txt").read().split("\n")
get_ocean_features(ECD, OF)

ECD.close()

# %% --

# %% -
tempdir = ECD.tempdir
"""
names = [
    "B_{}".format(";".join(["{:+01.6g},{:+01.6g}".format(*xs) for xs in c["bounds"]]))
    for c in ECD._configs
]
"""
names = ["Mid-atlantic Ridge"]
outdir = Path(os.getcwd()) / "data"
process_downloads(outdir, rename=names, from_tempdir=Path(tempdir), clear_tempdir=False)

pths = [outdir / f for f in os.listdir(tempdir)]
