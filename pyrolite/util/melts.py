import os, sys, platform
import subprocess
from pathlib import Path
import io
import requests
from xml.etree import ElementTree as ET
import xmljson
import dicttoxml
from environs import Env
import zipfile
import logging

from .general import copy_file, extract_zip, remove_tempdir


def check_perl():
    """Checks whether perl is installed on the system."""
    try:
        p = subprocess.check_output("which perl")
        returncode = 0
    except subprocess.CalledProcessError as e:
        output = e.output
        returncode = e.returncode

    return returncode == 0


def download_melts(directory):
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

    system = platform.system()
    release = platform.release()
    version = platform.version()
    bits, linkage = platform.architecture()
    bits = bits[:2]

    zipsource = "https://magmasource.caltech.edu/alphamelts/zipfiles/"
    if system =='Linux':
        if ('Microsoft' in release) or ('Microsoft' in version):
           url = zipsource + "wsl_alphamelts_1-8.zip"
        else:
           url = zipsource + "linux_alphamelts_1-8.zip"

    elif system == 'Darwin':
        url = zipsource + "macosx_alphamelts_1-8.zip"
    elif system == 'Windows':
        url = zipsource + "windows_alphamelts_1-8.zip"
        install_file =  'alphamelts_win{}.exe'.format(bits)
    else:
        raise NotImplementedError('System unknown: {}'.format(system))

    # Set install directory for .bat files
    directory = Path(directory)
    if directory:
        install_dir = directory
    else:
        install_dir = '.'

    if not install_dir.exists():
        install_dir.mkdir(parents=True)

    r = requests.get(url, stream=True)
    if r.ok:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        extract_zip(z, install_dir)


def install_melts(install_dir,
                  link_dir=None,
                  eg_dir=None,
                  native=True,
                  temp_dir=Path("~").expanduser()/'temp'/'temp_melts',
                  keep_tempdir=False):
    """
    Parameters
    ----------
    install_dir : str | pathlib.Path
        Directory into which to install melts executable.
    link_dir : str | pathlib.Path, None
        Directory into which to deposit melts links.
    eg_dir : str | pathlib.Path
        Directory into which to deposit melts examples.
    native : bool, True
        Whether to install using perl scripts (windows).
    temp_dir : str | pathlib.Path, $USER$/temp/temp_melts
        Temporary directory for melts file download and install.
    keep_tempdir : bool, False
        Whether to cache tempoary files and preserve the temporary directory.
    """
    system = platform.system()
    release = platform.release()
    version = platform.version()
    bits, linkage = platform.architecture()
    bits = bits[:2]

    temp_dir = Path(temp_dir)

    if (temp_dir / 'install.command').exists():
        pass
    else:
        print('Downloading Melts')
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True)
        download_melts(temp_dir)

    install_dir = Path(install_dir)

    if not install_dir.exists():
        install_dir.mkdir(parents=True)

    if link_dir is not None:
        link_dir = Path(link_dir)
    else:
        link_dir = install_dir / 'links'

    if not link_dir.exists():
        link_dir.mkdir(parents=True)

    if eg_dir is not None:
        eg_dir = Path(eg_dir)
    else:
        eg_dir = install_dir / 'examples'

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
            install_source = os.path.join(temp_dir, 'install.command')
            args = ["perl", install_source]

            # [C:\Users\<>\Documents\bin]
            # [./\examples]
            # use default settings file
            # continue
            # return to finish
            inputs = ['', str(link_dir), str(eg_dir), '', 'y', '', '',]
            p = subprocess.run(args,
                               input=('\n'.join(inputs)).encode('UTF-8'),
                               stdout=subprocess.PIPE)

            for line in p.stdout.decode('UTF-8').split('\r\n'):
                print(line)
            assert p.returncode == 0

            # copy files from tempdir to install_dir
            regs = ['command', 'command_auto_file', 'path', 'perl']
            comms = ['column_pick', 'file_format', 'run_alphamelts']
            for (prefixes, ext) in [(regs, '.reg'),
                                    (comms, '.command')]:
                for prefix in prefixes:
                    temp_regpath = (temp_dir / prefix).with_suffix(ext)
                    install_regpath = install_dir / temp_regpath.name
                    shutil.copy(str(temp_regpath), str(install_regpath))
        elif native:

            # need to split into platforms
            egs = []
            for g in ['*.melts', '*.txt', '*.m ']:
                egs += list(temp_dir.glob(g))
            comms = ['column_pick', 'file_format', 'run_alphamelts']
            comms = [(temp_dir / i).with_suffix('.command') for i in comms]

            files_to_copy = []
            if system == 'Windows':
                alphafile =  temp_dir / 'alphamelts_win{}.exe'.format(bits)
                bats = comms + [temp_dir / 'alphamelts']
                bats = [i.with_suffix('.bat')  for i in bats]
                batdata = {}

                for cf in comms:
                    batdata[cf.stem] = '''@echo off\n"{}" %*'''.format(
                                            install_dir / cf.name)
                batdata['alphamelts'] = '''@echo off\n"{}"'''.format(
                                            install_dir / alphafile.name)
                for b in bats:
                    with open(b, 'w') as fout:
                        fout.write(batdata[b.stem]) # dummy bats

                files_to_copy +=  [(link_dir, bats)]

                #regs = ['command', 'command_auto_file', 'path', 'perl']

            elif system == 'Linux':
                alphafile =  temp_dir / 'alphamelts_linux{}'.format(bits)
            elif system == 'Darwin':
                alphafile =  temp_dir / 'alphamelts_macosx{}'.format(bits)


            files_to_copy += [(eg_dir, egs),
                              (install_dir, comms),
                              (install_dir, [alphafile])]
            for (target, files) in files_to_copy:
                for fn in files:
                    copy_file(temp_dir / fn.name, target / fn.name)
    except AssertionError:
        raise AssertionError
    finally:
        if not keep_tempdir:
            remove_tempdir(temp_dir)


def melts_query(data_dict, url_sfx='Compute'):
    """
    Execute query against the MELTS web services.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing data to be sent to the web query.
    url_sfx : str, Compute
        URL suffix to denote specific web service (Compute | Oxides | Phases).
    """
    url = 'http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/' + url_sfx
    xmldata = dicttoxml.dicttoxml(data_dict,
                                  custom_root='MELTSinput',
                                  root=True,
                                  attr_type=False)
    headers = {"content-type": "text/xml",
               "data-type": "xml"}
    resp = requests.post(url, data=xmldata, headers=headers)
    resp.raise_for_status()
    result = xmljson.parker.data(ET.fromstring(resp.text))
    return result


def melts_compute(data_dict):
    """
    Execute 'Compute' query against the MELTS web services.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing data to be sent to the Compute web query.
    """
    url_sfx = "Compute"
    result = melts_query(data_dict, url_sfx=url_sfx)
    assert 'Success' in result['status']
    return result


def melts_oxides(data_dict):
    """
    Execute 'Oxides' query against the MELTS web services.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing data to be sent to the Oxides web query.
    """
    model = data_dict['initialize'].pop('modelSelection', 'MELTS_v1.0.x')
    data_dict = {'modelSelection': model}
    url_sfx = "Oxides"
    result = melts_query(data_dict, url_sfx=url_sfx)
    return result['Oxide']


def melts_phases(data_dict):
    """
    Execute 'Phases' query against the MELTS web services.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing data to be sent to the Phases web query.
    """
    model = data_dict['initialize'].pop('modelSelection', 'MELTS_v1.0.x')
    data_dict = {'modelSelection': model}
    url_sfx = "Phases"
    result = melts_query(data_dict, url_sfx=url_sfx)
    return result['Phase']
