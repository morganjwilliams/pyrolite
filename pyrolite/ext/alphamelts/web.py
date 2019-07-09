"""
Minimal REST interfaces to the MELTS web services.

Todo
------

    * Function to generate a valid parameter dictionary to pass to the service.
"""
import logging
import requests
import dicttoxml, xmljson
from xml.etree import ElementTree as ET
from ...util.web import internet_connection

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def melts_query(data_dict, url_sfx="Compute"):
    """
    Execute query against the MELTS web services.

    Parameters
    ----------
    data_dict : :class:`dict`
        Dictionary containing data to be sent to the web query.
    url_sfx : :class:`str`, :code:`Compute`
        URL suffix to denote specific web service (Compute | Oxides | Phases).

    Returns
    --------
    :class:`dict`
        Dictionary containing results.
    """
    try:
        assert internet_connection()
        url = "http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/" + url_sfx
        xmldata = dicttoxml.dicttoxml(
            data_dict, custom_root="MELTSinput", root=True, attr_type=False
        )
        headers = {"content-type": "text/xml", "data-type": "xml"}
        resp = requests.post(url, data=xmldata, headers=headers)
        resp.raise_for_status()
        result = xmljson.parker.data(ET.fromstring(resp.text))
        return result
    except AssertionError:
        raise AssertionError("Must be connected to the internet to run query.")


def melts_compute(data_dict):
    """
    Execute 'Compute' query against the MELTS web services.

    Parameters
    ----------
    data_dict : :class:`dict`
        Dictionary containing data to be sent to the Compute web query.

    Returns
    --------
    :class:`dict`
        Dictionary containing results.
    """
    url_sfx = "Compute"
    result = melts_query(data_dict, url_sfx=url_sfx)
    assert "Success" in result["status"]
    return result


def melts_oxides(data_dict):
    """
    Execute 'Oxides' query against the MELTS web services.

    Parameters
    ----------
    data_dict : :class:`dict`
        Dictionary containing data to be sent to the Oxides web query.

    Returns
    --------
    :class:`dict`
        Dictionary containing results.
    """
    model = data_dict["initialize"].pop("modelSelection", "MELTS_v1.0.x")
    data_dict = {"modelSelection": model}
    url_sfx = "Oxides"
    result = melts_query(data_dict, url_sfx=url_sfx)
    return result["Oxide"]


def melts_phases(data_dict):
    """
    Execute 'Phases' query against the MELTS web services.

    Parameters
    ----------
    data_dict : :class:`dict`
        Dictionary containing data to be sent to the Phases web query.

    Returns
    --------
    :class:`dict`
        Dictionary containing results.
    """
    model = data_dict["initialize"].pop("modelSelection", "MELTS_v1.0.x")
    data_dict = {"modelSelection": model}
    url_sfx = "Phases"
    result = melts_query(data_dict, url_sfx=url_sfx)
    return result["Phase"]
