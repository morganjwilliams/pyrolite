import http.client as httplib
from collections.abc import Callable

import requests
from requests.models import Response

from .log import Handle

logger = Handle(__name__)


def urlify(url: str):
    """Strip a string to return a valid URL."""
    return url.strip().replace(" ", "_")


def have_internet_connection(target: str = "pypi.org", secure: bool = True) -> bool:
    """
    Tests for an active internet connection, based on an optionally specified
    target.

    Parameters
    ----------
    target : str
        URL to check connectivity, defaults to www.google.com

    Returns
    -------
    bool
        Boolean indication of whether a HTTP connection can be established at the given
        url.
    """
    mode = [httplib.HTTPConnection, httplib.HTTPSConnection][secure]
    conn = mode(target, timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False


def download_file(
    url: str, encoding: str | None = "UTF-8", postprocess: Callable | None = None
) -> str | bytes | None:
    """
    Downloads a specific file from a url.

    Parameters
    ----------
    url : str
        URL of specific file to download.
    encoding : str
        String encoding.
    postprocess : Callable
        Callable function to post-process the requested content.
    """
    with requests.Session() as s:
        try:
            response: Response = s.get(url)
            if response.status_code == requests.codes.ok:
                logger.debug(f"Response recieved from {url}.")
                out: bytes | None = response.content

                if out is not None and encoding is not None:
                    out: str = response.content.decode(encoding)
                if postprocess is not None:
                    out: str = postprocess(out)
            else:
                msg = f"Failed download - bad status code at {url}"
                logger.warning(msg)
                response.raise_for_status()
                out = None
        except requests.exceptions.ConnectionError:
            logger.warning(f"Failed Connection to {url}")
            out = None
    return out
