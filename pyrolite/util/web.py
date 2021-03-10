import requests

try:
    import httplib
except:
    import http.client as httplib

from .log import Handle

logger = Handle(__name__)


def urlify(url):
    """Strip a string to return a valid URL."""
    return url.strip().replace(" ", "_")


def internet_connection(target="pypi.org", secure=True):
    """
    Tests for an active internet connection, based on an optionally specified
    target.

    Parameters
    ----------
    target : :class:`str`
        URL to check connectivity, defaults to www.google.com

    Returns
    -------
    :class:`bool`
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


def download_file(url: str, encoding="UTF-8", postprocess=None):
    """
    Downloads a specific file from a url.

    Parameters
    ----------
    url : :class:`str`
        URL of specific file to download.
    encoding : :class:`str`
        String encoding.
    postprocess : :class:`callable`
        Callable function to post-process the requested content.
    """
    with requests.Session() as s:
        try:
            response = s.get(url)
            if response.status_code == requests.codes.ok:
                logger.debug("Response recieved from {}.".format(url))
                out = response.content

                if out is not None and encoding is not None:
                    out = response.content.decode(encoding)
                if postprocess is not None:
                    out = postprocess(out)
            else:
                msg = "Failed download - bad status code at {}".format(url)
                logger.warning(msg)
                response.raise_for_status()
                out = None
        except requests.exceptions.ConnectionError:
            logger.warning("Failed Connection to {}".format(url))
            out = None
    return out
