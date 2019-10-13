import re
import os, sys
import shlex
import struct
import getpass
import subprocess
from pathlib import Path
from subprocess import Popen, PIPE
from contextlib import contextmanager
from tinydb import TinyDB, Query

__backend__ = None
try:
    import psycopg2
    from psycopg2 import ProgrammingError as PGProgrammingError
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    __backend__ = psycopg2
except:
    pass

try:
    import sqlite3
    from sqlite3 import Error as SQLiteError
    from sqlite3 import OperationalError as SQLOperationalError
except:
    pass

try:
    import pyodbc
    from pyodbc import DatabaseError as PyODBCDatabaseError
    from pyodbc import ProgrammingError as PyODBCProgrammingError

    __backend__ = pyodbc
except:
    pass

import logging
from .text import *

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def _list_tindyb_unique_values(variable, dbpath=None):
    """
    List unique values from a column of a :mod:`TinyDB` json database.

    Parameters
    -----------
    variable : :class:`str`
        Name of the variable to check for unique values.
    dbpath : :class:`pathlib.Path` | :class:`str`
        Path to the relevant database.

    Returns
    ----------
    :class:`list`
    """

    with TinyDB(str(dbpath)) as db:
        out = list(set([a.get(variable, None) for a in db.all()]))
    return out


def check_access_driver():
    return [x for x in pyodbc.drivers() if x.startswith("Microsoft Access Driver")]


def bitness():
    bitness = struct.calcsize("P") * 8
    return "{} bit".format(bitness)


@contextmanager
def open_db_connection(
    connection_params,
    commit=False,
    encoding="utf-8",
    short_decoding="utf-8",
    wide_decoding="utf-16",
    backend=__backend__,
):
    """
    https://github.com/mkleehammer/pyodbc/wiki/Unicode

    Todo
    ------
        Implement pooled connections.
        http://initd.org/psycopg/docs/pool.html
    """
    if isinstance(connection_params, str):
        connection = backend.connect(connection_params)
    elif isinstance(connection_params, dict):
        connection = backend.connect(**connection_params)
    else:
        raise NotImplementedError

    if backend.__name__ == "pyodbc":
        connection.autocommit = False
        connection.setencoding(encoding)
        connection.setdecoding(pyodbc.SQL_CHAR, encoding=short_decoding)
        connection.setdecoding(pyodbc.SQL_WCHAR, encoding=wide_decoding)

    cursor = connection.cursor()

    def rollback(crsr):
        try:
            try:
                crsr.execute("ROLLBACK;")
            except SQLOperationalError:
                print("No transaction to rollback.")
        except PyODBCProgrammingError as err:
            print("ROLLBACK not supported.")

    try:
        yield connection, cursor
    except PyODBCDatabaseError as err:
        error, = err.args
        sys.stderr.write(error.message)
        rollback(cursor)
        raise err
    else:
        if commit:
            try:
                cursor.execute("COMMIT;")
            except SQLOperationalError:
                print("No active transaction to commit.")
        else:
            rollback(cursor)
    finally:
        connection.close()
