import os, sys
from pathlib import Path
import re
import struct
import subprocess
from subprocess import Popen, PIPE
import shlex
from contextlib import contextmanager
import getpass
import pyodbc, psycopg2, sqlite3
from psycopg2 import ProgrammingError as PGProgrammingError
from sqlite3 import Error as SQLiteError
from sqlite3 import OperationalError as SQLOperationalError
from pyodbc import DatabaseError as PyODBCDatabaseError
from pyodbc import ProgrammingError as PyODBCProgrammingError
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from .text import *

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()


def check_access_driver():
    return [x for x in pyodbc.drivers()
            if x.startswith('Microsoft Access Driver')]

def bitness():
    bitness = struct.calcsize("P") * 8
    return '{} bit'.format(bitness)

@contextmanager
def open_db_connection(connection_params,
                       commit=False,
                       encoding='utf-8',
                       short_decoding='utf-8',
                       wide_decoding='utf-16',
                       backend=pyodbc):
    """
    Todo: Implement pooled connections.
    http://initd.org/psycopg/docs/pool.html

    https://github.com/mkleehammer/pyodbc/wiki/Unicode
    """
    if isinstance(connection_params, str):
        connection = backend.connect(connection_params)
    elif isinstance(connection_params, dict):
        connection = backend.connect(**connection_params)
    else:
        raise NotImplementedError

    if backend.__name__ == 'pyodbc':
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
                print('No active transaction to commit.')
        else:
            rollback(cursor)
    finally:
        connection.close()
