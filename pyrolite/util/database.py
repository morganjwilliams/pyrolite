import struct
import sys
from contextlib import contextmanager

from .log import Handle

logger = Handle(__name__)


__backend__ = None
try:
    import psycopg2
    from pysycopg2.errors import DatabaseError, OperationalError, ProgrammingError

    __backend__ = psycopg2
except:
    pass

try:
    import pyodbc
    from pyodbc import DatabaseError, OperationalError, ProgrammingError

    __backend__ = pyodbc
except:
    pass


def check_access_driver():
    return [x for x in pyodbc.drivers() if x.startswith("Microsoft Access Driver")]


def bitness():
    bitness = struct.calcsize("P") * 8
    return f"{bitness} bit"


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
    ----
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
            except OperationalError:
                logger.info("No transaction to rollback.")
        except ProgrammingError:
            logger.error("ROLLBACK not supported.")

    try:
        yield connection, cursor
    except DatabaseError as err:
        (error,) = err.args
        sys.stderr.write(error.message)
        rollback(cursor)
        raise err
    else:
        if commit:
            try:
                cursor.execute("COMMIT;")
            except OperationalError:
                logger.info("No active transaction to commit.")
        else:
            rollback(cursor)
    finally:
        connection.close()
