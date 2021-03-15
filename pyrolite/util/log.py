import logging
import io

# http://docs.python-guide.org/en/latest/writing/logging/


def Handle(
    logger,
    handler=logging.NullHandler(),
    formatter="%(asctime)s %(name)s - %(levelname)s: %(message)s",
    level=None,
):
    """
    Handle a logger with a standardised formatting.

    Parameters
    -----------
    logger : :class:`logging.Logger` | :class:`str`
        Logger or module name to source a logger from.
    handler : :class:`logging.Handler`
        Handler for the logging messages.
    formatter : :class:`str` | :class:`logging.Formatter`
        Formatter for the logging handler. Strings will be passed to
        the :class:`logging.Formatter` constructor.
    level : :class:`str`
        Logging level for the handler.

    Returns
    ----------
    :class:`logging.Logger`
        Configured logger.
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    elif isinstance(logger, logging.Logger):
        pass
    else:
        raise NotImplementedError
    if isinstance(formatter, str):
        formatter = logging.Formatter(formatter)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if level is not None:
        logger.setLevel(getattr(logging, level))
    return logger


class ToLogger(io.StringIO):
    """
    Output stream which will output to logger module instead of stdout.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(ToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


def stream_log(logger=None, level="INFO"):
    """
    Stream the log from a specific package or subpackage.

    Parameters
    ----------
    logger : :class:`str` | :class:`logging.Logger`
        Name of the logger or module to monitor logging from.
    level : :class:`str`, :code:`'INFO'`
        Logging level at which to set the handler output.

    Returns
    -------
    :class:`logging.Logger`
        Logger for the specified package with stream handler added.
    """
    # remove ipython active stream handler if present

    if logger is None:
        logger = logging.getLogger()  # root logger
        propagate = True
    else:
        propagate = False

    if isinstance(logger, str):
        logger = logging.getLogger(logger)  # module logger
    elif isinstance(logger, logging.Logger):
        pass  # enable passing a logger instance
    else:
        raise NotImplementedError

    logger.propagate = propagate  # don't duplicate by propagating to root
    int_level = getattr(logging, level)
    # check there are no handlers other than Null
    active_handlers = [
        i
        for i in logger.handlers
        if isinstance(i, (logging.StreamHandler))  # not a null handler
    ]

    if active_handlers:
        handler = active_handlers[0]  # use the existing one
    else:
        handler = logging.StreamHandler()  # make a new one

    if handler.level <= int_level:
        handler.setLevel(int_level)

    fmt = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if (logger.level == 0) or (logger.level > int_level):
        logger.setLevel(int_level)

    return logger
