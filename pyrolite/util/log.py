import io
import logging

# http://docs.python-guide.org/en/latest/writing/logging/


def Handle(
    logger: logging.Logger | str,
    handler_class: str = logging.StreamHandler,
    formatter: str = "%(asctime)s %(name)s - %(levelname)s: %(message)s",
    level: str | int | None = None,
) -> logging.Logger:
    """
    Handle a logger with a standardised formatting.

    Parameters
    -----------
    logger : logging.Logger | str
        Logger or module name to source a logger from.
    handler_class : `logging.Handler`
        Handler class for the logging messages.
    formatter : str | `logging.Formatter`
        Formatter for the logging handler. Strings will be passed to
        the `logging.Formatter` constructor.
    level : str
        Logging level for the handler.

    Returns
    ----------
    logging.Logger
        Configured logger.
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    elif isinstance(logger, logging.Logger):
        pass
    else:
        raise NotImplementedError
    logger.propagate = False
    if isinstance(formatter, str):
        formatter = logging.Formatter(formatter)

    active_handlers = [
        i
        for i in logger.handlers
        if isinstance(i, (handler_class))  # not a null handler
    ]
    if active_handlers:
        handler = active_handlers[0]  # use the existing stream handler
    else:
        handler = handler_class()
    handler.setFormatter(formatter)
    if handler not in active_handlers:
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
        super().__init__()
        self.logger = logger
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)
