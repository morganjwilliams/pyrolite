import logging


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
