import logging
from datetime import datetime


def setup_custom_logger(logfile_loc: str, mode: str, name: str = 'runner') -> logging.Logger:
    """
    Starts a custom logger that logs all the results of the run.
    :param logfile_loc: Location of logfile on disk
    :param mode: logging mode
    :param name: name of logger
    :return:
    """
    sh = logging.StreamHandler()
    logger = logging.getLogger(name)
    if mode == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.addHandler(sh)

    fh = logging.FileHandler(logfile_loc)
    logger.addHandler(fh)
    logger.info('Starting logfile at {}.'.format(datetime.now()))
    return logger


def close_logger(log: logging.Logger) -> None:
    """
    Removes all handlers from logger and effectively closes the logger down.
    :param log: a handler to a logger
    :return:
    """
    log.info('Closing logfile at {}.'.format(datetime.now()))
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)
