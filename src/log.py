import logging
from datetime import datetime


def setup_custom_logger(name,logfile_loc,mode):
    '''
    Starts a logger that prints to a file and to the screen.
    :param name:
    :param logfile_loc:
    :return:
    '''
    # formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    sh = logging.StreamHandler()
    # sh.setFormatter(formatter)

    logger = logging.getLogger(name)
    if mode == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.addHandler(sh)

    fh = logging.FileHandler(logfile_loc)
    # fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('Starting logfile at {}.'.format(datetime.now()))
    return logger


def close_logger(log):
    '''
    Removes all handlers from logger and effectively closes the logger down.
    :param log:
    :return:
    '''
    log.info('Closing logfile at {}.'.format(datetime.now()))
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)