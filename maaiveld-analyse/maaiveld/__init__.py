import logging


LOGFILE = "maaiveld.log"
LOGFORMAT = '%(asctime)s - %(processName)-14s - %(module)-14s - %(levelname)s - %(message)s'


def configure_logging():
    """Configure logging for both main and worker processes."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove inherited handlers (important for subprocesses)
    logger.handlers.clear()

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(LOGFORMAT))

    fh = logging.FileHandler(LOGFILE, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOGFORMAT))

    logger.addHandler(sh)
    logger.addHandler(fh)
