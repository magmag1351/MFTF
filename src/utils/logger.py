import logging

def setup_logger():
    logger = logging.getLogger("FatigueMonitor")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
