"""Central logging configuration for the project."""
import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d')}.log")

_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    return logging.getLogger(name)
