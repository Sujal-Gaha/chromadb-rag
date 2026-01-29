import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# SIMPLIFIED FORMAT - remove colorlog dependency for now
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create handlers
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True,  # Override any existing config
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
