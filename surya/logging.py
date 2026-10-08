import logging
import warnings
from surya.settings import settings

_logger = logging.getLogger("surya")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(settings.LOGLEVEL)
# Surya owns its output, including in library mode.
_logger.propagate = False


def configure_logging():
    get_logger().setLevel(settings.LOGLEVEL)
    warnings.simplefilter(action="ignore", category=FutureWarning)


def get_logger() -> logging.Logger:
    return _logger
