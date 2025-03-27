import logging
import os
import sys

logger = logging.getLogger("deepseek.agents")
log_level = os.getenv("LOG_LEVEL", "DEBUG")
logger.setLevel(getattr(logging, log_level.upper()))
logger.addHandler(logging.StreamHandler(sys.stdout))
