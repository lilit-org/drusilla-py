import logging
import os
import sys

logger = logging.getLogger("deepseek.agents")
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "DEBUG").upper()))
logger.addHandler(logging.StreamHandler(sys.stdout))
