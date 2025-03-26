import logging
import sys

logger = logging.getLogger("deepseek.agents")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def enable_verbose_stdout_logging():
    """Enable verbose logging to stdout for the deepseek.agents logger."""
    pass
