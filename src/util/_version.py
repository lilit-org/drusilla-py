"""Version information for the agents package."""

import importlib.metadata
import sys

try:
    __version__ = importlib.metadata.version("agents") if sys.meta_path else "0.0.0"
except (importlib.metadata.PackageNotFoundError, ImportError):
    __version__ = "0.0.0"
