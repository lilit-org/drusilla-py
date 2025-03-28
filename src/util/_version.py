import importlib.metadata

FALLBACK_VERSION = "0.0.1"

try:
    __version__ = importlib.metadata.version("agents")
except importlib.metadata.PackageNotFoundError:
    __version__ = FALLBACK_VERSION
