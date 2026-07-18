"""Personal RAG system package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("personal-rag-system")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
