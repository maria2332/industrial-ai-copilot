"""Industrial AI Copilot: RAG over technical documentation plus predictive maintenance."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("industrial-ai-copilot")
except PackageNotFoundError:  # running from source without `pip install -e .`
    __version__ = "0.0.0"
