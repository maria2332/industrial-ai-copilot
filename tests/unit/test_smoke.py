"""Smoke tests: the package and its planned subpackages are importable and documented."""

import importlib

import pytest

import industrial_ai

SUBPACKAGES = [
    "config",
    "llm",
    "ingestion",
    "rag",
    "ml",
    "agents",
    "equipment",
    "api",
    "evaluation",
    "utils",
]


def test_package_exposes_version() -> None:
    assert isinstance(industrial_ai.__version__, str)
    assert industrial_ai.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable_and_documented(name: str) -> None:
    module = importlib.import_module(f"industrial_ai.{name}")
    assert module.__doc__, f"industrial_ai.{name} must document its responsibility"
