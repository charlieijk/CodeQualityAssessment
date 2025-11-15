"""
Data collection and preprocessing utilities for model training.
"""

from importlib import import_module
from typing import Any

__all__ = ["DatasetBuilder", "DatasetEntry", "DatasetBuilderConfig"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("src.data_pipeline.dataset_builder")
        return getattr(module, name)
    raise AttributeError(f"module 'src.data_pipeline' has no attribute {name!r}")
