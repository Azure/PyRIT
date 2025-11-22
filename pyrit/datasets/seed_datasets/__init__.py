# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dataset loaders with automatic discovery and unified interface.

This module provides:
- DatasetLoader: Abstract base class for all dataset loaders
- Automatic registration of all dataset loader implementations (local and remote)
- Remote dataset loaders in the 'remote' submodule
- Local dataset loaders in the 'local' submodule

Example usage:
    >>> from pyrit.datasets.dataset_loaders import DatasetLoader
    >>> 
    >>> # Fetch all datasets (local and remote)
    >>> all_datasets = await DatasetLoader.fetch_all_datasets()
    >>> 
    >>> # Fetch specific datasets
    >>> specific = await DatasetLoader.fetch_all_datasets(
    ...     dataset_names=["harmbench", "DarkBench"]
    ... )
"""

# Import base class
from pyrit.datasets.seed_datasets.dataset_loader import DatasetLoader

# Import remote loaders to trigger registration
from pyrit.datasets.seed_datasets import remote  # noqa: F401

# Import local loaders to trigger registration
from pyrit.datasets.seed_datasets import local  # noqa: F401

__all__ = [
    "DatasetLoader",
]
