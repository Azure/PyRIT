# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Local dataset loaders with automatic discovery.

Automatically discovers and registers all YAML dataset files from the seed_datasets directory.
"""

from pyrit.datasets.seed_datasets.local.local_dataset_loader import LocalDatasetLoader

__all__ = [
    "LocalDatasetLoader",
]
