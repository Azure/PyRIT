# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Local dataset loaders with automatic discovery.

Automatically discovers and registers all YAML dataset files from the seed_datasets directory.
"""

import logging
from pathlib import Path

from pyrit.datasets.seed_datasets.local.local_dataset_loader import LocalDatasetLoader

logger = logging.getLogger(__name__)

# Get the path to the seed_datasets directory
SEED_DATASETS_PATH = Path(__file__).parent / "seed_datasets"

# Auto-discover and register all YAML files
if SEED_DATASETS_PATH.exists():
    for yaml_file in SEED_DATASETS_PATH.glob("**/*.prompt"):
        try:
            # Create an instance which triggers registration via __init_subclass__
            LocalDatasetLoader(file_path=yaml_file)
            logger.debug(f"Registered local dataset: {yaml_file.name}")
        except Exception as e:
            logger.warning(f"Failed to register local dataset {yaml_file}: {e}")
else:
    logger.warning(f"Seed datasets directory not found: {SEED_DATASETS_PATH}")

__all__ = [
    "LocalDatasetLoader",
]
