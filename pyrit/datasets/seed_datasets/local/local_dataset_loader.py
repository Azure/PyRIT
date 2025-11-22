# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path

from pyrit.datasets.seed_datasets.dataset_loader import DatasetLoader
from pyrit.models.seed_dataset import SeedDataset

logger = logging.getLogger(__name__)


class LocalDatasetLoader(DatasetLoader):
    """
    Loader for local YAML dataset files.

    This loader discovers and loads datasets from local YAML files.
    Each YAML file should be in the standard SeedDataset format.
    """

    def __init__(self, *, file_path: Path):
        """
        Initialize the local dataset loader.

        Args:
            file_path: Path to the YAML dataset file.
        """
        self.file_path = file_path
        
        # Pre-load to get dataset name
        try:
            dataset = SeedDataset.from_yaml_file(file_path)
            # Use the dataset_name from the YAML if available, otherwise use filename
            self._dataset_name = getattr(dataset, 'name', None) or file_path.stem
        except Exception as e:
            logger.warning(f"Could not pre-load dataset from {file_path}: {e}")
            self._dataset_name = file_path.stem

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    async def fetch_dataset(self) -> SeedDataset:
        """
        Load the dataset from the local YAML file.

        Returns:
            SeedDataset: The loaded dataset.

        Raises:
            Exception: If the dataset cannot be loaded.
        """
        try:
            logger.info(f"Loading local dataset from {self.file_path}")
            dataset = SeedDataset.from_yaml_file(self.file_path)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load local dataset from {self.file_path}: {e}")
            raise
