# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
import inspect
import logging
from typing import Any, Dict, List, Optional, Type

from pyrit.models.seed_dataset import SeedDataset

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """
    Abstract base class for loading datasets with automatic registration.

    All concrete subclasses are automatically registered and can be discovered
    via get_all_loaders() class method. This enables automatic discovery of
    both local and remote dataset loaders.

    Subclasses must implement:
    - fetch_dataset(): Fetch and return the dataset as a SeedDataset
    - dataset_name property: Human-readable name for the dataset
    """

    _registry: Dict[str, Type["DatasetLoader"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically register non-abstract subclasses.

        This is called when a class inherits from DatasetLoader.
        """
        super().__init_subclass__(**kwargs)
        # Only register concrete (non-abstract) classes
        if not inspect.isabstract(cls):
            DatasetLoader._registry[cls.__name__] = cls
            logger.debug(f"Registered dataset loader: {cls.__name__}")

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Return the human-readable name of the dataset.

        Returns:
            str: The dataset name (e.g., "HarmBench", "JailbreakBench JBB-Behaviors")
        """
        pass

    @abstractmethod
    async def fetch_dataset(self) -> SeedDataset:
        """
        Fetch the dataset and return as a SeedDataset.

        Returns:
            SeedDataset: The fetched dataset with prompts.

        Raises:
            Exception: If the dataset cannot be fetched or processed.
        """
        pass

    @classmethod
    def get_all_loaders(cls) -> Dict[str, Type["DatasetLoader"]]:
        """
        Get all registered dataset loader classes.

        Returns:
            Dict[str, Type[DatasetLoader]]: Dictionary mapping class names to loader classes.
        """
        return cls._registry.copy()

    @classmethod
    def get_all_dataset_names(cls) -> List[str]:
        """
        Get the names of all registered datasets.

        Returns:
            List[str]: List of dataset names from all registered loaders.

        Example:
            >>> names = DatasetLoader.get_all_dataset_names()
            >>> print(f"Available datasets: {', '.join(names)}")
        """
        dataset_names = []
        for loader_class in cls._registry.values():
            try:
                # Instantiate to get dataset name
                loader = loader_class()
                dataset_names.append(loader.dataset_name)
            except Exception as e:
                logger.warning(f"Could not get dataset name from {loader_class.__name__}: {e}")
        return sorted(dataset_names)

    @classmethod
    async def fetch_all_datasets(
        cls,
        *,
        dataset_names: Optional[List[str]] = None,
    ) -> List[SeedDataset]:
        """
        Fetch all registered datasets with optional filtering.

        Args:
            dataset_names: Optional list of dataset names to fetch. If None, fetches all.
                          Names should match the dataset_name property of loaders.

        Returns:
            List[SeedDataset]: List of all fetched datasets.

        Example:
            >>> # Fetch all datasets (local and remote)
            >>> all_datasets = await DatasetLoader.fetch_all_datasets()
            >>> 
            >>> # Fetch specific datasets
            >>> specific = await DatasetLoader.fetch_all_datasets(
            ...     dataset_names=["harmbench", "DarkBench"]
            ... )
        """
        datasets = []

        for loader_name, loader_class in cls._registry.items():
            try:
                # Instantiate to check dataset name for filtering
                loader = loader_class()

                # Apply dataset name filter if specified
                if dataset_names is not None:
                    if loader.dataset_name not in dataset_names:
                        logger.debug(f"Skipping {loader_name} - not in filter list")
                        continue

                logger.info(f"Fetching dataset: {loader_name}")
                dataset = await loader.fetch_dataset()
                datasets.append(dataset)
            except Exception as e:
                logger.error(f"Failed to fetch dataset {loader_name}: {e}")

        logger.info(f"Successfully fetched {len(datasets)} out of {len(cls._registry)} datasets")
        return datasets
