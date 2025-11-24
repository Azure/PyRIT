# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
import inspect
import logging
from typing import Any, Dict, List, Optional, Type

from pyrit.models.seed_dataset import SeedDataset

logger = logging.getLogger(__name__)


class SeedDatasetProvider(ABC):
    """
    Abstract base class for providing seed datasets with automatic registration.

    All concrete subclasses are automatically registered and can be discovered
    via get_all_providers() class method. This enables automatic discovery of
    both local and remote dataset providers.

    Subclasses must implement:
    - fetch_dataset(): Fetch and return the dataset as a SeedDataset
    - dataset_name property: Human-readable name for the dataset
    """

    _registry: Dict[str, Type["SeedDatasetProvider"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically register non-abstract subclasses.

        This is called when a class inherits from SeedDatasetProvider.
        """
        super().__init_subclass__(**kwargs)
        # Only register concrete (non-abstract) classes
        if not inspect.isabstract(cls) and getattr(cls, "should_register", True):
            SeedDatasetProvider._registry[cls.__name__] = cls
            logger.debug(f"Registered dataset provider: {cls.__name__}")

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
    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch the dataset and return as a SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.
                   Remote datasets will use DB_DATA_PATH for caching.

        Returns:
            SeedDataset: The fetched dataset with prompts.

        Raises:
            Exception: If the dataset cannot be fetched or processed.
        """
        pass

    @classmethod
    def get_all_providers(cls) -> Dict[str, Type["SeedDatasetProvider"]]:
        """
        Get all registered dataset provider classes.

        Returns:
            Dict[str, Type[SeedDatasetProvider]]: Dictionary mapping class names to provider classes.
        """
        return cls._registry.copy()

    @classmethod
    def get_all_dataset_names(cls) -> List[str]:
        """
        Get the names of all registered datasets.

        Returns:
            List[str]: List of dataset names from all registered providers.

        Example:
            >>> names = SeedDatasetProvider.get_all_dataset_names()
            >>> print(f"Available datasets: {', '.join(names)}")
        """
        dataset_names = set()
        for provider_class in cls._registry.values():
            try:
                # Instantiate to get dataset name
                provider = provider_class()
                dataset_names.add(provider.dataset_name)
            except Exception as e:
                logger.warning(f"Could not get dataset name from {provider_class.__name__}: {e}")
        return sorted(list(dataset_names))

    @classmethod
    async def fetch_all_datasets(
        cls,
        *,
        dataset_names: Optional[List[str]] = None,
        cache: bool = True,
    ) -> List[SeedDataset]:
        """
        Fetch all registered datasets with optional filtering and caching.

        Args:
            dataset_names: Optional list of dataset names to fetch. If None, fetches all.
                          Names should match the dataset_name property of providers.
            cache: Whether to cache the fetched datasets. Defaults to True.
                   This uses DB_DATA_PATH for caching remote datasets.

        Returns:
            List[SeedDataset]: List of all fetched datasets.

        Example:
            >>> # Fetch all datasets (local and remote)
            >>> all_datasets = await SeedDatasetProvider.fetch_all_datasets()
            >>> 
            >>> # Fetch specific datasets without caching
            >>> specific = await SeedDatasetProvider.fetch_all_datasets(
            ...     dataset_names=["harmbench", "DarkBench"],
            ...     cache=False
            ... )
        """
        datasets = {}

        for provider_name, provider_class in cls._registry.items():
            try:
                # Instantiate the provider
                provider = provider_class()

                # Apply dataset name filter if specified
                if dataset_names is not None:
                    if provider.dataset_name not in dataset_names:
                        logger.debug(f"Skipping {provider_name} - not in filter list")
                        continue

                logger.info(f"Fetching dataset: {provider_name}")
                dataset = await provider.fetch_dataset(cache=cache)
                
                if provider.dataset_name in datasets:
                    # Merge with existing dataset
                    existing_dataset = datasets[provider.dataset_name]
                    existing_dataset.seeds.extend(dataset.seeds)
                else:
                    datasets[provider.dataset_name] = dataset
                    
            except Exception as e:
                logger.error(f"Failed to fetch dataset {provider_name}: {e}")

        logger.info(f"Successfully fetched {len(datasets)} unique datasets from {len(cls._registry)} providers")
        return list(datasets.values())
