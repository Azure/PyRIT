# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

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
                raise ValueError(f"Could not get dataset name from {provider_class.__name__}: {e}")
        return sorted(list(dataset_names))

    @classmethod
    async def fetch_all_datasets(
        cls,
        *,
        dataset_names: Optional[List[str]] = None,
        cache: bool = True,
        raise_on_error: bool = False,
        max_concurrency: int = 10,
    ) -> List[SeedDataset]:
        """
        Fetch all registered datasets with optional filtering and caching.
        
        Datasets are fetched concurrently for improved performance. The default
        concurrency of 10 allows for significant speedup while avoiding overwhelming
        remote servers or network resources.

        Args:
            dataset_names: Optional list of dataset names to fetch. If None, fetches all.
                          Names should match the dataset_name property of providers.
            cache: Whether to cache the fetched datasets. Defaults to True.
                   This uses DB_DATA_PATH for caching remote datasets.
            raise_on_error: Whether to raise exceptions when a dataset fails to load.
                           If False (default), errors are logged and dataset is skipped.
                           If True, the first error will stop execution and be raised.
            max_concurrency: Maximum number of datasets to fetch concurrently. Defaults to 10.
                            Set to 1 for sequential execution. Higher values (e.g., 20) may
                            speed up initial downloads but could overwhelm network resources.

        Returns:
            List[SeedDataset]: List of all fetched datasets.

        Raises:
            Exception: If raise_on_error=True and any dataset fails to load.

        Example:
            >>> # Fetch all datasets with default concurrency (10)
            >>> all_datasets = await SeedDatasetProvider.fetch_all_datasets()
            >>> 
            >>> # Fetch specific datasets without caching, higher concurrency
            >>> specific = await SeedDatasetProvider.fetch_all_datasets(
            ...     dataset_names=["harmbench", "DarkBench"],
            ...     cache=False,
            ...     max_concurrency=20
            ... )
            >>> 
            >>> # Sequential execution
            >>> datasets = await SeedDatasetProvider.fetch_all_datasets(
            ...     max_concurrency=1
            ... )
        """
        
        async def fetch_single_dataset(
            provider_name: str, provider_class: Type["SeedDatasetProvider"]
        ) -> Optional[Tuple[str, SeedDataset]]:
            """Helper to fetch a single dataset with error handling."""
            try:
                provider = provider_class()
                
                # Apply dataset name filter if specified
                if dataset_names is not None:
                    if provider.dataset_name not in dataset_names:
                        logger.debug(f"Skipping {provider_name} - not in filter list")
                        return None
                
                logger.info(f"Fetching dataset: {provider_name}")
                dataset = await provider.fetch_dataset(cache=cache)
                return (provider.dataset_name, dataset)
                
            except Exception as e:
                if raise_on_error:
                    raise
                logger.error(f"Failed to fetch dataset {provider_name}: {e}")
                return None
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def fetch_with_semaphore(
            provider_name: str, provider_class: Type["SeedDatasetProvider"]
        ) -> Optional[Tuple[str, SeedDataset]]:
            """Wrapper to enforce concurrency limit."""
            async with semaphore:
                return await fetch_single_dataset(provider_name, provider_class)
        
        # Fetch all datasets with controlled concurrency
        tasks = [
            fetch_with_semaphore(provider_name, provider_class)
            for provider_name, provider_class in cls._registry.items()
        ]
        
        # When raise_on_error=False, gather returns exceptions in the results list
        # When raise_on_error=True, gather will raise immediately on first exception
        if raise_on_error:
            results = await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge datasets with the same name
        datasets: Dict[str, SeedDataset] = {}
        for result in results:
            # Skip None results (filtered datasets)
            if result is None:
                continue
            
            # Handle exceptions
            if isinstance(result, (Exception, BaseException)):
                if raise_on_error:
                    raise result
                # Error already logged in fetch_single_dataset
                continue
            
            # At this point, result must be a tuple
            dataset_name, dataset = result
            
            if dataset_name in datasets:
                # Merge with existing dataset by creating new list with combined seeds
                existing_dataset = datasets[dataset_name]
                combined_seeds = list(existing_dataset.seeds) + list(dataset.seeds)
                existing_dataset.seeds = combined_seeds
            else:
                datasets[dataset_name] = dataset

        logger.info(f"Successfully fetched {len(datasets)} unique datasets from {len(cls._registry)} providers")
        return list(datasets.values())
