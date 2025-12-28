# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dataset configuration for scenarios.

This module provides the DatasetConfiguration class that allows scenarios to be configured
with either explicit SeedGroups or dataset names (mutually exclusive).
"""

import random
from typing import List, Optional

from pyrit.memory import CentralMemory
from pyrit.models import SeedGroup


class DatasetConfiguration:
    """
    Configuration for scenario datasets.

    This class provides a unified way to specify the dataset source for scenarios.
    Only ONE of `seed_groups`, `dataset_name`, or `dataset_names` can be set.

    Args:
        seed_groups (Optional[List[SeedGroup]]): Explicit list of SeedGroups to use.
        dataset_name (Optional[str]): Name of a single dataset to load from memory.
        dataset_names (Optional[List[str]]): Names of multiple datasets to load from memory.
        max_dataset_size (Optional[int]): If set, randomly samples up to this many SeedGroups
            from the configured dataset source.
    """

    def __init__(
        self,
        *,
        seed_groups: Optional[List[SeedGroup]] = None,
        dataset_name: Optional[str] = None,
        dataset_names: Optional[List[str]] = None,
        max_dataset_size: Optional[int] = None,
    ) -> None:
        """
        Initialize a DatasetConfiguration.

        Args:
            seed_groups (Optional[List[SeedGroup]]): Explicit list of SeedGroups to use.
            dataset_name (Optional[str]): Name of a single dataset to load from memory.
            dataset_names (Optional[List[str]]): Names of multiple datasets to load from memory.
            max_dataset_size (Optional[int]): If set, randomly samples up to this many SeedGroups.

        Raises:
            ValueError: If more than one of seed_groups, dataset_name, or dataset_names is set.
            ValueError: If max_dataset_size is less than 1.
        """
        # Validate that only one data source is set
        sources_set = sum([
            seed_groups is not None,
            dataset_name is not None,
            dataset_names is not None,
        ])

        if sources_set > 1:
            raise ValueError(
                "Only one of 'seed_groups', 'dataset_name', or 'dataset_names' can be set. "
                "Use 'seed_groups' to provide explicit SeedGroups, 'dataset_name' for a single dataset, "
                "or 'dataset_names' for multiple datasets."
            )

        if max_dataset_size is not None and max_dataset_size < 1:
            raise ValueError("'max_dataset_size' must be a positive integer (>= 1).")

        # Store private attributes
        self._seed_groups = list(seed_groups) if seed_groups is not None else None
        self._max_dataset_size = max_dataset_size

        # Normalize dataset_name to dataset_names for internal consistency
        if dataset_name is not None:
            self._dataset_names: Optional[List[str]] = [dataset_name]
        else:
            self._dataset_names = list(dataset_names) if dataset_names is not None else None

    def get_seed_groups(self) -> List[SeedGroup]:
        """
        Resolve and return seed groups based on the configuration.

        This method handles all resolution logic:
        1. If seed_groups is set, use those directly
        2. If dataset_names is set, load from memory using those names
        3. If neither is set, raises ValueError

        In all cases, max_dataset_size is applied if set.

        Returns:
            List[SeedGroup]: The resolved seed groups, potentially sampled down
                to max_dataset_size.

        Raises:
            ValueError: If no seed groups could be resolved from the configuration.
        """
        seed_groups: List[SeedGroup] = []

        if self._seed_groups is not None:
            # Use explicit seed groups
            seed_groups = list(self._seed_groups)
        elif self._dataset_names is not None:
            # Load from specified dataset names
            memory = CentralMemory.get_memory_instance()
            for name in self._dataset_names:
                loaded = memory.get_seed_groups(dataset_name=name)
                if loaded:
                    seed_groups.extend(loaded)


        if not seed_groups:
            raise ValueError(
                "DatasetConfiguration has no seed_groups. "
                "Set seed_groups, dataset_name, or dataset_names."
            )

        # Apply max_dataset_size sampling
        return self._apply_max_dataset_size(seed_groups)

    def get_default_dataset_names(self) -> List[str]:
        """
        Get the list of default dataset names for this configuration.

        This is used by the CLI to display what datasets the scenario uses by default.

        Returns:
            List[str]: List of dataset names, or empty list if using explicit seed_groups.
        """
        if self._dataset_names is not None:
            return list(self._dataset_names)
        return []

    def _apply_max_dataset_size(self, seed_groups: List[SeedGroup]) -> List[SeedGroup]:
        """
        Apply max_dataset_size sampling to a list of seed groups.

        Args:
            seed_groups (List[SeedGroup]): The seed groups to potentially sample from.

        Returns:
            List[SeedGroup]: The original list if max_dataset_size is not set,
                or a random sample of up to max_dataset_size items.
        """
        if self._max_dataset_size is None or len(seed_groups) <= self._max_dataset_size:
            return seed_groups
        return random.sample(seed_groups, self._max_dataset_size)

    def has_data_source(self) -> bool:
        """
        Check if this configuration has a data source configured.

        Returns:
            bool: True if seed_groups or dataset_names is configured.
        """
        return self._seed_groups is not None or self._dataset_names is not None
