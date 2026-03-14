# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypedDict

"""
Contains metadata objects for datasets (i.e. subclasses of SeedDatasetProvider).

SeedDatasetMetadata is the internal schema used to normalize metadata fields
from different sources:
- Remote providers that declare metadata as class attributes
- Local prompt files that store metadata at the top level

SeedDatasetFilter is the user-facing filter schema consumed by
SeedDatasetProvider.get_all_dataset_names().
"""


class SeedDatasetSize(Enum):
    """Ordinal size (by bucket) of the dataset."""

    TINY = "tiny"  # < 10
    SMALL = "small"  # >= 10, < 100
    MEDIUM = "medium"  # >= 100, < 500
    LARGE = "large"  # >= 500, < 5000
    HUGE = "huge"  # >= 5000


class SeedDatasetLoadingRank(Enum):
    """
    Represents the general difficulty of loading in a dataset.
    """

    # Default is equivalent to "fastest" in the sense that datasets marked
    # with a default rank will always get loaded.
    DEFAULT = "default"

    # These represent actual ranks.
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"

    # Unknown corresponds to an untested dataset that won't be loaded. It is the
    # default provided in SeedDatasetProvider.
    UNKNOWN = "unknown"


class SeedDatasetModality(Enum):
    """
    Type of data contained in the dataset.
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class SeedDatasetSourceType(Enum):
    """
    Where the dataset is pulled from.
    """

    REMOTE = "remote"
    LOCAL = "local"


@dataclass
class SeedDatasetFilter:
    """
    Filter object for datasets. Passed to `get_all_dataset_names` in
    SeedDatasetProvider.
    """

    tags: Optional[set[str]] = None
    sizes: Optional[list[SeedDatasetSize]] = None
    modalities: Optional[list[SeedDatasetModality]] = None
    source_types: Optional[list[SeedDatasetSourceType]] = None
    ranks: Optional[list[SeedDatasetLoadingRank]] = None
    harm_categories: Optional[list[str]] = None


@dataclass(frozen=True)
class SeedDatasetMetadata:
    """
    Metadata object for datasets. Holds the same fields as the filter
    object.
    """

    tags: Optional[set[str]] = None
    size: Optional[SeedDatasetSize] = None
    modalities: Optional[list[SeedDatasetModality]] = None
    source_type: Optional[SeedDatasetSourceType] = None
    rank: SeedDatasetLoadingRank = SeedDatasetLoadingRank.UNKNOWN
    harm_categories: Optional[list[str]] = None


class SeedDatasetMetadataUtilities:
    """
    Utilities for deriving metadata for datasets. Currently, only static attributes
    are supported.

    The default working location for datasets is the in-memory database.
    """

    class Metrics(TypedDict):
        """
        Typed dictionary for easier retrieval and calculation of dataset metrics.
        """

        exact_size: int
        loading_time_ms: float
        modalities_found: set[str]
        source_type: str
        harm_categories_found: set[str]
        tags: set[str]

    # Stores working dataset calculations.
    # Maps name to metrics, which are later converted into SeedDatasetMetadata.
    _cache: dict[str, Metrics] = {}

    @classmethod
    def populate_datasets(cls) -> None:
        """
        Populate metadata for all registered datasets.

        WARNING: Because metadata is stored as class attributes, this method can directly
        change source files. Be extra careful when running it.
        """
        # Get all dataset names
        # Calling SeedDatasetProvider would create a circular import, so we do this explicitly
        datasets: list[str] = []

        # Populate cache with empty (name, metrics) pairs
        for dataset in datasets:
            metrics: SeedDatasetMetadataUtilities.Metrics = {
                "exact_size": -1,
                "loading_time_ms": -1.0,
                "modalities_found": {"None"},
                "source_type": "None",
                "harm_categories_found": {"None"},
                "tags": {"None"},
            }
            cls._cache[dataset] = metrics

            # Using a list, for each dataset name, load it in depending on class type
            # Invoke the appropriate helper to parse it

            # If local, local_helper

            # If remote, remote_helper

            # Get contents from the memory database
            # Note that we have to load it into the memory_database to get timing
            # We also want the helper to do no initialization, just extract the relevant
            # types and get ready to call a timing library

            # Calculate metrics one by one

        # Once out of the loop, calculate metadata fields

        # Loading rank by comparing relative speeds

        # Size by comparing buckets

        # Convert all others to types

        # Update (if update = True) the datasets

        # If remote, write to the file using regex
        # E.g. harm_categories: ... should appear in source

        # If local, make sure the .prompt is formatted nicely

    @classmethod
    def _local_helper(cls) -> None:
        """
        Load local datasets into the working cache.
        """

    @classmethod
    def _remote_helper(cls) -> None:
        """
        Load remote datasets into the working cache.
        """

    @classmethod
    def _remote_writer(cls) -> None:
        """
        Write updated metadata to a remote dataset source file.
        """

    @classmethod
    def _local_writer(cls) -> None:
        """
        Write updated metadata to a local .prompt file.
        """
