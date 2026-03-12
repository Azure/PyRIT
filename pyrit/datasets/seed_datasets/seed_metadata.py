# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

"""
TODO Finish docstring

Contains metadata objects for datasets (i.e. subclasses of SeedDatasetProvider).

We have one DatasetMetadata dataclass that is our ground truth. As we instantiate datasets
using the subclass call in SeedDatasetProvider, we create DatasetMetadata and assign it to
a private variable there.

Some fields are dynamic (e.g. loading statistics, timestamp, dataset size) and are left as
NoneType until the SeedDatasetProvider actually downloads/parses the dataset and puts it in
CentralMemory.
"""


class SeedDatasetSize(Enum):
    """Ordinal size (by bucket) of the dataset."""

    TINY = "tiny"  # < 10
    SMALL = "small"  # >= 10, < 100
    MEDIUM = "medium"  # >= 100, < 500
    LARGE = "large"  # >= 500, < 5000
    HUGE = "huge"  # >= 5000


class SeedDatasetLoadingRank(Enum):
    """Represents the general difficulty of loading in a dataset."""

    DEFAULT = "default"
    EXTENDED = "extended"
    SLOW = "slow"


class SeedDatasetModality(Enum):
    """
    ...
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class SeedDatasetSourceType(Enum):
    """
    ...
    """

    GENERIC_URL = "generic_url"
    LOCAL = "local"
    HUGGING_FACE = "hugging_face"


@dataclass
class SeedDatasetFilter:
    """
    ...
    """

    tags: Optional[set[str]]
    sizes: Optional[list[SeedDatasetSize]]
    modalities: Optional[list[SeedDatasetModality]]
    sources: Optional[list[SeedDatasetSourceType]]
    ranks: Optional[list[SeedDatasetLoadingRank]]
    harm_categories: Optional[list[str]]


@dataclass(frozen=True)
class SeedDatasetMetadata:
    """
    ...
    """

    tags: Optional[set[str]]
    size: Optional[SeedDatasetSize]
    modalities: Optional[list[SeedDatasetModality]]
    source: Optional[SeedDatasetSourceType]
    rank: Optional[SeedDatasetLoadingRank]
    harm_categories: Optional[list[str]]


class SeedDatasetMetadataUtilities:
    """
    Collected utilities for managing and updating SeedDatasetMetadata.
    """

    @staticmethod
    def populate_metadata() -> None:
        """
        WARNING: Because this function updates the metadata for each SeedDatasetProvider,
        it changes the provider's corresopnding source file. Run with caution!

        Update the metadata per SeedDatasetProvider.
        """
