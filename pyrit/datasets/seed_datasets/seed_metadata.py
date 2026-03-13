# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

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

    DEFAULT = "default"
    EXTENDED = "extended"
    SLOW = "slow"


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
    sources: Optional[list[SeedDatasetSourceType]] = None
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
    source: Optional[SeedDatasetSourceType] = None
    rank: Optional[SeedDatasetLoadingRank] = None
    harm_categories: Optional[list[str]] = None
