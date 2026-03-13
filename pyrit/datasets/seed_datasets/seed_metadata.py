# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pyrit.common.path import DATASETS_PATH

"""
Contains metadata objects for datasets (i.e. subclasses of SeedDatasetProvider).

The ground truth is SeedDatasetMetadata. This is 
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

    tags: Optional[set[str]]
    sizes: Optional[list[SeedDatasetSize]]
    modalities: Optional[list[SeedDatasetModality]]
    sources: Optional[list[SeedDatasetSourceType]]
    ranks: Optional[list[SeedDatasetLoadingRank]]
    harm_categories: Optional[list[str]]


@dataclass(frozen=True)
class SeedDatasetMetadata:
    """
    Metadata object for datasets. Holds the same fields as the filter
    object.
    """

    tags: Optional[set[str]]
    size: Optional[SeedDatasetSize]
    modalities: Optional[list[SeedDatasetModality]]
    source: Optional[SeedDatasetSourceType]
    rank: Optional[SeedDatasetLoadingRank]
    harm_categories: Optional[list[str]]


class SeedDatasetMetadataUtilities:
    """
    Collected utilities for managing and updating metadata.
    """

    @staticmethod
    def populate_metadata() -> None:
        """
        WARNING: Because this function updates the metadata for each SeedDatasetProvider,
        it changes the provider's corresopnding source file. Run with caution!

        Updates the metadata per SeedDatasetProvider.
        """

        # 1 Gather all dataset files

        # 2 For each file, download and store in the database (in-memory)

        # 3 Count the number of entries exactly and identify its threshold

        # 4 If harm categories are found in source, add them

        # 5 Inspect type of prompts to identify modalities present

        # 6 Inspect source file to find where it pulled from

        # 7 Leave rank optional for now
