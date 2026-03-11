# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from dataclasses import dataclass

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


class DatasetLoadingRank(Enum):
    """Represents the general difficulty of loading in a dataset."""
    DEFAULT = "default"
    EXTENDED = "extended"
    SLOW = "slow"


class DatasetModalities(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class DatasetSourceType(Enum):
    GENERIC_URL = "generic_url"
    LOCAL = "local"
    HUGGING_FACE = "hugging_face"


@dataclass
class DatasetMetadata:
    # TODO: separate dynamic fields from static fields and mark dynamic fields as None
    size: int
    modalities: list[DatasetModalities]
    source: DatasetSourceType
    rank: DatasetLoadingRank


class DatasetFilters(Enum):
    # TODO: This is a bad way of extracting the fields from DatasetMetadata.
    # A metaclass or even just calling getattr might be better.
    SIZE = "size"
    MODALITIES = "modalities"
    SOURCE = "source"
    RANK = "rank"

# TODO These stubs should be moved somewhere, maybe as static methods to the metadata dataclass?


def _validate_filter_value(v):
    """Check if the filter value given is valid."""
