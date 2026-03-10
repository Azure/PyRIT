# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from dataclasses import dataclass


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
    size: int
    modalities: list[DatasetModalities]
    source: DatasetSourceType
    loading_rank: DatasetLoadingRank
