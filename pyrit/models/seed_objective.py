# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging

from dataclasses import dataclass

from pyrit.common.path import PATHS_DICT
from pyrit.models.seed import Seed

logger = logging.getLogger(__name__)

@dataclass
class SeedObjective(Seed):
    """Represents a seed objective with various attributes and metadata."""

    def __post_init__(self) -> None:
        """Post-initialization to render the template to replace existing values"""
        self.value = self.render_template_value_silent(**PATHS_DICT)
        self.data_type = "text"
    
    def set_encoding_metadata(self):
        """
        This method sets the encoding data for the prompt within metadata dictionary.
        """
        return  # No encoding metadata for text
