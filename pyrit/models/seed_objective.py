# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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

    @classmethod
    def from_yaml_with_required_parameters(
        cls,
        template_path: Union[str, Path],
        required_parameters: list[str],
        error_message: Optional[str] = None,
    ) -> "SeedObjective":
        """
        Load a Seed from a YAML file and validate that it contains specific parameters.

        Args:
            template_path: Path to the YAML file containing the template.
            required_parameters: List of parameter names that must exist in the template.
            error_message: Custom error message if validation fails. If None, a default message is used.

        Returns:
            SeedObjective: The loaded and validated seed of the specific subclass type.
        """
        return cls.from_yaml_file(template_path)
