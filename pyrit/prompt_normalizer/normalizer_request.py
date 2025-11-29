# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

from pyrit.models import SeedGroup
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)


@dataclass
class NormalizerRequest:
    """
    Represents a single request sent to normalizer.
    """

    seed_group: SeedGroup
    request_converter_configurations: list[PromptConverterConfiguration]
    response_converter_configurations: list[PromptConverterConfiguration]
    conversation_id: str | None

    def __init__(
        self,
        *,
        seed_group: SeedGroup,
        request_converter_configurations: list[PromptConverterConfiguration] = [],
        response_converter_configurations: list[PromptConverterConfiguration] = [],
        conversation_id: Optional[str] = None,
    ):
        """
        Initialize a normalizer request.

        Args:
            seed_group (SeedGroup): The group of seed prompts to be normalized.
            request_converter_configurations (list[PromptConverterConfiguration]): Configurations for converting
                the request. Defaults to an empty list.
            response_converter_configurations (list[PromptConverterConfiguration]): Configurations for converting
                the response. Defaults to an empty list.
            conversation_id (Optional[str]): The ID of the conversation. Defaults to None.
        """
        self.seed_group = seed_group
        self.request_converter_configurations = request_converter_configurations
        self.response_converter_configurations = response_converter_configurations
        self.conversation_id = conversation_id

    def validate(self) -> None:
        """
        Validate the normalizer request.

        Raises:
            ValueError: If the seed group is empty or prompts have different sequences.
        """
        if not self.seed_group or len(self.seed_group.prompts) < 1:
            raise ValueError("Seed group must be provided.")

        if not self.seed_group.is_single_request():
            raise ValueError("Sequence must be equal for every piece of a single normalizer request.")
