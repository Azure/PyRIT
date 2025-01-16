# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from dataclasses import dataclass

from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration


@dataclass
class NormalizerRequest(abc.ABC):
    seed_prompt_group: SeedPromptGroup
    request_converter_configurations: list[PromptConverterConfiguration]
    response_converter_configurations: list[PromptConverterConfiguration]
    conversation_id: str


    def __init__(
            self,
            *,
            seed_prompt_group: SeedPromptGroup,
            request_converter_configurations: list[PromptConverterConfiguration] = [],
            response_converter_configurations: list[PromptConverterConfiguration] = [],
            conversation_id: str = None,
    ):
        self.seed_prompt_group = seed_prompt_group
        self.request_converter_configurations = request_converter_configurations
        self.response_converter_configurations = response_converter_configurations
        self.conversation_id = conversation_id
