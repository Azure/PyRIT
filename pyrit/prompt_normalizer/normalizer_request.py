# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from dataclasses import dataclass
from typing import Optional

from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration

# TODO delete these
class NormalizerRequestPiece(abc.ABC):

    def __init__(
        self,
        *,
        prompt_value: str,
        prompt_data_type: PromptDataType,
        request_converters: list[PromptConverter] = [],
        labels: Optional[dict[str, str]] = None,
        metadata: str = None,
    ) -> None:
        """
        Represents a piece of a normalizer request.

        It represents the minimum unit of data that must be converted before sending to a target.
        A piece of text, with a type, that is run through a series of converters and may contain metadata.

        Args:
            request_converters (list[PromptConverter]): A list of PromptConverter objects.
            prompt_value (str): The prompt value.
            prompt_data_type (PromptDataType): The data type of the prompt.
            labels (Optional[dict[str, str]]): The labels to apply to the prompt. Defaults to None.
            metadata (str, Optional): Additional metadata. Defaults to None.

        Raises:
            ValueError: If prompt_converters is not a non-empty list of PromptConverter objects.
            ValueError: If prompt_text is not a string.
        """

        self.request_converters = request_converters
        self.prompt_value = prompt_value
        self.prompt_data_type = prompt_data_type
        self.labels = labels
        self.metadata = metadata

        self.validate()

    def validate(self):
        """
        Validates the NormalizerRequestPiece.

        Raises:
            ValueError: If doesn't validate
        """
        if not self.prompt_value:
            raise ValueError("prompt_text must be a str")

        if not isinstance(self.request_converters, list) or not all(
            isinstance(converter, PromptConverter) for converter in self.request_converters
        ):
            raise ValueError("prompt_converters must be a PromptConverter List")

        # this validates the media exists, if needed
        data_serializer_factory(
            category="prompt-memory-entries", data_type=self.prompt_data_type, value=self.prompt_value
        )


class NormalizerRequest:
    def __init__(
        self,
        request_pieces: list[NormalizerRequestPiece],
        response_converters: list[PromptConverterConfiguration] = [],
        conversation_id: str = None,
    ):
        """
        Represents a normalizer request.

        response_converters will run in the order the response is received.
        """

        self.request_pieces = request_pieces
        self.response_converters = response_converters
        self.conversation_id = conversation_id

    def validate(self):
        if not self.request_pieces or len(self.request_pieces) == 0:
            raise ValueError("request_pieces must be a list of NormalizerRequestPiece objects")

        for piece in self.request_pieces:
            piece.validate()


@dataclass
class NormalizerRequest2(abc.ABC):
    seed_prompt_group: SeedPromptGroup
    request_converter_configurations: list[PromptConverterConfiguration]
    response_converter_configurations: list[PromptConverterConfiguration]

    def __init__(
            self,
            *,
            seed_prompt_group: SeedPromptGroup,
            request_converter_configurations: list[PromptConverterConfiguration] = [],
            response_converter_configurations: list[PromptConverterConfiguration] = []
    ):
        self.seed_prompt_group = seed_prompt_group
        self.request_converter_configurations = request_converter_configurations
        self.response_converter_configurations = response_converter_configurations
