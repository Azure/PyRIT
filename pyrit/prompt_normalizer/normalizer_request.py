# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.models import data_serializer_factory, PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.prompt_response_converter_configuration import PromptResponseConverterConfiguration


class NormalizerRequestPiece(abc.ABC):

    def __init__(
        self,
        *,
        prompt_value: str,
        prompt_data_type: PromptDataType,
        request_converters: list[PromptConverter] = [],
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
            metadata (str, Optional): Additional metadata. Defaults to None.

        Raises:
            ValueError: If prompt_converters is not a non-empty list of PromptConverter objects.
            ValueError: If prompt_text is not a string.
        """

        self.request_converters = request_converters
        self.prompt_value = prompt_value
        self.prompt_data_type = prompt_data_type
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
        data_serializer_factory(data_type=self.prompt_data_type, value=self.prompt_value)


class NormalizerRequest:
    def __init__(
        self,
        request_pieces: list[NormalizerRequestPiece],
        response_converters: list[PromptResponseConverterConfiguration] = [],
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
