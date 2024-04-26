# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import data_serializer_factory


class NormalizerRequestPiece(abc.ABC):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_converters: list[PromptConverter],
        prompt_text: str,
        prompt_data_type: PromptDataType,
        metadata: str = None,
    ) -> None:
        """
        Represents a piece of a normalizer request.

        It represents the minimum unit of data that must be converted before sending to a target.
        A piece of text, with a type, that is run through a series of converters and may contain metadata.

        Args:
            prompt_converters (list[PromptConverter]): A list of PromptConverter objects.
            prompt_text (str): The prompt text.
            prompt_data_type (PromptDataType): The data type of the prompt.
            metadata (str, optional): Additional metadata. Defaults to None.

        Raises:
            ValueError: If prompt_converters is not a non-empty list of PromptConverter objects.
            ValueError: If prompt_text is not a string.
        """

        self.prompt_converters = prompt_converters
        self.prompt_text = prompt_text
        self.prompt_data_type = prompt_data_type
        self.metadata = metadata

        self.validate()

    def validate(self):
        """
        Validates the NormalizerRequestPiece.

        Raises:
            ValueError: If doesn't validate
        """
        if not self.prompt_text:
            raise ValueError("prompt_text must be a str")

        if not isinstance(self.prompt_converters, list) or not all(
            isinstance(converter, PromptConverter) for converter in self.prompt_converters
        ):
            raise ValueError("prompt_converters must be a PromptConverter List")

        # this validates the media exists, if needed
        data_serializer_factory(data_type=self.prompt_data_type, prompt_text=self.prompt_text)


class NormalizerRequest:
    def __init__(self, request_pieces: list[NormalizerRequestPiece]):
        self.request_pieces = request_pieces

    def validate(self):
        if not self.request_pieces or len(self.request_pieces) == 0:
            raise ValueError("request_pieces must be a list of NormalizerRequestPiece objects")

        for piece in self.request_pieces:
            piece.validate
