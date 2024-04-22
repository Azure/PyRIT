# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter


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

        if not isinstance(prompt_converters, list) or not all(
            isinstance(converter, PromptConverter) for converter in prompt_converters
        ):
            raise ValueError("prompt_converters must be a PromptConverter List")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")

        self.prompt_converters = prompt_converters
        self.prompt_text = prompt_text
        self.prompt_data_type = prompt_data_type
        self.metadata = metadata


class NormalizerRequest:
    def __init__(self, request_pieces: list[NormalizerRequestPiece]):
        self.request_pieces = request_pieces
