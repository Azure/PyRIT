# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter


class NormalizerRequestPiece(abc.ABC):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_converters: "list[PromptConverter]",
        prompt_text: str,
        prompt_data_type: PromptDataType,
        metadata: str = None,
    ) -> None:

        if (
            not isinstance(prompt_converters, list)
            or len(prompt_converters) == 0
            or not all(isinstance(converter, PromptConverter) for converter in prompt_converters)
        ):
            raise ValueError("prompt_converters must be a PromptConverter List and be non-empty")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")

        self.prompt_converters = prompt_converters
        self.prompt_text = prompt_text
        self.prompt_data_type = prompt_data_type
        self.metadata = metadata


class NormalizerRequest:
    def __init__(self, request_pieces: list[NormalizerRequestPiece]):
        self.request_pieces = request_pieces
