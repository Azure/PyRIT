# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter


class Prompt(abc.ABC):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_target: PromptTarget,
        prompt_converters: list[PromptConverter],
        prompt_text: str,
        conversation_id: str,
    ) -> None:

        if (
            not isinstance(prompt_converters, list)
            or len(prompt_converters) == 0
            or not all(isinstance(converter, PromptConverter) for converter in prompt_converters)
        ):
            raise ValueError("prompt_converters must be a list[PromptConverter] and be non-empty")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")


        self.prompt_converters = prompt_converters
        self.prompt_text = prompt_text
        self.prompt_data_type = prompt_data_type
        self.metadata = metadata


class PromptRequestResponse():    
    def __init__(self, request_pieces: list[PromptRequestPiece]):
        self.request_pieces = request_pieces
