# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from pyrit.common.utils import select_word_indices
from pyrit.models.literals import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_converter.prompt_converter import ConverterResult


class WordLevelConverter(PromptConverter):
    def __init__(self, mode: str = "all", **mode_kwargs):
        self.mode = mode
        self.mode_kwargs = mode_kwargs

    @abc.abstractmethod
    async def convert_word_async(self, word: str) -> str:
        pass

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        if prompt is None:
            raise TypeError("Prompt cannot be None")

        if input_type != "text":
            raise ValueError(f"Input type {input_type} not supported")

        words = prompt.split()
        selected_indices = select_word_indices(words=words, mode=self.mode, **self.mode_kwargs)

        # Convert only selected words
        for idx in selected_indices:
            words[idx] = await self.convert_word_async(words[idx])

        return ConverterResult(output_text=" ".join(words), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
