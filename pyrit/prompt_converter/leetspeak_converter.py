# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class LeetspeakConverter(PromptConverter):
    """Converts a string to a leetspeak version"""

    def __init__(self) -> None:
        self.leet_substitutions = {
            "a": ["4", "@", "/\\", "@", "^", "/-\\"],
            "b": ["8", "6", "13", "|3", "/3", "!3"],
            "c": ["(", "[", "<", "{"],
            "e": ["3"],
            "g": ["9"],
            "i": ["1", "!"],
            "l": ["1", "|"],
            "o": ["0"],
            "s": ["5", "$"],
            "t": ["7"],
            "z": ["2"],
        }

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter to generate leatspeak version of a prompt.
        Since there are multiple character variations, this is non-deterministic.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        converted_prompt = []
        for char in prompt:
            if char.lower() in self.leet_substitutions:
                converted_prompt.append(random.choice(self.leet_substitutions.get(char.lower(), char)))
            else:
                converted_prompt.append(char)
        await asyncio.sleep(0)
        return ConverterResult(output_text="".join(converted_prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
