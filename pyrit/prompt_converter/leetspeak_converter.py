# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
import random


class LeatspeakConverter(PromptConverter):
    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter to generate leatspeak version of a prompt
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        leet_substitutions = {
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

        converted_prompt = ""

        for char in prompt.lower():
            if char in leet_substitutions:
                converted_prompt += random.choice(leet_substitutions[char])
            else:
                converted_prompt += char

        return ConverterResult(output_text=converted_prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
