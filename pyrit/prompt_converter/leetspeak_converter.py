# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class LeetspeakConverter(PromptConverter):
    """Converts a string to a leetspeak version"""

    def __init__(self, deterministic: bool = False, custom_substitutions: dict = None) -> None:
        """
        Initialize the converter with optional deterministic mode and custom substitutions.
        
        :param deterministic: If True, use the first substitution for each character.
        :param custom_substitutions: A dictionary of custom substitutions to override the defaults.
        """
        default_substitutions  = {
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

        # Use custom substitutions if provided, otherwise default to the standard ones
        self.leet_substitutions = custom_substitutions if custom_substitutions else default_substitutions
        self.deterministic = deterministic

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt to leetspeak.
        
        :param prompt: The text to convert.
        :param input_type: The type of input data.
        :return: A ConverterResult containing the leetspeak version of the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        converted_prompt = []
        for char in prompt:
            if self.deterministic:
                # Use the first substitution for deterministic mode
                converted_prompt.append(self.leet_substitutions.get(char.lower(), [char])[0])
            else:
                # Randomly select a substitution for each character
                converted_prompt.append(random.choice(self.leet_substitutions.get(char.lower(), char)))
        else:
            converted_prompt.append(char)


        return ConverterResult(output_text="".join(converted_prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
