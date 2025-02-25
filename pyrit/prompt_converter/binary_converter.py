# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from enum import Enum

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class BinaryConverter(PromptConverter):
    """
    A converter that transforms input text into its binary representation
    with configurable bits per character (8, 16, or 32).
    """

    class BitsPerChar(Enum):
        BITS_8 = 8
        BITS_16 = 16
        BITS_32 = 32

    def __init__(self, bits_per_char: BinaryConverter.BitsPerChar = BitsPerChar.BITS_16):
        if not isinstance(bits_per_char, BinaryConverter.BitsPerChar):
            raise TypeError("bits_per_char must be an instance of BinaryConverter.BitsPerChar Enum.")
        self.bits_per_char = bits_per_char

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the input text to binary representation with specified bits per character.

        Args:
            prompt (str): The input text to be converted.
            input_type (PromptDataType): The type of the input data.

        Returns:
            ConverterResult: The result containing the binary representation of the input text.

        Raises:
            ValueError: If the input type is not supported or bits_per_char is invalid.
        """
        if not self.input_supported(input_type):
            raise ValueError(f"Input type '{input_type}' not supported.")

        bits = self.bits_per_char.value

        # Check if bits_per_char is sufficient for the characters in the prompt
        max_code_point = max((ord(char) for char in prompt), default=0)
        min_bits_required = max_code_point.bit_length()
        if bits < min_bits_required:
            raise ValueError(
                f"bits_per_char={bits} is too small for the characters in the prompt. "
                f"Minimum required bits: {min_bits_required}."
            )

        # Convert each character in the prompt to its binary representation
        binary_representation = " ".join(format(ord(char), f"0{bits}b") for char in prompt)
        return ConverterResult(output_text=binary_representation, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
