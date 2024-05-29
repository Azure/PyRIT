# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import string
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class CaesarConverter(PromptConverter):
    """
    Converter to encode prompt using caesar cipher.

    Encodes by using given offset.
    Using offset=1, 'Hello 123' would encode to 'Ifmmp 234', as each character would shift by 1.
    Shifts for digits 0-9 only work if the offset is less than 10, if the offset is equal to or greather than 10,
    any numeric values will not be shifted.

    Parameters
    ---
    caesar_offset: int
        Offset for caesar cipher, range 0 to 25 (inclusive). Can also be negative for shifting backwards.
    """

    def __init__(self, *, caesar_offset: int) -> None:
        if caesar_offset < -25 or caesar_offset > 25:
            raise ValueError("caesar offset value invalid, must be between -25 and 25 inclusive.")
        self.caesar_offset = caesar_offset

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that caesar cipher encodes the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        output_text = self._caesar(prompt)
        await asyncio.sleep(0)
        return ConverterResult(output_text=output_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def _caesar(self, text: str) -> str:
        def shift(alphabet: str) -> str:
            return alphabet[self.caesar_offset :] + alphabet[: self.caesar_offset]

        alphabet = (string.ascii_lowercase, string.ascii_uppercase, string.digits)
        shifted_alphabet = tuple(map(shift, alphabet))
        translation_table = str.maketrans("".join(alphabet), "".join(shifted_alphabet))
        return text.translate(translation_table)
