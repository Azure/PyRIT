# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.common.utils import get_random_indices
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class SuperscriptConverter(PromptConverter):
    """
    Converts the input text to superscript text. Supports various modes for conversion.

    Supported modes:
    - 'all': Converts all words. The default mode.
    - 'alternate': Converts every other word. Configurable.
    - 'random': Converts a random selection of words based on a percentage.

    Note:
        This converter leaves characters that do not have a superscript equivalent unchanged.
    """

    def __init__(
        self,
        mode: Optional[str] = "all",
        alternate_step: Optional[int] = 2,
        random_percentage: Optional[int] = 50,
    ):
        """
        Initialize the SuperscriptConverter.

        Args:
            mode (Optional[str]): Conversion mode - 'all', or 'alternate'. Defaults to 'all'.
            alternate_step (Optional[int]): For 'alternate' mode, convert every nth word. Defaults to 2.
            random_percentage (Optional[int]): For 'random' mode, percentage of words to convert. Defaults to 50.
        """
        self.mode = mode
        self.alternate_step = alternate_step
        self.random_percentage = random_percentage
        self._superscript_map = {
            "0": "\u2070",
            "1": "\u00b9",
            "2": "\u00b2",
            "3": "\u00b3",
            "4": "\u2074",
            "5": "\u2075",
            "6": "\u2076",
            "7": "\u2077",
            "8": "\u2078",
            "9": "\u2079",
            "a": "\u1d43",
            "b": "\u1d47",
            "c": "\u1d9c",
            "d": "\u1d48",
            "e": "\u1d49",
            "f": "\u1da0",
            "g": "\u1d4d",
            "h": "\u02b0",
            "i": "\u2071",
            "j": "\u02b2",
            "k": "\u1d4f",
            "l": "\u02e1",
            "m": "\u1d50",
            "n": "\u207f",
            "o": "\u1d52",
            "p": "\u1d56",
            "r": "\u02b3",
            "s": "\u02e2",
            "t": "\u1d57",
            "u": "\u1d58",
            "v": "\u1d5b",
            "w": "\u02b7",
            "x": "\u02e3",
            "y": "\u02b8",
            "z": "\u1dbb",
            "A": "\u1d2c",
            "B": "\u1d2d",
            "D": "\u1d30",
            "E": "\u1d31",
            "G": "\u1d33",
            "H": "\u1d34",
            "I": "\u1d35",
            "J": "\u1d36",
            "K": "\u1d37",
            "L": "\u1d38",
            "M": "\u1d39",
            "N": "\u1d3a",
            "O": "\u1d3c",
            "P": "\u1d3e",
            "R": "\u1d3f",
            "T": "\u1d40",
            "U": "\u1d41",
            "V": "\u2c7d",
            "W": "\u1d42",
            "+": "\u207a",
            "-": "\u207b",
            "=": "\u207c",
            "(": "\u207d",
            ")": "\u207e",
        }

    def _to_superscript(self, text: str) -> str:
        return "".join(self._superscript_map.get(char, char) for char in text)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        words = prompt.split()
        result = []

        if self.mode == "alternate":
            # Convert every nth word
            for i, word in enumerate(words):
                if i % self.alternate_step == 0:
                    result.append(self._to_superscript(word))
                else:
                    result.append(word)

        elif self.mode == "random":
            # Convert random words based on percentage
            word_count = len(words)
            random_indices = get_random_indices(0, word_count, self.random_percentage / 100.0)
            for i, word in enumerate(words):
                if i in random_indices:
                    result.append(self._to_superscript(word))
                else:
                    result.append(word)

        # TODO: add more modes here

        else:
            # Convert every word if mode is not recognized or it's actually 'all'
            for word in words:
                result.append(self._to_superscript(word))

        converted_text = " ".join(result)
        return ConverterResult(output_text=converted_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
