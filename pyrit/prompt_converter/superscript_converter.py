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
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "a": "ᵃ",
            "b": "ᵇ",
            "c": "ᶜ",
            "d": "ᵈ",
            "e": "ᵉ",
            "f": "ᶠ",
            "g": "ᵍ",
            "h": "ʰ",
            "i": "ⁱ",
            "j": "ʲ",
            "k": "ᵏ",
            "l": "ˡ",
            "m": "ᵐ",
            "n": "ⁿ",
            "o": "ᵒ",
            "p": "ᵖ",
            "r": "ʳ",
            "s": "ˢ",
            "t": "ᵗ",
            "u": "ᵘ",
            "v": "ᵛ",
            "w": "ʷ",
            "x": "ˣ",
            "y": "ʸ",
            "z": "ᶻ",
            "A": "ᴬ",
            "B": "ᴮ",
            "D": "ᴰ",
            "E": "ᴱ",
            "G": "ᴳ",
            "H": "ᴴ",
            "I": "ᴵ",
            "J": "ᴶ",
            "K": "ᴷ",
            "L": "ᴸ",
            "M": "ᴹ",
            "N": "ᴺ",
            "O": "ᴼ",
            "P": "ᴾ",
            "R": "ᴿ",
            "T": "ᵀ",
            "U": "ᵁ",
            "V": "ⱽ",
            "W": "ᵂ",
            "+": "⁺",
            "-": "⁻",
            "=": "⁼",
            "(": "⁽",
            ")": "⁾",
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
