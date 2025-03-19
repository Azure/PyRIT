# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class SuperscriptConverter(PromptConverter):
    """
    Converts the input text to superscript text.
    
    Note: This converter leaves unsupported characters unchanged.
    """

    def __init__(self):
        self._superscript_map = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
            "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ",
            "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
            "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
            "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
            "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ", "G": "ᴳ",
            "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ",
            "M": "ᴹ", "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "R": "ᴿ",
            "T": "ᵀ", "U": "ᵁ", "V": "ⱽ", "W": "ᵂ", "+": "⁺",
            "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        }

    def _to_superscript(self, text: str) -> str:
        return "".join(self._superscript_map.get(char, char) for char in text)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        words = prompt.split()
        result = []

        for word in words:
            result.append(self._to_superscript(word))

        converted_text = " ".join(result)
        result = ConverterResult(output_text=converted_text, output_type="text")
        return result

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
