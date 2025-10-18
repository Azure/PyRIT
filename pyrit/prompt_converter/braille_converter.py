# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class BrailleConverter(PromptConverter):
    """
    Encodes text using Braille representation.
    """

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """Converts the given prompt using the Braille representation."""
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        
        brail_text = self._get_braille(prompt)

        return ConverterResult(output_text=brail_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

    def _get_braille(self, text) -> str:
        """
        This retrieves the braille representation of the input text.

        Taken from Garak https://github.com/NVIDIA/garak/blob/main/garak/probes/encoding.py
        """
        characterUnicodes = {
            "a": "\u2801",
            "b": "\u2803",
            "k": "\u2805",
            "l": "\u2807",
            "c": "\u2809",
            "i": "\u280A",
            "f": "\u280B",
            "m": "\u280D",
            "s": "\u280E",
            "p": "\u280F",
            "e": "\u2811",
            "h": "\u2813",
            "o": "\u2815",
            "r": "\u2817",
            "d": "\u2819",
            "j": "\u281A",
            "g": "\u281B",
            "n": "\u281D",
            "t": "\u281E",
            "q": "\u281F",
            "u": "\u2825",
            "v": "\u2827",
            "x": "\u282D",
            "z": "\u2835",
            "w": "\u283A",
            "y": "\u283D",
            "num": "\u283C",
            "caps": "\u2820",
            ".": "\u2832",
            "'": "\u2804",
            ",": "\u2802",
            "-": "\u2824",
            "/": "\u280C",
            "!": "\u2816",
            "?": "\u2826",
            "$": "\u2832",
            ":": "\u2812",
            ";": "\u2830",
            "(": "\u2836",
            ")": "\u2836",
            "1": "\u2801",
            "2": "\u2803",
            "3": "\u2809",
            "4": "\u2819",
            "5": "\u2811",
            "6": "\u280B",
            "7": "\u281B",
            "8": "\u2813",
            "9": "\u280A",
            "0": "\u281A",
            " ": " ",
        }
        numberPunctuations = [".", ",", "-", "/", "$"]
        escapeCharacters = ["\n", "\r", "\t"]

        output = ""

        for char in text:
            is_number = False
            if char in escapeCharacters:
                output += char
            elif char.isupper():
                if char.lower() in characterUnicodes:
                    output += characterUnicodes["caps"]
                    output += characterUnicodes[char.lower()]
            elif char in characterUnicodes:
                if char.isdigit():
                    if not is_number:
                        is_number = True
                        output += characterUnicodes["num"]
                output += characterUnicodes[char]
            if is_number and char not in numberPunctuations:
                is_number = False

        return output
