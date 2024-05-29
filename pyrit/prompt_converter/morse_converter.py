# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class MorseConverter(PromptConverter):
    """
    Converter to encode prompts using morse code.

    Uses '-' and '.' characters, with ' ' to separate characters and '/' to separate words.

    Invalid/unsupported characters replaced with error sequence '........'.
    """

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that morse code encodes the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        output_text = self._morse(prompt)
        await asyncio.sleep(0)
        return ConverterResult(output_text=output_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def _morse(self, text: str) -> str:
        text_clean = " ".join([line.strip() for line in str.splitlines(text)])
        morse_mapping = {
            "A": ".-",
            "B": "-...",
            "C": "-.-.",
            "D": "-..",
            "E": ".",
            "F": "..-.",
            "G": "--.",
            "H": "....",
            "I": "..",
            "J": ".---",
            "K": "-.-",
            "L": ".-..",
            "M": "--",
            "N": "-.",
            "O": "---",
            "P": ".--.",
            "Q": "--.-",
            "R": ".-.",
            "S": "...",
            "T": "-",
            "U": "..-",
            "V": "...-",
            "W": ".--",
            "X": "-..-",
            "Y": "-.--",
            "Z": "--..",
            "0": "-----",
            "1": ".----",
            "2": "..---",
            "3": "...--",
            "4": "....-",
            "5": ".....",
            "6": "-....",
            "7": "--...",
            "8": "---..",
            "9": "----.",
            "'": ".----.",
            '"': ".-..-.",
            ":": "---...",
            "@": ".--.-.",
            ",": "--..--",
            ".": ".-.-.-",
            "!": "-.-.--",
            "?": "..--..",
            "-": "-....-",
            "/": "-..-.",
            "+": ".-.-.",
            "=": "-...-",
            "(": "-.--.",
            ")": "-.--.-",
            "&": ".-...",
            " ": "/",
        }
        extended_mapping = {
            "%": "------..-.-----",
            "À": ".--.-",
            "Å": ".--.-",
            "Ä": ".-.-",
            "Ą": ".-.-",
            " Æ": ".-.-",
            "Ć": "-.-..",
            "Ĉ": "-.-..",
            "Ç": "-.-..",
            "Ĥ": "----",
            "Š": "----",
            "Đ": "..-..",
            "É": "..-..",
            "Ę": "..-..",
            "Ð": "..--.",
            "È": ".-..-",
            "Ł": ".-..-",
            "Ĝ": "--.-.",
            "Ĵ": ".---.",
            "Ń": "--.--",
            "Ñ": "--.--",
            "Ó": "---.",
            "Ö": "---.",
            "Ø": "---.",
            "Ś": "...-...",
            "Ŝ": "...-.",
            "Þ": ".--..",
            "Ü": "..--",
            "Ŭ": "..--",
            "Ź": "--..-.",
            "Ż": "--..-",
        }
        EXTENDED_CHAR_SUPPORT = True
        supported_charset = "".join(morse_mapping.keys())
        print(supported_charset)
        if EXTENDED_CHAR_SUPPORT:
            supported_charset += "".join(extended_mapping.keys())
            morse_mapping = {**morse_mapping, **extended_mapping}
        error_char = "........"
        return " ".join(
            [morse_mapping[char] if char in supported_charset else error_char for char in text_clean.upper()]
        )
