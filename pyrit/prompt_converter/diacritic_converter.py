# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import unicodedata

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class DiacriticConverter(PromptConverter):
    """
    A PromptConverter that applies diacritics to specified characters in a string.
    """

    def __init__(self, target_chars: str = "aeiou", accent: str = "acute"):
        """
        Initializes the DiacriticConverter.

        Args:
            target_chars (str): Characters to apply the diacritic to. Defaults to "aeiou".
            accent (str): Type of diacritic to apply (default is 'acute').

             Available options are:
            - 'acute': "\u0301"
            - 'grave': "\u0300"
            - 'tilde': "\u0303"
            - 'umlaut': "\u0308"

        Raises:
            ValueError: If `target_chars` is empty.
        """
        super().__init__()

        if not target_chars:
            raise ValueError("target_chars cannot be empty.")

        self._target_chars = set(target_chars)
        self._accent = accent

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

    def _get_accent_mark(self) -> str:
        """
        Retrieves the Unicode character for the specified diacritic accent.
        """
        diacritics = {
            "acute": "\u0301",  # Acute accent
            "grave": "\u0300",  # Grave accent
            "tilde": "\u0303",  # Tilde
            "umlaut": "\u0308",  # Umlaut/Diaeresis
            # Add other accents as needed
        }

        if self._accent not in diacritics:
            raise ValueError(f"Accent '{self._accent}' not recognized. Choose from {list(diacritics.keys())}.")

        return diacritics[self._accent]

    def _add_diacritic(self, text: str) -> str:
        """
        Applies the diacritic to each specified target character in the text.

        Args:
            text (str): The input text in which diacritics will be added.

        Returns:
            str: The text with diacritical marks applied to specified characters.
        """
        accent_mark = self._get_accent_mark()

        # Apply accent to each target character in the string
        return "".join(
            unicodedata.normalize("NFC", char + accent_mark) if char in self._target_chars else char for char in text
        )

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt by applying diacritics to target characters.

        Args:
            prompt (str): The prompt to be converted.

        Returns:
            ConverterResult: The result containing the modified prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Only 'text' input type is supported.")

        # Apply diacritic transformation to the entire string
        modified_text = self._add_diacritic(prompt)

        return ConverterResult(output_text=modified_text, output_type="text")
