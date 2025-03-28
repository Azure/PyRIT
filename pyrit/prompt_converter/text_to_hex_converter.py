# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class TextToHexConverter(WordLevelConverter):
    """Converts text to a hexadecimal encoded utf-8 string"""

    async def convert_word_async(self, word: str) -> str:
        return word.encode("utf-8").hex().upper()
