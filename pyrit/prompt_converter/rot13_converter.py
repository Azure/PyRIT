# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class ROT13Converter(WordLevelConverter):
    """
    Encodes prompts using the ROT13 cipher.
    """

    async def convert_word_async(self, word: str) -> str:
        return codecs.encode(word, "rot13")
