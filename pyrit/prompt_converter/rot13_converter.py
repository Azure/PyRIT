# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class ROT13Converter(WordLevelConverter):
    """Simple converter that just ROT13 encodes the prompt"""

    async def convert_word_async(self, word: str) -> str:
        return codecs.encode(word, "rot13")
