# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal
from pyrit.prompt_converter.word_level_converter import WordLevelConverter

class BinAsciiConverter(WordLevelConverter):
    """
    Converts text to a hexadecimal encoded utf-8 string.
    """
    EncodingFunc = Literal[
        "hex",
        "quoted-printable",
        "UUencode",
    ]

    def __init__(self, *, encoding_func: EncodingFunc = "hex"):
        super().__init__()
        self._encoding_func = encoding_func

    async def convert_word_async(self, word: str) -> str:
        return word.encode("utf-8").hex().upper()

    def join_words(self, words: list[str]) -> str:
        if self._mode == "all":
            return "20".join(words)  # 20 is the hex representation of space
        return super().join_words(words)
