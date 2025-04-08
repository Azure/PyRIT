# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class UnicodeReplacementConverter(WordLevelConverter):
    """Simple converter that returns the unicode representation of the prompt."""

    def __init__(self, *, encode_spaces: bool = False, mode: str = "all", **mode_kwargs):
        """
        Args:
            encode_spaces (bool): If True, spaces in the prompt will be replaced with unicode representation.
                                  Default is False.
        """
        super().__init__(mode=mode, **mode_kwargs)
        self.encode_spaces = encode_spaces

    async def convert_word_async(self, word: str) -> str:
        return "".join(f"\\u{ord(ch):04x}" for ch in word)

    def join_words(self, words: list[str]) -> str:
        if self.encode_spaces:
            return "\\u0020".join(words)
        return super().join_words(words)
