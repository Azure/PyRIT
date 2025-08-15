# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class FirstLetterConverter(WordLevelConverter):
    """
    Replaces each word of the prompt with its first letter (or digit).
    Whitespace and words that do not contain any letter or digit are ignored.
    """

    async def convert_word_async(self, word: str) -> str:
        stripped_word = "".join(filter(str.isalnum, word))
        return stripped_word[:1]

    def join_words(self, words: list[str]) -> str:
        return "".join(words)  # Join first letters without separator
