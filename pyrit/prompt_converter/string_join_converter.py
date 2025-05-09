# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class StringJoinConverter(WordLevelConverter):
    """Converts text by joining its characters with the specified join value"""

    def __init__(self, *, join_value="-"):
        super().__init__()
        self.join_value = join_value

    async def convert_word_async(self, word: str) -> str:
        return self.join_value.join(word)
