# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class EmojiConverter(WordLevelConverter):
    """
    Converts English text to randomly chosen circle or square character emojis.

    Inspired by https://github.com/BASI-LABS/parseltongue/blob/main/src/utils.ts
    """

    emoji_dict = {
        "a": ["🅐", "🅰️", "🄰"],
        "b": ["🅑", "🅱️", "🄱"],
        "c": ["🅒", "🅲", "🄲"],
        "d": ["🅓", "🅳", "🄳"],
        "e": ["🅔", "🅴", "🄴"],
        "f": ["🅕", "🅵", "🄵"],
        "g": ["🅖", "🅶", "🄶"],
        "h": ["🅗", "🅷", "🄷"],
        "i": ["🅘", "🅸", "🄸"],
        "j": ["🅙", "🅹", "🄹"],
        "k": ["🅚", "🅺", "🄺"],
        "l": ["🅛", "🅻", "🄻"],
        "m": ["🅜", "🅼", "🄼"],
        "n": ["🅝", "🅽", "🄽"],
        "o": ["🅞", "🅾️", "🄾"],
        "p": ["🅟", "🅿️", "🄿"],
        "q": ["🅠", "🆀", "🅀"],
        "r": ["🅡", "🆁", "🅁"],
        "s": ["🅢", "🆂", "🅂"],
        "t": ["🅣", "🆃", "🅃"],
        "u": ["🅤", "🆄", "🅄"],
        "v": ["🅥", "🆅", "🅅"],
        "w": ["🅦", "🆆", "🅆"],
        "x": ["🅧", "🆇", "🅇"],
        "y": ["🅨", "🆈", "🅈"],
        "z": ["🅩", "🆉", "🅉"],
    }

    async def convert_word_async(self, word: str) -> str:
        word = word.lower()
        result = []
        for char in word:
            if char in EmojiConverter.emoji_dict:
                result.append(random.choice(EmojiConverter.emoji_dict[char]))
            else:
                result.append(char)
        return "".join(result)
