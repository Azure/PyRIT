# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import binascii
import re
from typing import List, Literal, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class BinAsciiConverter(WordLevelConverter):
    """
    Converts text to various binary-to-ASCII encodings.

    Supports hex, quoted-printable, and UUencode formats.
    """

    EncodingFunc = Literal[
        "hex",
        "quoted-printable",
        "UUencode",
    ]

    def __init__(
        self,
        *,
        encoding_func: EncodingFunc = "hex",
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = None,
        regex: Optional[Union[str, re.Pattern]] = None,
        word_split_separator: Optional[str] = " ",
    ) -> None:
        """
        Initialize the BinAsciiConverter.

        Args:
            encoding_func (str): The encoding function to use. Options: "hex", "quoted-printable", "UUencode".
                Defaults to "hex".
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
            word_split_separator (Optional[str]): Separator used to split words in the input text.
                Defaults to " ".
        """
        super().__init__(
            indices=indices,
            keywords=keywords,
            proportion=proportion,
            regex=regex,
            word_split_separator=word_split_separator,
        )

        if encoding_func not in ["hex", "quoted-printable", "UUencode"]:
            raise ValueError(
                f"Invalid encoding_func '{encoding_func}'. "
                "Must be one of: 'hex', 'quoted-printable', 'UUencode'"
            )

        self._encoding_func = encoding_func

    async def convert_word_async(self, word: str) -> str:
        """
        Convert a word using the specified encoding function.

        Args:
            word (str): The word to encode.

        Returns:
            str: The encoded word.
        """
        if self._encoding_func == "hex":
            return word.encode("utf-8").hex().upper()
        elif self._encoding_func == "quoted-printable":
            return binascii.b2a_qp(word.encode("utf-8")).decode("ascii")
        elif self._encoding_func == "UUencode":
            return self._uuencode_chunk(word)
        else:
            raise ValueError(f"Unsupported encoding function: {self._encoding_func}")

    def _uuencode_chunk(self, text: str) -> str:
        """
        Encode text using UUencode in 45-byte chunks.

        Args:
            text (str): The text to encode.

        Returns:
            str: The UUencoded text.
        """
        payload = text.encode("utf-8")
        hash_chunks = []
        for i in range(0, len(payload), 45):
            test_chunk = payload[i : i + 45]
            hash_chunks.append(binascii.b2a_uu(test_chunk))
        return "".join(chunk.decode("ascii") for chunk in hash_chunks).rstrip("\n")

    def join_words(self, words: list[str]) -> str:
        """
        Join words appropriately based on the encoding type and mode.

        Args:
            words (list[str]): The list of encoded words to join.

        Returns:
            str: The joined string.
        """
        if self._mode == "all":
            if self._encoding_func == "hex":
                return "20".join(words)  # 20 is the hex representation of space
            elif self._encoding_func == "quoted-printable":
                # Quoted-printable uses =20 for space
                return "=20".join(words)
            elif self._encoding_func == "UUencode":
                # UUencode: join with encoded space
                return "".join(words)  # UUencode handles spaces within encoding
        return super().join_words(words=words)
