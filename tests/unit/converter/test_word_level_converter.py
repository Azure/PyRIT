# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from unittest.mock import patch

import pytest

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class SimpleWordLevelConverter(WordLevelConverter):
    """Simple implementation of WordLevelConverter for testing purposes"""

    async def convert_word_async(self, word: str) -> str:
        return word.upper()


class TestWordLevelConverter:
    @pytest.mark.asyncio
    async def test_convert_async_all_mode(self):
        converter = SimpleWordLevelConverter()
        result = await converter.convert_async(prompt="hello world this is a test")
        assert result.output_text == "HELLO WORLD THIS IS A TEST"

    @pytest.mark.asyncio
    async def test_convert_async_custom_mode(self):
        converter = SimpleWordLevelConverter(indices=[0, 2, 4])
        result = await converter.convert_async(prompt="hello world this is a test")
        assert result.output_text == "HELLO world THIS is A test"

    @pytest.mark.asyncio
    async def test_convert_async_keywords_mode(self):
        converter = SimpleWordLevelConverter(keywords=["hello", "test"])
        result = await converter.convert_async(prompt="hello world this is a test")
        assert result.output_text == "HELLO world this is a TEST"

    @pytest.mark.asyncio
    async def test_convert_async_regex_mode(self):
        converter = SimpleWordLevelConverter(regex=r"^[aeiou]")
        result = await converter.convert_async(prompt="hello awesome interesting text")
        assert result.output_text == "hello AWESOME INTERESTING text"

    @pytest.mark.asyncio
    async def test_convert_async_random_mode(self):
        with patch("pyrit.prompt_converter.word_level_converter.get_random_indices", return_value=[0, 2]):
            converter = SimpleWordLevelConverter(proportion=0.5)
            result = await converter.convert_async(prompt="hello world this is")
            assert result.output_text == "HELLO world THIS is"

    @pytest.mark.asyncio
    async def test_join_words_override(self):
        class CustomJoinConverter(SimpleWordLevelConverter):
            def join_words(self, words: list[str]) -> str:
                return "#".join(words)

        converter = CustomJoinConverter()
        result = await converter.convert_async(prompt="hello world test")
        assert result.output_text == "HELLO#WORLD#TEST"

    def test_select_word_indices_all_mode(self):
        converter = SimpleWordLevelConverter()
        assert converter._select_word_indices(words=["word1", "word2", "word3"]) == [0, 1, 2]
        assert converter._select_word_indices(words=[]) == []

        large_word_list = [f"word{i}" for i in range(1000)]
        assert converter._select_word_indices(words=large_word_list) == list(range(1000))

    def test_select_word_indices_indices_mode(self):
        converter = SimpleWordLevelConverter(indices=[0, 2])
        assert converter._select_word_indices(words=["word1", "word2", "word3"]) == [0, 2]

        converter = SimpleWordLevelConverter(indices=[])
        assert converter._select_word_indices(words=["word1", "word2", "word3"]) == []

        converter = SimpleWordLevelConverter(indices=[0, 1])
        assert converter._select_word_indices(words=[]) == []

        with pytest.raises(ValueError):
            converter = SimpleWordLevelConverter(indices=[0, 3, -1, 5])
            converter._select_word_indices(words=["word1", "word2", "word3"])

        large_word_list = [f"word{i}" for i in range(1000)]
        custom_indices = list(range(0, 1000, 10))  # every 10th index
        converter = SimpleWordLevelConverter(indices=custom_indices)
        assert converter._select_word_indices(words=large_word_list) == custom_indices

    def test_select_word_indices_keywords_mode(self):
        converter = SimpleWordLevelConverter(keywords=["pyrit", "test"])
        assert converter._select_word_indices(words=["word1", "pyrit", "word3", "test"]) == [1, 3]

        converter = SimpleWordLevelConverter(keywords=["pyrit"])
        assert converter._select_word_indices(words=[]) == []

        converter = SimpleWordLevelConverter(keywords=[])
        assert converter._select_word_indices(words=["word1", "word2", "word3"]) == []

        converter = SimpleWordLevelConverter(keywords=["pyrit"])
        assert converter._select_word_indices(words=["word1", "word2", "word3"]) == []

        large_word_list = [f"word{i}" for i in range(1000)]
        large_word_list[123] = "pyrit"
        large_word_list[456] = "pyrit"
        large_word_list[789] = "test"
        converter = SimpleWordLevelConverter(keywords=["pyrit", "test"])
        assert converter._select_word_indices(words=large_word_list) == [123, 456, 789]

    def test_select_word_indices_regex_mode(self):
        converter = SimpleWordLevelConverter(regex=r"word\d")
        assert converter._select_word_indices(words=["word1", "word2", "pyrit", "word4"]) == [0, 1, 3]

        converter = SimpleWordLevelConverter(regex=r"pyrit")
        assert converter._select_word_indices(words=["word1", "word2", "word3"]) == []

        converter = SimpleWordLevelConverter(regex=r"word\d")
        assert converter._select_word_indices(words=[]) == []

        pattern = re.compile(r"word\d")
        converter = SimpleWordLevelConverter(regex=pattern)
        assert converter._select_word_indices(words=["word1", "word2", "pyrit", "word4"]) == [0, 1, 3]

        large_word_list = [f"word{i}" for i in range(1000)]
        large_word_list[123] = "don't"
        large_word_list[456] = "match"
        large_word_list[789] = "these"
        converter = SimpleWordLevelConverter(regex=r"word\d+")
        regex_results = converter._select_word_indices(words=large_word_list)
        assert len(regex_results) == 997  # 1000 - 3 (123, 456, 789 don't match)
        assert 123 not in regex_results
        assert 456 not in regex_results
        assert 789 not in regex_results

    def test_select_word_indices_random_mode(self):
        with patch("pyrit.prompt_converter.word_level_converter.get_random_indices", return_value=[0, 2]):
            converter = SimpleWordLevelConverter(proportion=0.5)
            result = converter._select_word_indices(words=["word1", "word2", "word3", "word4"])
            assert result == [0, 2]

        converter = SimpleWordLevelConverter(proportion=0.5)
        assert converter._select_word_indices(words=[]) == []

        converter = SimpleWordLevelConverter(proportion=0)
        assert converter._select_word_indices(words=["word1", "word2", "word3", "word4"]) == []

        converter = SimpleWordLevelConverter(proportion=1.0)
        assert len(converter._select_word_indices(words=["word1", "word2", "word3", "word4"])) == 4

        # Test with actual randomness but verify length is correct
        large_word_list = [f"word{i}" for i in range(1000)]
        converter = SimpleWordLevelConverter(proportion=0.43)
        random_results = converter._select_word_indices(words=large_word_list)
        assert len(random_results) == 430  # 43% of 1000

    def test_initialization_and_validation(self):
        # Default mode (all words)
        converter = SimpleWordLevelConverter()
        assert converter._mode == "all"

        # Test that multiple criteria raise an error
        with pytest.raises(ValueError, match="Only one selection criteria can be provided"):
            SimpleWordLevelConverter(indices=[0], keywords=["test"])
        with pytest.raises(ValueError, match="Only one selection criteria can be provided"):
            SimpleWordLevelConverter(proportion=0.5, regex=r"test")

        # Test individual modes are set correctly
        converter = SimpleWordLevelConverter(indices=[0, 1])
        assert converter._mode == "indices"
        assert converter._indices == [0, 1]
        converter = SimpleWordLevelConverter(keywords=["test"])
        assert converter._mode == "keywords"
        assert converter._keywords == ["test"]
        converter = SimpleWordLevelConverter(regex=r"test")
        assert converter._mode == "regex"
        assert converter._regex == r"test"
        converter = SimpleWordLevelConverter(proportion=0.5)
        assert converter._mode == "proportion"
        assert converter._proportion == 0.5
