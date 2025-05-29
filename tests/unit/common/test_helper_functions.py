# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import re

from unittest.mock import patch

from pyrit.common.utils import combine_dict, get_random_indices, select_word_indices


def test_combine_non_empty_dict():
    dict1 = {"a": "b"}
    dict2 = {"c": "d"}
    assert combine_dict(dict1, dict2) == {"a": "b", "c": "d"}


def test_combine_empty_dict():
    dict1 = {}
    dict2 = {}
    assert combine_dict(dict1, dict2) == {}


def test_combine_first_empty_dict():
    dict1 = {"a": "b"}
    dict2 = {}
    assert combine_dict(dict1, dict2) == {"a": "b"}


def test_combine_second_empty_dict():
    dict1 = {}
    dict2 = {"c": "d"}
    assert combine_dict(dict1, dict2) == {"c": "d"}


def test_combine_dict_same_keys():
    dict1 = {"c": "b"}
    dict2 = {"c": "d"}
    assert combine_dict(dict1, dict2) == {"c": "d"}


def test_get_random_indices():
    with patch("random.sample", return_value=[2, 4, 6]):
        result = get_random_indices(start=0, size=10, proportion=0.3)
        assert result == [2, 4, 6]

    assert get_random_indices(start=5, size=10, proportion=0) == []
    assert sorted(get_random_indices(start=27, size=10, proportion=1)) == list(range(27, 37))

    with pytest.raises(ValueError):
        get_random_indices(start=-1, size=10, proportion=0.5)
    with pytest.raises(ValueError):
        get_random_indices(start=0, size=0, proportion=0.5)
    with pytest.raises(ValueError):
        get_random_indices(start=0, size=10, proportion=-1)
    with pytest.raises(ValueError):
        get_random_indices(start=0, size=10, proportion=1.01)


def test_word_indices_all_mode():
    assert select_word_indices(words=["word1", "word2", "word3"], mode="all") == [0, 1, 2]
    assert select_word_indices(words=[], mode="all") == []

    large_word_list = [f"word{i}" for i in range(1000)]
    assert select_word_indices(words=large_word_list, mode="all") == list(range(1000))


def test_word_indices_custom_mode():
    assert select_word_indices(words=["word1", "word2", "word3"], mode="custom", indices=[0, 2]) == [0, 2]
    assert select_word_indices(words=["word1", "word2", "word3"], mode="custom", indices=[]) == []
    assert select_word_indices(words=["word1", "word2", "word3"], mode="custom") == []
    assert select_word_indices(words=[], mode="custom", indices=[0, 1]) == []

    with pytest.raises(ValueError):
        select_word_indices(words=["word1", "word2", "word3"], mode="custom", indices=[0, 3, -1, 5])

    large_word_list = [f"word{i}" for i in range(1000)]
    custom_indices = list(range(0, 1000, 10))  # every 10th index
    assert select_word_indices(words=large_word_list, mode="custom", indices=custom_indices) == custom_indices


def test_word_indices_keywords_mode():
    assert select_word_indices(words=["word1", "word2", "pyrit", "word4"], mode="keywords", keywords=["pyrit"]) == [2]
    assert select_word_indices(
        words=["word1", "pyrit", "word3", "test"], mode="keywords", keywords=["pyrit", "test"]
    ) == [1, 3]

    assert select_word_indices(words=[], mode="keywords", keywords=["pyrit"]) == []
    assert select_word_indices(words=["word1", "word2", "word3"], mode="keywords") == []
    assert select_word_indices(words=["word1", "word2", "word3"], mode="keywords", keywords=[]) == []
    assert select_word_indices(words=["word1", "word2", "word3"], mode="keywords", keywords=["pyrit"]) == []

    large_word_list = [f"word{i}" for i in range(1000)]
    large_word_list[123] = "pyrit"
    large_word_list[456] = "pyrit"
    large_word_list[789] = "test"
    assert select_word_indices(words=large_word_list, mode="keywords", keywords=["pyrit", "test"]) == [123, 456, 789]


def test_word_indices_regex_mode():
    assert select_word_indices(words=["word1", "word2", "pyrit", "word4"], mode="regex", regex=r"word\d") == [0, 1, 3]
    assert select_word_indices(words=["word1", "word2", "word3"], mode="regex") == [0, 1, 2]  # default pattern is "."
    assert select_word_indices(words=["word1", "word2", "word3"], mode="regex", regex=r"pyrit") == []
    assert select_word_indices(words=[], mode="regex", regex=r"word\d") == []

    pattern = re.compile(r"word\d")
    assert select_word_indices(words=["word1", "word2", "pyrit", "word4"], mode="regex", regex=pattern) == [0, 1, 3]

    large_word_list = [f"word{i}" for i in range(1000)]
    large_word_list[123] = "don't"
    large_word_list[456] = "match"
    large_word_list[789] = "these"
    regex_results = select_word_indices(words=large_word_list, mode="regex", regex=r"word\d+")
    assert len(regex_results) == 997  # 1000 - 3 (123, 456, 789 don't match)
    assert 123 not in regex_results
    assert 456 not in regex_results
    assert 789 not in regex_results


def test_word_indices_random_mode():
    with patch("random.sample", return_value=[0, 2]):
        result = select_word_indices(words=["word1", "word2", "word3", "word4"], mode="random")
        assert result == [0, 2]
        result = select_word_indices(words=["word1", "word2", "word3", "word4"], mode="random", proportion=0.5)
        assert result == [0, 2]

    assert select_word_indices(words=[], mode="random", proportion=0.5) == []
    assert select_word_indices(words=["word1", "word2", "word3", "word4"], mode="random", proportion=0) == []
    assert len(select_word_indices(words=["word1", "word2", "word3", "word4"], mode="random", proportion=1)) == 4

    # Test with actual randomness but verify length is correct
    large_word_list = [f"word{i}" for i in range(1000)]
    random_results = select_word_indices(words=large_word_list, mode="random", proportion=0.43)
    assert len(random_results) == 430  # 43% of 1000


def test_word_indices_invalid_mode():
    # Should default to "all" mode with warning
    assert select_word_indices(words=["word1", "word2"], mode="invalid") == [0, 1]  # type: ignore
    assert select_word_indices(words=["word1", "word2", "word3"], mode="invalid") == [0, 1, 2]  # type: ignore
    assert select_word_indices(words=[], mode="invalid") == []  # type: ignore
