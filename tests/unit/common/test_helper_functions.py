# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

from pyrit.common.utils import combine_dict, select_word_indices


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


def test_word_indices_selection():
    assert select_word_indices(words=["word1", "word2", "word3"], mode="all") == [0, 1, 2]
    assert select_word_indices(words=["word1", "word2", "word3"], mode="custom", indices=[0, 2]) == [0, 2]
    assert select_word_indices(words=["word1", "word2", "pyrit", "word4"], mode="keywords", keywords=["pyrit"]) == [2]
    assert select_word_indices(words=["word1", "word2", "pyrit", "word4"], mode="regex", regex=r"word\d") == [0, 1, 3]

    with patch("random.sample", return_value=[0, 2]):
        result = select_word_indices(words=["word1", "word2", "word3", "word4"], mode="random", percentage=50)
    assert result == [0, 2]

    assert select_word_indices(words=["word1", "word2"], mode="invalid_mode") == [0, 1]
