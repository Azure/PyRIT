# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from unittest.mock import patch

from pyrit.common.utils import combine_dict, get_random_indices


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
