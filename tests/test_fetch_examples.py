# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from hashlib import md5
import requests

# Import the functions to be tested
from pyrit.datasets.fetch_examples import (
    _get_cache_file_name,
    _read_cache,
    _write_cache,
    _fetch_from_repository,
    fetch_examples,
)


# Define constants for SOURCE_URL URL and file type
SOURCE_URL = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
FILE_TYPE = "json"


def test_cache_path_creation():
    cache_file_name = _get_cache_file_name(SOURCE_URL, FILE_TYPE)
    expected_hash = md5(SOURCE_URL.encode("utf-8")).hexdigest()
    assert cache_file_name == f"{expected_hash}.json"


def test_fetch_from_repository():
    response = requests.get(SOURCE_URL)
    assert response.status_code == 200
    examples = response.json()
    fetched_examples = _fetch_from_repository(SOURCE_URL, FILE_TYPE)
    assert examples == fetched_examples


def test_read_cache():
    mock_cache_file = mock_open(read_data='[{"prompt": "example"}]')
    with patch("pathlib.Path.open", mock_cache_file):
        cache_file = Path("cache_file.json")
        examples = _read_cache(cache_file, "json")
        assert examples == [{"prompt": "example"}]


def test_write_cache():
    cache_file = Path("cache_file.json")
    examples = [{"prompt": "example"}]
    mock_file = mock_open()
    with patch("pathlib.Path.open", mock_file):
        _write_cache(cache_file, examples, "json")
        # Verify that the open function was called
        mock_file.assert_called_once_with("w")
        # Verify that write was called at least once
        mock_file().write.assert_called()


def test_fetch_examples_with_cache():
    # Clear cache before running test
    data_home = Path().home() / ".pyrit_test"
    if data_home.exists():
        for file in data_home.iterdir():
            file.unlink()
        data_home.rmdir()

    # Fetch examples without cache
    examples = fetch_examples(
        SOURCE_URL, source_type="repository", file_type=FILE_TYPE, cache=False, data_home=str(data_home)
    )
    assert isinstance(examples, list) and len(examples) > 0  # Check if we got some data

    # Fetch examples with cache enabled
    examples_cached = fetch_examples(
        SOURCE_URL, source_type="repository", file_type=FILE_TYPE, cache=True, data_home=str(data_home)
    )
    assert examples_cached == examples  # Should match the previous fetched examples

    # Mock the _fetch_from_repository function to ensure it is not called again
    with patch("pyrit.datasets.fetch_examples._fetch_from_repository", MagicMock(return_value=examples)) as mock_fetch:
        examples_cached_again = fetch_examples(
            SOURCE_URL, source_type="repository", file_type=FILE_TYPE, cache=True, data_home=str(data_home)
        )
        assert examples_cached_again == examples  # Should match the previous fetched examples
        mock_fetch.assert_not_called()  # Ensure the GET request was not called again
