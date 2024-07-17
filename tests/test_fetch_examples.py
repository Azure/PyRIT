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


def test_cache_path_creation():
    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/59c652b/examples.json"
    file_type = "json"
    cache_file_name = _get_cache_file_name(source, file_type)
    expected_hash = md5(source.encode("utf-8")).hexdigest()
    assert cache_file_name == f"{expected_hash}.json"


def test_fetch_from_repository():
    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/59c652b/examples.json"
    file_type = "json"
    response = requests.get(source)
    assert response.status_code == 200
    examples = response.json()
    fetched_examples = _fetch_from_repository(source, file_type)
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
    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/59c652b/examples.json"
    file_type = "json"

    # Clear cache before running test
    data_home = Path().home() / ".pyrit_test"
    if data_home.exists():
        for file in data_home.iterdir():
            file.unlink()
        data_home.rmdir()

    # Fetch examples without cache
    examples = fetch_examples(
        source, source_type="repository", file_type=file_type, cache=False, data_home=str(data_home)
    )
    assert isinstance(examples, list) and len(examples) > 0  # Check if we got some data

    # Fetch examples with cache enabled
    examples_cached = fetch_examples(
        source, source_type="repository", file_type=file_type, cache=True, data_home=str(data_home)
    )
    assert examples_cached == examples  # Should match the previous fetched examples

    # Mock the _fetch_from_repository function to ensure it is not called again
    with patch("pyrit.datasets.fetch_examples._fetch_from_repository", MagicMock(return_value=examples)) as mock_fetch:
        examples_cached_again = fetch_examples(
            source, source_type="repository", file_type=file_type, cache=True, data_home=str(data_home)
        )
        assert examples_cached_again == examples  # Should match the previous fetched examples
        mock_fetch.assert_not_called()  # Ensure the GET request was not called again
