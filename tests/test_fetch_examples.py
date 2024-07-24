# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path
from hashlib import md5
import json

# Import the functions to be tested
from pyrit.datasets.fetch_examples import (
    _get_cache_file_name,
    _read_cache,
    _write_cache,
    _fetch_from_public_url,
    fetch_examples,
)

# Constants used for testing
SOURCE_URL = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
# The source URL points to an example JSON file used for testing purposes.
FILE_TYPE = "json"  # Specifies the type of file being tested.


def test_cache_path_creation():
    """
    Test that the cache file name is created correctly from the source URL and file type.
    """
    cache_file_name = _get_cache_file_name(SOURCE_URL, FILE_TYPE)
    expected_hash = md5(SOURCE_URL.encode("utf-8")).hexdigest()
    assert cache_file_name == f"{expected_hash}.json"


def test_fetch_from_public_url():
    """
    Test fetching data from a public URL.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"prompt": "example"}]
    mock_response.text = json.dumps([{"prompt": "example"}])  # Ensure it's a valid JSON string

    # Mock the requests.get function to return the mock response
    with patch("requests.get", return_value=mock_response):
        fetched_examples = _fetch_from_public_url(SOURCE_URL, FILE_TYPE)
        assert fetched_examples == [{"prompt": "example"}]


def test_read_cache():
    """
    Test reading data from a cache file.
    """
    mock_cache_file = mock_open(read_data='[{"prompt": "example"}]')

    with patch("pathlib.Path.open", mock_cache_file):
        cache_file = Path("cache_file.json")
        examples = _read_cache(cache_file, "json")
        assert examples == [{"prompt": "example"}]


def test_write_cache():
    """
    Test writing data to a cache file.
    """
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
    """
    Test fetching examples with caching enabled and disabled.
    """

    # Clear cache before running test
    data_home = Path().home() / ".pyrit_test" / "datasets"
    if data_home.exists():
        for file in data_home.iterdir():
            file.unlink()
        data_home.rmdir()

    # Fetch examples without cache
    examples = fetch_examples(source=SOURCE_URL, source_type="public_url", cache=False, data_home=str(data_home))
    assert isinstance(examples, list) and len(examples) > 0  # Check if we got some data

    # Fetch examples with cache enabled
    examples_cached = fetch_examples(source=SOURCE_URL, source_type="public_url", cache=True, data_home=str(data_home))
    assert examples_cached == examples  # Should match the previous fetched examples

    # Mock the _fetch_from_public_url function to ensure it is not called again
    with patch("pyrit.datasets.fetch_examples._fetch_from_public_url", MagicMock(return_value=examples)) as mock_fetch:
        examples_cached_again = fetch_examples(
            source=SOURCE_URL, source_type="public_url", cache=True, data_home=str(data_home)
        )
        assert examples_cached_again == examples  # Should match the previous fetched examples
        mock_fetch.assert_not_called()  # Ensure the GET request was not called again
