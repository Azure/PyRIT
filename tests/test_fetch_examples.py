# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path
from hashlib import md5
import json
from pyrit.common.path import RESULTS_PATH
import pytest


# Import the functions to be tested
from pyrit.datasets.fetch_example_datasets import (
    _get_cache_file_name,
    _read_cache,
    _write_cache,
    _fetch_from_public_url,
    fetch_examples,
)


# These URLs are placeholders and will be mocked in tests
SOURCE_URLS = {
    "json": "https://example.com/examples.json",
    "csv": "https://example.com/examples.csv",
    "txt": "https://example.com/examples.txt",
}


FILE_TYPES = ["json", "csv", "txt"]
UNSUPPORTED_FILE_TYPES = ["xml", "pdf", "docx"]  # Unsupported file types for testing


@pytest.mark.parametrize("file_type,url", [(ft, SOURCE_URLS[ft]) for ft in FILE_TYPES])
def test_cache_path_creation(file_type, url):
    """
    Test that the cache file name is created correctly from the source URL and file type.
    """
    cache_file_name = _get_cache_file_name(url, file_type)
    expected_hash = md5(url.encode("utf-8")).hexdigest()
    assert cache_file_name == f"{expected_hash}.{file_type}"


@pytest.mark.parametrize(
    "file_type,url,content,expected",
    [
        ("json", SOURCE_URLS["json"], json.dumps([{"prompt": "example"}]), [{"prompt": "example"}]),
        ("csv", SOURCE_URLS["csv"], "prompt\nexample", [{"prompt": "example"}]),
        ("txt", SOURCE_URLS["txt"], "example\n", [{"prompt": "example"}]),
    ],
)
def test_fetch_from_public_url(file_type, url, content, expected):
    """
    Test fetching data from a public URL for different file types.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = content

    # Mock the requests.get function to return the mock response
    with patch("requests.get", return_value=mock_response):
        fetched_examples = _fetch_from_public_url(url, file_type)
        assert fetched_examples == expected


@pytest.mark.parametrize(
    "file_type,content,expected",
    [
        ("json", '[{"prompt": "example"}]', [{"prompt": "example"}]),
        ("csv", "prompt\nexample\n", [{"prompt": "example"}]),
        ("txt", "example\n", [{"prompt": "example"}]),
    ],
)
def test_read_cache(file_type, content, expected):
    """
    Test reading data from a cache file for different file types.
    """
    mock_cache_file = mock_open(read_data=content)

    with patch("pathlib.Path.open", mock_cache_file):
        cache_file = Path(f"cache_file.{file_type}")
        examples = _read_cache(cache_file, file_type)
        assert examples == expected


@pytest.mark.parametrize(
    "file_type,examples",
    [
        ("json", [{"prompt": "example"}]),
        ("csv", [{"prompt": "example"}]),
        ("txt", [{"prompt": "example"}]),
    ],
)
def test_write_cache(file_type, examples):
    """
    Test writing data to a cache file for different file types.
    """
    cache_file = Path(f"cache_file.{file_type}")
    mock_file = mock_open()

    with patch("pathlib.Path.open", mock_file):
        _write_cache(cache_file, examples, file_type)
        # Verify that the open function was called
        mock_file.assert_called_once_with("w", encoding="utf-8")
        # Verify that write was called at least once
        mock_file().write.assert_called()


@pytest.mark.parametrize("file_type", UNSUPPORTED_FILE_TYPES)
def test_fetch_from_public_url_unsupported(file_type):
    """
    Test fetching data from a public URL for unsupported file types.
    """
    url = f"https://example.com/examples.{file_type}"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "example content"

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Invalid file_type. Expected one of: json, csv, txt."):
            _fetch_from_public_url(url, file_type)


@pytest.mark.parametrize("file_type", UNSUPPORTED_FILE_TYPES)
def test_read_cache_unsupported(file_type):
    """
    Test reading data from a cache file for unsupported file types.
    """
    cache_file_dir = RESULTS_PATH / ".pyrit_test" / "datasets"
    cache_file = cache_file_dir / f"cache_file.{file_type}"

    # Ensure the file exists before testing
    cache_file_dir.mkdir(parents=True, exist_ok=True)
    cache_file.touch()

    with pytest.raises(ValueError, match="Invalid file_type. Expected one of: json, csv, txt."):
        _read_cache(cache_file, file_type)

    # Cleanup the created file after the test
    cache_file.unlink()


@pytest.mark.parametrize("file_type", UNSUPPORTED_FILE_TYPES)
def test_write_cache_unsupported(file_type):
    """
    Test writing data to a cache file for unsupported file types.
    """
    cache_file = RESULTS_PATH / ".pyrit_test" / "datasets" / f"cache_file.{file_type}"
    examples = [{"prompt": "example"}]

    with pytest.raises(ValueError, match="Invalid file_type. Expected one of: json, csv, txt."):
        _write_cache(cache_file, examples, file_type)


def test_fetch_examples_with_cache():
    """
    Test fetching examples with caching enabled and disabled.
    """

    # Clear cache before running test
    data_home = RESULTS_PATH / ".pyrit_test" / "datasets"
    if data_home.exists():
        for file in data_home.iterdir():
            file.unlink()
        data_home.rmdir()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"prompt": "example"}]
    mock_response.text = json.dumps([{"prompt": "example"}])

    with patch("requests.get", return_value=mock_response):
        # Fetch examples without cache
        examples = fetch_examples(
            source=SOURCE_URLS["json"], source_type="public_url", cache=False, data_home=str(data_home)
        )
        assert isinstance(examples, list) and len(examples) > 0  # Check if we got some data

        # Fetch examples with cache enabled
        examples_cached = fetch_examples(
            source=SOURCE_URLS["json"], source_type="public_url", cache=True, data_home=str(data_home)
        )
        assert examples_cached == examples  # Should match the previous fetched examples

        # Mock the _fetch_from_public_url function to ensure it is not called again
        with patch(
            "pyrit.datasets.fetch_example_datasets._fetch_from_public_url", MagicMock(return_value=examples)
        ) as mock_fetch:
            examples_cached_again = fetch_examples(
                source=SOURCE_URLS["json"], source_type="public_url", cache=True, data_home=str(data_home)
            )
            assert examples_cached_again == examples  # Should match the previous fetched examples
            mock_fetch.assert_not_called()  # Ensure the GET request was not called again
