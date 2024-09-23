# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

from pyrit.models.storage_io import DiskStorageIO, AzureBlobStorageIO, SupportedContentType


@pytest.fixture
def azure_blob_storage_io():
    """Fixture to create an instance of AzureBlobStorageIO."""
    return AzureBlobStorageIO(container_url="dummy")


@pytest.mark.asyncio
async def test_disk_storage_io_read_file():
    storage = DiskStorageIO()
    path = "sample.txt"
    content = b"Test content"

    with patch("aiofiles.open", new_callable=MagicMock) as mock_open:
        mock_file = mock_open.return_value.__aenter__.return_value
        mock_file.read = AsyncMock(return_value=content)

        result = await storage.read_file(path)
        assert result == content
        mock_open.assert_called_once_with(path, "rb")


@pytest.mark.asyncio
async def test_disk_storage_io_write_file():
    storage = DiskStorageIO()
    path = "sample.txt"
    content = b"Test content"

    with patch("aiofiles.open", new_callable=MagicMock) as mock_open:
        mock_file = mock_open.return_value.__aenter__.return_value
        mock_file.write = AsyncMock()

        await storage.write_file(path, content)
        mock_open.assert_called_once_with(path, "wb")
        mock_file.write.assert_called_once_with(content)


@pytest.mark.asyncio
async def test_disk_storage_io_path_exists():
    storage = DiskStorageIO()
    path = "sample.txt"

    with patch("os.path.exists", return_value=True) as mock_exists:
        result = await storage.path_exists(path)
        assert result is True
        mock_exists.assert_called_once_with(path)


@pytest.mark.asyncio
async def test_disk_storage_io_is_file():
    storage = DiskStorageIO()
    path = "sample.txt"

    with patch("os.path.isfile", return_value=True) as mock_isfile:
        result = await storage.is_file(path)
        assert result is True
        mock_isfile.assert_called_once_with(path)


@pytest.mark.asyncio
async def test_disk_storage_io_create_directory_if_not_exists():
    storage = DiskStorageIO()
    directory_path = "sample_dir"

    with patch("os.makedirs") as mock_mkdir, patch("pathlib.Path.exists", return_value=False) as mock_exists:
        await storage.create_directory_if_not_exists(directory_path)
        mock_exists.assert_called_once()
        mock_mkdir.assert_called_once_with(Path(directory_path), exist_ok=True)


@pytest.mark.asyncio
async def test_azure_blob_storage_io_read_file(azure_blob_storage_io):
    azure_blob_storage_io._client_async = Mock()  # Use Mock since get_blob_client is sync

    mock_blob_client = AsyncMock()
    mock_blob_stream = AsyncMock()

    azure_blob_storage_io._client_async.get_blob_client = Mock(return_value=mock_blob_client)
    mock_blob_client.download_blob = AsyncMock(return_value=mock_blob_stream)
    mock_blob_stream.readall = AsyncMock(return_value=b"Test file content")

    result = await azure_blob_storage_io.read_file(
        "https://account.blob.core.windows.net/container/dir1/dir2/sample.png"
    )

    assert result == b"Test file content"


@pytest.mark.asyncio
async def test_azure_blob_storage_io_write_file():
    container_url = "https://youraccount.blob.core.windows.net/yourcontainer"
    azure_blob_storage_io = AzureBlobStorageIO(
        container_url=container_url, blob_content_type=SupportedContentType.PLAIN_TEXT
    )

    mock_blob_client = AsyncMock()
    mock_container_client = AsyncMock()

    mock_blob_client.upload_blob = AsyncMock()

    mock_container_client.get_blob_client.return_value = mock_blob_client

    with patch.object(azure_blob_storage_io, "_create_container_client_async", return_value=None):
        azure_blob_storage_io._client_async = mock_container_client
        azure_blob_storage_io._upload_blob_async = AsyncMock()

        data_to_write = b"Test data"
        path = "https://youraccount.blob.core.windows.net/yourcontainer/testfile.txt"

        await azure_blob_storage_io.write_file(path, data_to_write)

        azure_blob_storage_io._upload_blob_async.assert_awaited_with(
            file_name="testfile.txt", data=data_to_write, content_type=SupportedContentType.PLAIN_TEXT.value
        )


@pytest.mark.asyncio
async def test_azure_storage_io_path_exists(azure_blob_storage_io):
    azure_blob_storage_io._client_async = Mock()

    mock_blob_client = AsyncMock()

    azure_blob_storage_io._client_async.get_blob_client = Mock(return_value=mock_blob_client)
    mock_blob_client.get_blob_properties = AsyncMock()
    file_path = "https://example.blob.core.windows.net/container/dir1/dir2/blob_name.txt"
    exists = await azure_blob_storage_io.path_exists(file_path)
    assert exists is True


@pytest.mark.asyncio
async def test_azure_storage_io_is_file(azure_blob_storage_io):
    azure_blob_storage_io._client_async = Mock()

    mock_blob_client = AsyncMock()

    azure_blob_storage_io._client_async.get_blob_client = Mock(return_value=mock_blob_client)
    mock_blob_properties = Mock(size=1024)
    mock_blob_client.get_blob_properties = AsyncMock(return_value=mock_blob_properties)
    file_path = "https://example.blob.core.windows.net/container/dir1/dir2/blob_name.txt"
    is_file = await azure_blob_storage_io.is_file(file_path)
    assert is_file is True


def test_azure_storage_io_parse_blob_url_valid(azure_blob_storage_io):
    file_path = "https://example.blob.core.windows.net/container/dir1/dir2/blob_name.txt"
    container_name, blob_name = azure_blob_storage_io.parse_blob_url(file_path)

    assert container_name == "container"
    assert blob_name == "dir1/dir2/blob_name.txt"


def test_azure_storage_io_parse_blob_url_invalid(azure_blob_storage_io):
    with pytest.raises(ValueError, match="Invalid blob URL"):
        azure_blob_storage_io.parse_blob_url("invalid_url")


def test_azure_storage_io_parse_blob_url_without_scheme(azure_blob_storage_io):
    with pytest.raises(ValueError, match="Invalid blob URL"):
        azure_blob_storage_io.parse_blob_url("example.blob.core.windows.net/container/dir1/blob_name.txt")


def test_azure_storage_io_parse_blob_url_without_netloc(azure_blob_storage_io):
    with pytest.raises(ValueError, match="Invalid blob URL"):
        azure_blob_storage_io.parse_blob_url("https:///container/dir1/blob_name.txt")
