# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
from PIL import Image
import os
import tempfile
from unittest.mock import AsyncMock, patch
import pytest

from pyrit.memory import DuckDBMemory, CentralMemory
from pyrit.models import (
    ImagePathDataTypeSerializer,
    TextDataTypeSerializer,
    ErrorDataTypeSerializer,
    DataTypeSerializer,
    data_serializer_factory,
)


@pytest.fixture(scope="function")
def set_duckdb_in_memory():
    duckdb_in_memory = DuckDBMemory(db_path=":memory:")
    CentralMemory.set_memory_instance(duckdb_in_memory)


def test_data_serializer_factory_text_no_data_throws(set_duckdb_in_memory):
    with pytest.raises(ValueError):
        data_serializer_factory(data_type="text")


def test_data_serializer_factory_text_with_data(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="text", value="test")
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, TextDataTypeSerializer)
    assert serializer.data_type == "text"
    assert serializer.value == "test"
    assert serializer.data_on_disk() is False


def test_data_serializer_factory_error_with_data(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="error", value="test")
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, ErrorDataTypeSerializer)
    assert serializer.data_type == "error"
    assert serializer.value == "test"
    assert serializer.data_on_disk() is False


@pytest.mark.asyncio
async def test_data_serializer_text_read_data_throws(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="text", value="test")
    with pytest.raises(TypeError):
        await serializer.read_data()


@pytest.mark.asyncio
async def test_data_serializer_text_save_data_throws(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="text", value="test")
    with pytest.raises(TypeError):
        await serializer.save_data(b"\x00")


@pytest.mark.asyncio
async def test_data_serializer_error_read_data_throws(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="error", value="test")
    with pytest.raises(TypeError):
        await serializer.read_data()


@pytest.mark.asyncio
async def test_data_serializer_error_save_data_throws(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="error", value="test")
    with pytest.raises(TypeError):
        await serializer.save_data(b"\x00")


def test_image_path_normalizer_factory(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="image_path")
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, ImagePathDataTypeSerializer)
    assert serializer.data_type == "image_path"
    assert serializer.data_on_disk()


@pytest.mark.asyncio
async def test_image_path_save_data(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="image_path")
    await serializer.save_data(b"\x00")
    serializer_value = serializer.value
    assert serializer_value
    assert serializer_value.endswith(".png")
    assert os.path.isabs(serializer_value)
    assert os.path.exists(serializer_value)
    assert os.path.isfile(serializer_value)


@pytest.mark.asyncio
async def test_image_path_read_data(set_duckdb_in_memory):
    data = b"\x00\x11\x22\x33"
    normalizer = data_serializer_factory(data_type="image_path")
    await normalizer.save_data(data)
    assert await normalizer.read_data() == data
    read_normalizer = data_serializer_factory(data_type="image_path", value=normalizer.value)
    assert await read_normalizer.read_data() == data


@pytest.mark.asyncio
async def test_image_path_read_data_base64(set_duckdb_in_memory):
    data = b"AAAA"

    normalizer = data_serializer_factory(data_type="image_path")
    await normalizer.save_data(data)
    base_64_data = await normalizer.read_data_base64()
    assert base_64_data
    assert base_64_data == "QUFBQQ=="


@pytest.mark.asyncio()
async def test_path_not_exists(set_duckdb_in_memory):
    file_path = "non_existing_file.txt"
    serializer = data_serializer_factory(data_type="image_path", value=file_path)

    with pytest.raises(FileNotFoundError):
        await serializer.read_data()


def test_get_extension(set_duckdb_in_memory):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        expected_extension = ".jpg"
        extension = DataTypeSerializer.get_extension(temp_file_path)
        assert extension == expected_extension


def test_get_mime_type(set_duckdb_in_memory):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        expected_mime_type = "image/jpeg"
        mime_type = DataTypeSerializer.get_mime_type(temp_file_path)
        assert mime_type == expected_mime_type


@pytest.mark.asyncio
async def test_save_b64_image(set_duckdb_in_memory):
    serializer = data_serializer_factory(data_type="image_path")
    await serializer.save_b64_image("\x00")
    serializer_value = str(serializer.value)
    assert serializer_value
    assert serializer_value.endswith(".png")
    assert os.path.isabs(serializer_value)
    assert os.path.exists(serializer_value)
    assert os.path.isfile(serializer_value)


@pytest.mark.asyncio
async def test_audio_path_save_data(set_duckdb_in_memory):
    """Test saving audio data to disk."""
    serializer = data_serializer_factory(data_type="audio_path")
    await serializer.save_data(b"audio_data")
    assert serializer.value.endswith(".mp3")
    assert os.path.exists(serializer.value)
    assert os.path.isfile(serializer.value)


@pytest.mark.asyncio
async def test_audio_path_read_data(set_duckdb_in_memory):
    """Test reading audio data from disk."""
    data = b"audio_content"
    serializer = data_serializer_factory(data_type="audio_path")
    await serializer.save_data(data)
    read_data = await serializer.read_data()
    assert read_data == data


@pytest.mark.asyncio
async def test_get_sha256_from_text(set_duckdb_in_memory):
    """Test SHA256 hash calculation for text data."""
    serializer = data_serializer_factory(data_type="text", value="test_string")
    sha256_hash = await serializer.get_sha256()
    expected_hash = hashlib.sha256(b"test_string").hexdigest()
    assert sha256_hash == expected_hash


@pytest.mark.asyncio
async def test_get_sha256_from_image_file(set_duckdb_in_memory):
    """Test SHA256 hash calculation for file data."""
    data = b"file_content.png"
    serializer = data_serializer_factory(data_type="image_path")
    await serializer.save_data(data)
    sha256_hash = await serializer.get_sha256()
    expected_hash = hashlib.sha256(data).hexdigest()
    assert sha256_hash == expected_hash


def test_is_azure_storage_url(set_duckdb_in_memory):
    """Test Azure Storage URL validation."""
    valid_url = "https://mystorageaccount.blob.core.windows.net/container/file.txt"
    invalid_url = "https://example.com/file.txt"

    serializer = data_serializer_factory(data_type="url", value=valid_url)
    assert serializer._is_azure_storage_url(valid_url) is True
    assert serializer._is_azure_storage_url(invalid_url) is False


@pytest.mark.asyncio
async def test_read_data_local_file_with_dummy_image(set_duckdb_in_memory):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
        image_path = temp_image_file.name
        image = Image.new("RGB", (10, 10), color="red")
        image.save(image_path)

    try:
        mock_storage_io = AsyncMock()
        mock_storage_io.path_exists.return_value = True
        with open(image_path, "rb") as f:
            mock_storage_io.read_file.return_value = f.read()

        with patch("pyrit.models.data_type_serializer.DiskStorageIO", return_value=mock_storage_io):
            serializer = data_serializer_factory(data_type="image_path", value=image_path)

            data = await serializer.read_data()

            with open(image_path, "rb") as f:
                expected_data = f.read()
            assert data == expected_data

            mock_storage_io.path_exists.assert_awaited_once_with(path=image_path)
            mock_storage_io.read_file.assert_awaited_once_with(image_path)
    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)
