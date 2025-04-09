# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import os
import re
import tempfile
from typing import get_args
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from pyrit.models import (
    AllowedCategories,
    DataTypeSerializer,
    ErrorDataTypeSerializer,
    ImagePathDataTypeSerializer,
    TextDataTypeSerializer,
    data_serializer_factory,
)


def test_allowed_categories():
    entries = get_args(AllowedCategories)
    assert len(entries) == 2
    assert entries[0] == "seed-prompt-entries"
    assert entries[1] == "prompt-memory-entries"


def test_data_serializer_factory_text_no_data_throws(duckdb_instance):
    with pytest.raises(ValueError):
        data_serializer_factory(category="prompt-memory-entries", data_type="text")


def test_data_serializer_factory_text_with_data(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="text", value="test")
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, TextDataTypeSerializer)
    assert serializer.data_type == "text"
    assert serializer.value == "test"
    assert serializer.data_on_disk() is False


def test_data_serializer_factory_error_with_data(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="error", value="test")
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, ErrorDataTypeSerializer)
    assert serializer.data_type == "error"
    assert serializer.value == "test"
    assert serializer.data_on_disk() is False


@pytest.mark.asyncio
async def test_data_serializer_text_read_data_throws(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="text", value="test")
    with pytest.raises(TypeError):
        await serializer.read_data()


@pytest.mark.asyncio
async def test_data_serializer_text_save_data_throws(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="text", value="test")
    with pytest.raises(TypeError):
        await serializer.save_data(b"\x00")


@pytest.mark.asyncio
async def test_data_serializer_error_read_data_throws(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="error", value="test")
    with pytest.raises(TypeError):
        await serializer.read_data()


@pytest.mark.asyncio
async def test_data_serializer_error_save_data_throws(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="error", value="test")
    with pytest.raises(TypeError):
        await serializer.save_data(b"\x00")


@pytest.mark.asyncio
async def test_data_serializer_factory_missing_category_raises_value_error():
    expected_error_message = (
        "The 'category' argument is mandatory and must be one of the following: "
        "('seed-prompt-entries', 'prompt-memory-entries')."
    )

    escaped_message = re.escape(expected_error_message)
    with pytest.raises(ValueError, match=escaped_message):
        await data_serializer_factory(data_type="text", value="test", category=None)


def test_image_path_normalizer_factory(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, ImagePathDataTypeSerializer)
    assert serializer.data_type == "image_path"
    assert serializer.data_on_disk()


@pytest.mark.asyncio
async def test_image_path_save_data(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    await serializer.save_data(b"\x00")
    serializer_value = serializer.value
    assert serializer_value
    assert serializer_value.endswith(".png")
    assert os.path.isabs(serializer_value)
    assert os.path.exists(serializer_value)
    assert os.path.isfile(serializer_value)


@pytest.mark.asyncio
async def test_image_path_read_data(duckdb_instance):
    data = b"\x00\x11\x22\x33"
    normalizer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    await normalizer.save_data(data)
    assert await normalizer.read_data() == data
    read_normalizer = data_serializer_factory(
        category="prompt-memory-entries", data_type="image_path", value=normalizer.value
    )
    assert await read_normalizer.read_data() == data


@pytest.mark.asyncio
async def test_image_path_read_data_base64(duckdb_instance):
    data = b"AAAA"

    normalizer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    await normalizer.save_data(data)
    base_64_data = await normalizer.read_data_base64()
    assert base_64_data
    assert base_64_data == "QUFBQQ=="


@pytest.mark.asyncio
async def test_path_not_exists(duckdb_instance):
    file_path = "non_existing_file.txt"
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path", value=file_path)

    with pytest.raises(FileNotFoundError):
        await serializer.read_data()


def test_get_extension(duckdb_instance):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        expected_extension = ".jpg"
        extension = DataTypeSerializer.get_extension(temp_file_path)
        assert extension == expected_extension


def test_get_mime_type(duckdb_instance):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        expected_mime_type = "image/jpeg"
        mime_type = DataTypeSerializer.get_mime_type(temp_file_path)
        assert mime_type == expected_mime_type


@pytest.mark.asyncio
async def test_save_b64_image(duckdb_instance):
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    await serializer.save_b64_image("\x00")
    serializer_value = str(serializer.value)
    assert serializer_value
    assert serializer_value.endswith(".png")
    assert os.path.isabs(serializer_value)
    assert os.path.exists(serializer_value)
    assert os.path.isfile(serializer_value)


@pytest.mark.asyncio
async def test_audio_path_save_data(duckdb_instance):
    """Test saving audio data to disk."""
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="audio_path")
    await serializer.save_data(b"audio_data")
    assert serializer.value.endswith(".mp3")
    assert os.path.exists(serializer.value)
    assert os.path.isfile(serializer.value)


@pytest.mark.asyncio
async def test_audio_path_read_data(duckdb_instance):
    """Test reading audio data from disk."""
    data = b"audio_content"
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="audio_path")
    await serializer.save_data(data)
    read_data = await serializer.read_data()
    assert read_data == data


@pytest.mark.asyncio
async def test_video_path_save_data(duckdb_instance):
    """Test saving video data to disk."""
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="video_path")
    video_data = b"video_data"
    await serializer.save_data(video_data)
    assert serializer.value.endswith(".mp4")  # Assuming the default extension is '.mp4'
    assert os.path.exists(serializer.value)
    assert os.path.isfile(serializer.value)


@pytest.mark.asyncio
async def test_video_path_read_data(duckdb_instance):
    """Test reading video data from disk."""
    video_data = b"video_content"
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="video_path")
    await serializer.save_data(video_data)
    read_data = await serializer.read_data()
    assert read_data == video_data


@pytest.mark.asyncio
async def test_video_path_save_with_custom_extension(duckdb_instance):
    """Test saving video data with a custom file extension."""
    custom_extension = "avi"
    serializer = data_serializer_factory(
        category="prompt-memory-entries", data_type="video_path", extension=custom_extension
    )
    video_data = b"video_data"
    await serializer.save_data(video_data)
    assert serializer.value.endswith(f".{custom_extension}")
    assert os.path.exists(serializer.value)
    assert os.path.isfile(serializer.value)


@pytest.mark.asyncio
async def test_get_sha256_from_text(duckdb_instance):
    """Test SHA256 hash calculation for text data."""
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="text", value="test_string")
    sha256_hash = await serializer.get_sha256()
    expected_hash = hashlib.sha256(b"test_string").hexdigest()
    assert sha256_hash == expected_hash


@pytest.mark.asyncio
async def test_get_sha256_from_image_file(duckdb_instance):
    """Test SHA256 hash calculation for file data."""
    data = b"file_content.png"
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    await serializer.save_data(data)
    sha256_hash = await serializer.get_sha256()
    expected_hash = hashlib.sha256(data).hexdigest()
    assert sha256_hash == expected_hash


def test_is_azure_storage_url(duckdb_instance):
    """Test Azure Storage URL validation."""
    valid_url = "https://mystorageaccount.blob.core.windows.net/container/file.txt"
    invalid_url = "https://example.com/file.txt"

    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="url", value=valid_url)
    assert serializer._is_azure_storage_url(valid_url) is True
    assert serializer._is_azure_storage_url(invalid_url) is False


@pytest.mark.asyncio
async def test_read_data_local_file_with_dummy_image(duckdb_instance):
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
            serializer = data_serializer_factory(
                category="prompt-memory-entries", data_type="image_path", value=image_path
            )

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


@pytest.mark.asyncio
async def test_get_data_filename(duckdb_instance):
    """Test get_data_filename when a file_name is provided."""
    serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
    provided_filename = "custom_image_name"
    filename = await serializer.get_data_filename(file_name=provided_filename)
    assert str(filename).endswith(f"{provided_filename}.{serializer.file_extension}")
    assert os.path.isabs(filename)
    assert os.path.exists(os.path.dirname(filename))
    assert not os.path.exists(filename)  # File should not exist yet
