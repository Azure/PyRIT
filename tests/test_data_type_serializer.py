# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import pytest

from pyrit.memory.duckdb_memory import DuckDBMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import (
    ImagePathDataTypeSerializer,
    TextDataTypeSerializer,
    ErrorDataTypeSerializer,
    DataTypeSerializer,
    data_serializer_factory,
)


@pytest.fixture(scope="function")
def duckdb_in_memory() -> MemoryInterface:
    file_memory = DuckDBMemory(db_path=":memory:")
    return file_memory


def test_data_serializer_factory_text_no_data_throws(duckdb_in_memory: MemoryInterface):
    with pytest.raises(ValueError):
        data_serializer_factory(data_type="text", memory=duckdb_in_memory)


def test_data_serializer_factory_text_with_data(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="text", value="test", memory=duckdb_in_memory)
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, TextDataTypeSerializer)
    assert serializer.data_type == "text"
    assert serializer.value == "test"
    assert serializer.data_on_disk() is False


def test_data_serializer_factory_error_with_data(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="error", value="test", memory=duckdb_in_memory)
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, ErrorDataTypeSerializer)
    assert serializer.data_type == "error"
    assert serializer.value == "test"
    assert serializer.data_on_disk() is False


@pytest.mark.asyncio
async def test_data_serializer_text_read_data_throws(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="text", value="test", memory=duckdb_in_memory)
    with pytest.raises(TypeError):
        await serializer.read_data()


@pytest.mark.asyncio
async def test_data_serializer_text_save_data_throws(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="text", value="test", memory=duckdb_in_memory)
    with pytest.raises(TypeError):
        await serializer.save_data(b"\x00")


@pytest.mark.asyncio
async def test_data_serializer_error_read_data_throws(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="error", value="test", memory=duckdb_in_memory)
    with pytest.raises(TypeError):
        await serializer.read_data()


@pytest.mark.asyncio
async def test_data_serializer_error_save_data_throws(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="error", value="test", memory=duckdb_in_memory)
    with pytest.raises(TypeError):
        await serializer.save_data(b"\x00")


def test_image_path_normalizer_factory(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="image_path", memory=duckdb_in_memory)
    assert isinstance(serializer, DataTypeSerializer)
    assert isinstance(serializer, ImagePathDataTypeSerializer)
    assert serializer.data_type == "image_path"
    assert serializer.data_on_disk()


@pytest.mark.asyncio
async def test_image_path_save_data(duckdb_in_memory: MemoryInterface):
    normalizer = data_serializer_factory(data_type="image_path", memory=duckdb_in_memory)
    await normalizer.save_data(b"\x00")
    assert normalizer.value
    assert normalizer.value.endswith(".png")
    assert os.path.isabs(normalizer.value)
    assert os.path.exists(normalizer.value)
    assert os.path.isfile(normalizer.value)


@pytest.mark.asyncio
async def test_image_path_read_data(duckdb_in_memory: MemoryInterface):
    data = b"\x00\x11\x22\x33"
    normalizer = data_serializer_factory(data_type="image_path", memory=duckdb_in_memory)
    await normalizer.save_data(data)
    assert await normalizer.read_data() == data
    read_normalizer = data_serializer_factory(data_type="image_path", value=normalizer.value, memory=duckdb_in_memory)
    assert await read_normalizer.read_data() == data


@pytest.mark.asyncio
async def test_image_path_read_data_base64(duckdb_in_memory: MemoryInterface):
    data = b"AAAA"

    normalizer = data_serializer_factory(data_type="image_path", memory=duckdb_in_memory)
    await normalizer.save_data(data)
    base_64_data = await normalizer.read_data_base64()
    assert base_64_data
    assert base_64_data == "QUFBQQ=="


@pytest.mark.asyncio()
async def test_path_not_exists(duckdb_in_memory: MemoryInterface):
    file_path = "non_existing_file.txt"
    serializer = data_serializer_factory(data_type="image_path", value=file_path, memory=duckdb_in_memory)

    with pytest.raises(FileNotFoundError):
        await serializer.read_data()


def test_get_extension():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        expected_extension = ".jpg"
        extension = DataTypeSerializer.get_extension(temp_file_path)
        assert extension == expected_extension


def test_get_mime_type():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        expected_mime_type = "image/jpeg"
        mime_type = DataTypeSerializer.get_mime_type(temp_file_path)
        assert mime_type == expected_mime_type


@pytest.mark.asyncio
async def test_save_b64_image(duckdb_in_memory: MemoryInterface):
    serializer = data_serializer_factory(data_type="image_path", memory=duckdb_in_memory)
    await serializer.save_b64_image("\x00")
    assert serializer.value
    assert serializer.value.endswith(".png")
    assert os.path.isabs(serializer.value)
    assert os.path.exists(serializer.value)
    assert os.path.isfile(serializer.value)
