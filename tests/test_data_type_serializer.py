# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import pytest

from pyrit.prompt_normalizer import DataTypeSerializer, data_serializer_factory
from pyrit.prompt_normalizer.data_type_serializer import ImagePathDataTypeSerializer, TextDataTypeSerializer


def test_data_serializer_factory_text_no_data_throws():
    with pytest.raises(TypeError):
        data_serializer_factory("text")


def test_data_serializer_factory_text_with_data():
    normalizer = data_serializer_factory(data_type="text", prompt_text="test")
    assert isinstance(normalizer, DataTypeSerializer)
    assert isinstance(normalizer, TextDataTypeSerializer)
    assert normalizer.data_type == "text"
    assert normalizer.prompt_text == "test"
    assert normalizer.data_on_disk() is False


def test_data_serializer_text_read_data_throws():
    normalizer = data_serializer_factory(data_type="text", prompt_text="test")
    with pytest.raises(TypeError):
        normalizer.read_data()


def test_data_serializer_text_save_data_throws():
    normalizer = data_serializer_factory(data_type="text", prompt_text="test")
    with pytest.raises(TypeError):
        normalizer.save_data(b"\x00")


def test_image_path_normalizer_factory_prompt_text_raises():
    with pytest.raises(FileNotFoundError):
        data_serializer_factory(data_type="image_path", prompt_text="no_real_path.txt")


def test_image_path_normalizer_factory():
    normalizer = data_serializer_factory(data_type="image_path")
    assert isinstance(normalizer, DataTypeSerializer)
    assert isinstance(normalizer, ImagePathDataTypeSerializer)
    assert normalizer.data_type == "image_path"
    assert normalizer.data_on_disk()


def test_image_path_save_data():
    normalizer = data_serializer_factory(data_type="image_path")
    normalizer.save_data(b"\x00")
    assert normalizer.prompt_text
    assert normalizer.prompt_text.endswith(".png")
    assert os.path.isabs(normalizer.prompt_text)
    assert os.path.exists(normalizer.prompt_text)
    assert os.path.isfile(normalizer.prompt_text)


def test_image_path_read_data():
    data = b"\x00\x11\x22\x33"

    normalizer = data_serializer_factory(data_type="image_path")
    normalizer.save_data(data)
    assert normalizer.read_data() == data
    read_normalizer = data_serializer_factory(data_type="image_path", prompt_text=normalizer.prompt_text)
    assert read_normalizer.read_data() == data


def test_image_path_read_data_base64():
    data = b"AAAA"

    normalizer = data_serializer_factory(data_type="image_path")
    normalizer.save_data(data)
    base_64_data = normalizer.read_data_base64()
    assert base_64_data
    assert base_64_data == "QUFBQQ=="


def test_path_exists():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        assert DataTypeSerializer.path_exists(temp_file_path) is True


def test_path_not_exists():
    file_path = "non_existing_file.txt"
    with pytest.raises(FileNotFoundError):
        DataTypeSerializer.path_exists(file_path)


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

def test_save_b64_image():
    normalizer = data_serializer_factory(data_type="image_path")
    normalizer.save_b64_image("\x00")
    assert normalizer.prompt_text
    assert normalizer.prompt_text.endswith(".png")
    assert os.path.isabs(normalizer.prompt_text)
    assert os.path.exists(normalizer.prompt_text)
    assert os.path.isfile(normalizer.prompt_text)