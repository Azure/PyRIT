# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.prompt_normalizer import DataTypeNormalizer, data_normalizer_factory
from pyrit.prompt_normalizer.data_type_normalizer import ImagePathDataTypeNormalizer, TextDataTypeNormalizer


def test_data_normalizer_factory_text_no_data_throws():
    with pytest.raises(TypeError):
        data_normalizer_factory("text")


def test_data_normalizer_factory_text_with_data():
    normalizer = data_normalizer_factory(data_type="text", prompt_text="test")
    assert isinstance(normalizer, DataTypeNormalizer)
    assert isinstance(normalizer, TextDataTypeNormalizer)
    assert normalizer.data_type == "text"
    assert normalizer.prompt_text == "test"
    assert normalizer.data_on_disk() is False


def test_data_normalizer_text_read_data_throws():
    normalizer = data_normalizer_factory(data_type="text", prompt_text="test")
    with pytest.raises(TypeError):
        normalizer.read_data()


def test_data_normalizer_text_save_data_throws():
    normalizer = data_normalizer_factory(data_type="text", prompt_text="test")
    with pytest.raises(TypeError):
        normalizer.save_data(b"\x00")


def test_image_path_normalizer_factory_prompt_text_raises():
    with pytest.raises(FileNotFoundError):
        data_normalizer_factory(data_type="image_path", prompt_text="no_real_path.txt")


def test_image_path_normalizer_factory():
    normalizer = data_normalizer_factory(data_type="image_path")
    assert isinstance(normalizer, DataTypeNormalizer)
    assert isinstance(normalizer, ImagePathDataTypeNormalizer)
    assert normalizer.data_type == "image_path"
    assert normalizer.data_on_disk()


def test_image_path_save_data():
    normalizer = data_normalizer_factory(data_type="image_path")
    normalizer.save_data(b"\x00")
    assert normalizer.prompt_text
    assert normalizer.prompt_text.endswith(".png")
    assert os.path.isabs(normalizer.prompt_text)
    assert os.path.exists(normalizer.prompt_text)
    assert os.path.isfile(normalizer.prompt_text)


def test_image_path_read_data():
    data = b"\x00\x11\x22\x33"

    normalizer = data_normalizer_factory(data_type="image_path")
    normalizer.save_data(data)
    assert normalizer.read_data() == data
    read_normalizer = data_normalizer_factory(data_type="image_path", prompt_text=normalizer.prompt_text)
    assert read_normalizer.read_data() == data
