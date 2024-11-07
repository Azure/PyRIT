# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Generator
from unittest.mock import patch
import pytest
import os

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.prompt_converter import AddImageTextConverter, AddTextImageConverter
from tests.mocks import get_memory_interface

from PIL import Image, ImageFont


@pytest.fixture
def image_text_converter_sample_image():
    img = Image.new("RGB", (100, 100), color=(125, 125, 125))
    img.save("test.png")
    return "test.png"


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


def test_add_image_text_converter_initialization(image_text_converter_sample_image):
    converter = AddImageTextConverter(
        img_to_add=image_text_converter_sample_image,
        font_name="arial.ttf",
        color=(255, 255, 255),
        font_size=20,
        x_pos=10,
        y_pos=10,
    )
    assert converter._img_to_add == "test.png"
    assert converter._font_name == "arial.ttf"
    assert converter._color == (255, 255, 255)
    assert converter._font_size == 20
    assert converter._x_pos == 10
    assert converter._y_pos == 10
    assert converter._font is not None
    assert type(converter._font) is ImageFont.FreeTypeFont
    os.remove("test.png")


def test_add_image_text_converter_invalid_font(image_text_converter_sample_image):
    with pytest.raises(ValueError):
        AddImageTextConverter(
            img_to_add=image_text_converter_sample_image, font_name="arial.otf"
        )  # Invalid font extension
    os.remove("test.png")


def test_add_image_text_converter_null_img_to_add():
    with pytest.raises(ValueError):
        AddImageTextConverter(img_to_add="", font_name="arial.ttf")


def test_add_image_text_converter_fallback_to_default_font(image_text_converter_sample_image, caplog):
    AddImageTextConverter(
        img_to_add=image_text_converter_sample_image,
        font_name="nonexistent_font.ttf",
        color=(255, 255, 255),
        font_size=20,
        x_pos=10,
        y_pos=10,
    )
    assert any(
        record.levelname == "WARNING" and "Cannot open font resource" in record.message for record in caplog.records
    )
    os.remove("test.png")


def test_image_text_converter_add_text_to_image(image_text_converter_sample_image):
    converter = AddImageTextConverter(
        img_to_add=image_text_converter_sample_image, font_name="arial.ttf", color=(255, 255, 255)
    )
    image = Image.open("test.png")
    pixels_before = list(image.getdata())
    updated_image = converter._add_text_to_image("Sample Text!")
    pixels_after = list(updated_image.getdata())
    assert updated_image
    # Check if at least one pixel changed, indicating that text was added
    assert pixels_before != pixels_after
    os.remove("test.png")


@pytest.mark.asyncio
async def test_add_image_text_converter_invalid_input_text(image_text_converter_sample_image) -> None:
    converter = AddImageTextConverter(img_to_add=image_text_converter_sample_image)
    with pytest.raises(ValueError):
        assert await converter.convert_async(prompt="", input_type="text")  # type: ignore
    os.remove("test.png")


@pytest.mark.asyncio
async def test_add_image_text_converter_invalid_file_path():
    converter = AddImageTextConverter(img_to_add="nonexistent_image.png", font_name="arial.ttf")
    with pytest.raises(FileNotFoundError):
        assert await converter.convert_async(prompt="Sample Text!", input_type="text")  # type: ignore


@pytest.mark.asyncio
async def test_add_image_text_converter_convert_async(
    image_text_converter_sample_image, memory_interface: MemoryInterface
) -> None:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):

        converter = AddImageTextConverter(img_to_add=image_text_converter_sample_image)
        converted_image = await converter.convert_async(prompt="Sample Text!", input_type="text")
        assert converted_image
        assert converted_image.output_text
        assert converted_image.output_type == "image_path"
        assert os.path.exists(converted_image.output_text)
        os.remove(converted_image.output_text)
        os.remove("test.png")


def test_text_image_converter_input_supported(image_text_converter_sample_image):
    converter = AddImageTextConverter(img_to_add=image_text_converter_sample_image)
    assert converter.input_supported("image_path") is False
    assert converter.input_supported("text") is True


@pytest.mark.asyncio
async def test_add_image_text_converter_equal_to_add_text_image(
    image_text_converter_sample_image, memory_interface: MemoryInterface
) -> None:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        converter = AddImageTextConverter(img_to_add=image_text_converter_sample_image)
        converted_image = await converter.convert_async(prompt="Sample Text!", input_type="text")
        text_image_converter = AddTextImageConverter(text_to_add="Sample Text!")
        converted_text_image = await text_image_converter.convert_async(prompt="test.png", input_type="image_path")
        pixels_image_text = list(Image.open(converted_image.output_text).getdata())
        pixels_text_image = list(Image.open(converted_text_image.output_text).getdata())
        assert pixels_image_text == pixels_text_image
        os.remove(converted_image.output_text)
        os.remove("test.png")
        os.remove(converted_text_image.output_text)
