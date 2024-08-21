# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.prompt_converter import QRCodeConverter


def test_qr_code_converter_initialization():
    converter = QRCodeConverter(
        scale=5,
        border=4,
        dark_color=(0, 0, 0),
        light_color=(255, 255, 255),
        data_dark_color=(0, 0, 0),
        data_light_color=(255, 255, 255),
        finder_dark_color=(0, 0, 0),
        finder_light_color=(255, 255, 255),
        border_color=(255, 255, 255),
        output_filename="sample_file.png",
    )
    assert converter._scale == 5
    assert converter._border == 4
    assert converter._dark_color == (0, 0, 0)
    assert converter._light_color == (255, 255, 255)
    assert converter._data_dark_color == (0, 0, 0)
    assert converter._data_light_color == (255, 255, 255)
    assert converter._finder_dark_color == (0, 0, 0)
    assert converter._finder_light_color == (255, 255, 255)
    assert converter._border_color == (255, 255, 255)
    assert converter._output_filename == "sample_file.png"


def test_qr_code_converter_color_initialization():
    converter = QRCodeConverter(dark_color=(2, 0, 2), light_color=(100, 150, 100))
    assert converter._dark_color == (2, 0, 2)
    assert converter._light_color == (100, 150, 100)
    assert converter._data_dark_color == converter._dark_color
    assert converter._data_light_color == converter._light_color
    assert converter._finder_dark_color == converter._dark_color
    assert converter._finder_light_color == converter._light_color
    assert converter._border_color == converter._light_color


def test_qr_code_converter_invalid_output_filename():
    with pytest.raises(ValueError):
        QRCodeConverter(output_filename="sample_file.jpg")


@pytest.mark.asyncio
async def test_qr_code_converter_invalid_prompt() -> None:
    converter = QRCodeConverter(output_filename="sample_file.png")
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="", input_type="text")


@pytest.mark.asyncio
async def test_qr_code_converter_convert_async() -> None:
    converter = QRCodeConverter(output_filename="sample_file.png")
    qr = await converter.convert_async(prompt="Sample prompt", input_type="text")
    assert qr
    assert qr.output_text == "sample_file.png"
    assert qr.output_type == "image_path"
    assert os.path.exists(qr.output_text)
    os.remove(qr.output_text)


@pytest.mark.asyncio
async def test_qr_code_converter_default_output_filename() -> None:
    converter = QRCodeConverter(scale=5, border=4, dark_color=(0, 0, 0), light_color=(255, 255, 255))
    qr = await converter.convert_async(prompt="Sample prompt", input_type="text")
    assert qr
    assert qr.output_text
    assert qr.output_type == "image_path"
    assert os.path.exists(qr.output_text)
    os.remove(qr.output_text)


def test_text_image_converter_input_supported():
    converter = QRCodeConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image_path") is False
