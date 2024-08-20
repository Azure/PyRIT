# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.prompt_converter import QRCodeConverter


def test_qr_code_converter_initialization():
    converter = QRCodeConverter(
        version=None,
        fit=True,
        box_size=10,
        border=4,
        back_color=(255, 255, 255),
        fill_color=(0, 0, 0),
        output_filename="sample_file.png",
    )
    assert not converter._version
    assert converter._fit
    assert converter._box_size == 10
    assert converter._border == 4
    assert converter._back_color == (255, 255, 255)
    assert converter._fill_color == (0, 0, 0)
    assert converter._output_filename == "sample_file.png"


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


def test_text_image_converter_input_supported():
    converter = QRCodeConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image_path") is False
