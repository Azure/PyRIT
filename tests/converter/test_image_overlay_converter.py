# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os

import pytest
from PIL import Image

from pyrit.prompt_converter import ImageOverlayConverter

from io import BytesIO


@pytest.fixture
def base_image_path():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save("base_test.png")
    return "base_test.png"


@pytest.fixture
def overlay_image_byte():
    img = Image.new("RGBA", (20, 20), color=(125, 125, 125, 125))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    return img_bytes


def test_image_overlay_converter_initialization(base_image_path):
    converter = ImageOverlayConverter(
        base_image_path=base_image_path, x_pos=10, y_pos=10, memory=None
    )
    assert converter._base_image_path == "base_test.png"
    assert converter._x_pos == 10
    assert converter._y_pos == 10
    os.remove("base_test.png")


def test_image_overlay_converter_invalid_image():
    with pytest.raises(ValueError):
        ImageOverlayConverter(base_image_path="")


def test_image_overlay_converter_add_overlay_image(base_image_path, overlay_image_byte):
    converter = ImageOverlayConverter(base_image_path=base_image_path)
    base_image = Image.open(base_image_path)
    overlay_image = Image.open(overlay_image_byte)
    pixels_before = list(base_image.getdata())

    # Adding overlay image
    updated_image = converter._add_overlay_image(overlay_image)
    pixels_after = list(updated_image.getdata())

    assert updated_image is not None
    # Check for pixels changes
    assert pixels_before != pixels_after
    os.remove("base_test.png")