# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pyrit.prompt_converter import ImageOverlayConverter

from io import BytesIO


@pytest.fixture
def base_image_path():
    # Create a temporary file with a unique name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        # Generate a simple image and save it to the temporary file path
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(tmp.name)
        temp_path = tmp.name  # Store the temporary file path

    yield temp_path  # Provide the path to the test

    # Cleanup after the test
    os.remove(temp_path)


@pytest.fixture
def overlay_image_path():
    # Create a temporary file with a unique name for overlay image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        # Generate a simple image and save it to the temporary file path
        img = Image.new("RGB", (10, 10), color=(100, 100, 100))
        img.save(tmp.name)
        temp_path = tmp.name  # Store the temporary file path

    yield temp_path  # Provide the path to the test

    # Cleanup after the test
    os.remove(temp_path)


def test_image_overlay_converter_initialization(base_image_path):
    # use MagicMock for memory
    memory_mock = MagicMock()

    converter = ImageOverlayConverter(
        base_image_path=base_image_path, x_pos=10, y_pos=15, memory=memory_mock
    )
    assert converter._base_image_path == base_image_path, " Base image path should be initialized"
    assert converter._x_pos == 10, "X position should be 10"
    assert converter._y_pos == 15, "Y position should be 15"


def test_image_overlay_converter_invalid_image():
    with pytest.raises(ValueError):
        ImageOverlayConverter(base_image_path="")


@pytest.mark.asyncio
async def test_image_overlay_converter_convert_async(base_image_path, overlay_image_path):
    # Initialize the converter with the base image path
    converter = ImageOverlayConverter(base_image_path=base_image_path)

    # Call the async `convert_async` method with the overlay image path as the prompt
    result = await converter.convert_async(prompt=overlay_image_path, input_type="image_path")

    # Verify that the result contains a valid file path in `output_text`
    assert isinstance(result.output_text, str), "The result should contain a file path as output_text."
    output_path = Path(result.output_text)

    # Check that the output path exists and is a file
    assert output_path.is_file(), "The output image file should exist."

    # Open the result image and verify its properties
    with Image.open(output_path) as img:
        # Ensure the output image dimensions match the base image
        assert img.size == Image.open(base_image_path).size, "The output image size should match the base image size."
        # Check if the image mode is appropriate
        assert img.mode in ["RGB", "RGBA"], "The output image mode should be RGB or RGBA."

    # Cleanup after test
    output_path.unlink()
