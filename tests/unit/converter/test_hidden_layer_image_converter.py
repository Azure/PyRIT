# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy
import pytest
from PIL import Image

from pyrit.prompt_converter import ConverterResult, HiddenLayerConverter


@pytest.fixture
def sample_benign_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.new("RGB", (256, 256), color=(192, 192, 192))
        img.save(tmp.name, "JPEG")
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_attack_image():
    with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as tmp:
        img = Image.new("RGB", (100, 100), color=(64, 64, 64))
        img.save(tmp.name, "JPEG")
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_invalid_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"This is not a valid image file.")
        yield tmp.name
    os.unlink(tmp.name)


class TestHiddenLayerConverter:
    def test_initialization_default_params(self, sample_benign_image):
        converter = HiddenLayerConverter(benign_image_path=sample_benign_image)
        assert converter.benign_image_path == sample_benign_image
        assert converter.size == (150, 150)
        assert converter.steps == 1000
        assert converter.learning_rate == 0.001

    def test_initialization_valid_params(self, sample_benign_image):
        converter = HiddenLayerConverter(
            benign_image_path=sample_benign_image,
            size=(128, 128),
            steps=500,
            learning_rate=0.01,
        )
        assert converter.benign_image_path == sample_benign_image
        assert converter.size == (128, 128)
        assert converter.steps == 500
        assert converter.learning_rate == 0.01

    def test_initialization_invalid_params(self, sample_benign_image):
        for path in [None, "", "invalid_path.txt", "image.png", "image.gif"]:
            with pytest.raises(ValueError):
                HiddenLayerConverter(benign_image_path=path)
        for size in [(128, 0), (0, 0), 128, -1, (-128, -128), (128,), (128, 128, 128)]:
            with pytest.raises(ValueError):
                HiddenLayerConverter(benign_image_path=sample_benign_image, size=size)
        for steps in [-1, 0]:
            with pytest.raises(ValueError):
                HiddenLayerConverter(benign_image_path=sample_benign_image, steps=steps)
        for learning_rate in [-0.01, 0, 1, 1.5]:
            with pytest.raises(ValueError):
                HiddenLayerConverter(benign_image_path=sample_benign_image, learning_rate=learning_rate)

    def test_validate_input_image(self, sample_benign_image):
        for invalid_path in [None, "", "invalid_path.txt", "image.png", "image.gif"]:
            with pytest.raises(ValueError):
                HiddenLayerConverter._validate_input_image(path=invalid_path)
        for nonexistent_path in ["image.jpg", "image.jpeg", "IMAGE.JPG"]:
            with pytest.raises(FileNotFoundError):
                HiddenLayerConverter._validate_input_image(path=nonexistent_path)
        HiddenLayerConverter._validate_input_image(path=sample_benign_image)  # should pass validation

    def test_load_and_preprocess_image(self, sample_benign_image):
        converter = HiddenLayerConverter(benign_image_path=sample_benign_image, size=(50, 50))
        processed_image = converter._load_and_preprocess_image(sample_benign_image)

        assert processed_image.shape == (50, 50, 3)  # height, width, channels
        assert processed_image.dtype == numpy.float32
        assert numpy.all(processed_image >= 0.0) and numpy.all(processed_image <= 1.0)

        for invalid_path in [None, "", "invalid_path.txt", "image.png", "image.gif"]:
            with pytest.raises(ValueError):
                converter._load_and_preprocess_image(invalid_path)

        with pytest.raises(ValueError):
            converter._load_and_preprocess_image(str(sample_invalid_image))

    def test_compute_mse_loss(self, sample_benign_image):
        converter = HiddenLayerConverter(benign_image_path=sample_benign_image)
        blended = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        target = numpy.array([[2.0, 3.0], [4.0, 5.0]])
        expected_loss = 1.0

        loss = converter._compute_mse_loss(blended, target)
        assert loss == expected_loss
        assert isinstance(loss, numpy.floating)

    def test_compute_gradients_alpha_layer(self, sample_benign_image):
        converter = HiddenLayerConverter(benign_image_path=sample_benign_image)
        blended = numpy.array([[2.0, 3.0], [4.0, 5.0]])
        foreground = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        background = numpy.array([[5.0, 6.0], [7.0, 8.0]])
        white_bg = numpy.array([[1.0, 1.0], [1.0, 1.0]])

        gradients = converter._compute_gradients_alpha_layer(blended, foreground, background, white_bg)
        expected = numpy.array([[2.0, 2.5], [3.0, 3.5]])
        numpy.testing.assert_array_almost_equal(gradients, expected)
        assert gradients.shape == blended.shape

    def test_create_blended_image(self, sample_benign_image):
        converter = HiddenLayerConverter(benign_image_path=sample_benign_image)
        attack_image = numpy.array([[[0.2, 0.4, 0.6]]], dtype=numpy.float32)  # 1x1x3 image
        alpha = numpy.array([[[0.8]]], dtype=numpy.float32)  # 1x1x1 alpha

        rgba_image = converter._create_blended_image(attack_image, alpha)

        assert rgba_image.shape == (1, 1, 4)  # RGBA
        assert rgba_image.dtype == numpy.uint8
        assert rgba_image[0, 0, 0] == int(0.2 * 255)  # R
        assert rgba_image[0, 0, 1] == int(0.4 * 255)  # G
        assert rgba_image[0, 0, 2] == int(0.6 * 255)  # B
        assert rgba_image[0, 0, 3] == int(0.8 * 255)  # A

    @pytest.mark.asyncio
    async def test_save_blended_image(self, sample_benign_image):
        with patch("pyrit.prompt_converter.hidden_layer_image_converter.data_serializer_factory") as mock_factory:
            mock_serializer = MagicMock()
            mock_serializer.file_extension = "png"
            mock_serializer.value = "mock_image_path.png"
            mock_serializer.save_b64_image = AsyncMock()
            mock_factory.return_value = mock_serializer

            converter = HiddenLayerConverter(benign_image_path=sample_benign_image)
            attack_image = numpy.ones((10, 10, 3), dtype=numpy.float32) * 0.5
            alpha = numpy.ones((10, 10, 1), dtype=numpy.float32) * 0.7

            result_path = await converter._save_blended_image(attack_image, alpha)

            assert result_path == "mock_image_path.png"
            mock_factory.assert_called_once_with(category="prompt-memory-entries", data_type="image_path")
            mock_serializer.save_b64_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_async_successful(self, sample_benign_image, sample_attack_image):
        with patch("pyrit.prompt_converter.hidden_layer_image_converter.data_serializer_factory") as mock_factory:
            mock_serializer = MagicMock()
            mock_serializer.file_extension = "png"
            mock_serializer.value = "output_image_path.png"
            mock_serializer.save_b64_image = AsyncMock()
            mock_factory.return_value = mock_serializer

            converter = HiddenLayerConverter(
                benign_image_path=sample_benign_image,
                size=(32, 32),
                steps=5,
            )

            result = await converter.convert_async(prompt=sample_attack_image, input_type="image_path")

            assert isinstance(result, ConverterResult)
            assert result.output_text == "output_image_path.png"
            assert result.output_type == "image_path"
            mock_serializer.save_b64_image.assert_called_once()
