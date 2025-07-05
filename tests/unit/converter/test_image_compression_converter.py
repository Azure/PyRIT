# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from pyrit.prompt_converter import ImageCompressionConverter


@pytest.fixture
def sample_image_bytes():
    """Sample RGB image for testing with configurable format and size."""

    def _create_image(format="PNG", size=(200, 200)):
        img = Image.new("RGB", size, color=(125, 125, 125))
        img_bytes = BytesIO()
        img.save(img_bytes, format=format)
        return img_bytes.getvalue()

    return _create_image


@pytest.fixture
def sample_transparent_image_bytes():
    """Sample RGBA image with transparency for testing with configurable format."""

    def _create_image(format="PNG"):
        img = Image.new("RGBA", (200, 200), color=(125, 125, 125, 128))
        img_bytes = BytesIO()
        img.save(img_bytes, format=format)
        return img_bytes.getvalue()

    return _create_image


def test_image_compression_converter_initialization():
    """Constructor/Initialization Tests"""

    for unsupported_format in ["GIF", "BMP", "TIFF", "ICO", "WEBM", "SVG"]:
        with pytest.raises(ValueError):
            ImageCompressionConverter(output_format=unsupported_format)  # type: ignore

    for invalid_quality in [-1, 101]:
        with pytest.raises(ValueError):
            ImageCompressionConverter(quality=invalid_quality)

    for invalid_compress_level in [-1, 10]:
        with pytest.raises(ValueError):
            ImageCompressionConverter(compress_level=invalid_compress_level)

    for invalid_method in [-1, 7]:
        with pytest.raises(ValueError):
            ImageCompressionConverter(method=invalid_method)

    for invalid_min_compression_threshold in [-1, -100]:
        with pytest.raises(ValueError):
            ImageCompressionConverter(min_compression_threshold=invalid_min_compression_threshold)


def test_image_compression_converter_quality_warning():
    """Test that high quality values for JPEG trigger a warning."""
    with patch("pyrit.prompt_converter.image_compression_converter.logger") as mock_logger:
        ImageCompressionConverter(output_format="JPEG", quality=98)
        mock_logger.warning.assert_called_once()


@pytest.mark.parametrize(
    "input_format, output_format, expected_output_format",
    [
        # Format preservation
        ("JPEG", None, "JPEG"),
        ("WEBP", None, "WEBP"),
        # Cross-format conversion
        ("PNG", "JPEG", "JPEG"),
        ("WEBP", "PNG", "PNG"),
    ],
)
@pytest.mark.asyncio
async def test_image_compression_converter_format_preservation_and_conversion(
    duckdb_instance, sample_image_bytes, input_format, output_format, expected_output_format
):
    """Test format preservation and conversion between formats."""
    converter = ImageCompressionConverter(
        output_format=output_format, min_compression_threshold=100, fallback_to_original=False
    )
    # Create larger image to ensure compression happens
    image_bytes = sample_image_bytes(format=input_format, size=(2048, 2048))

    with patch("pyrit.prompt_converter.image_compression_converter.data_serializer_factory") as mock_factory:
        mock_serializer = AsyncMock()
        mock_serializer.read_data.return_value = image_bytes
        mock_serializer.save_b64_image = AsyncMock()
        # Set the value to match input format initially
        mock_serializer.value = f"test_image.{input_format.lower()}"
        # Mock the file_extension property to be settable
        mock_serializer.file_extension = input_format.lower()
        mock_factory.return_value = mock_serializer

        await converter.convert_async(prompt=f"test_image.{input_format.lower()}", input_type="image_path")

        # Verify the save method was called
        mock_serializer.save_b64_image.assert_called_once()
        mock_serializer.read_data.assert_called_once()

        # Check that file extension was updated correctly
        expected_extension = expected_output_format.lower()
        assert mock_serializer.file_extension == expected_extension


@pytest.mark.parametrize(
    "input_format, output_format, background_color, expected_output_format",
    [
        ("PNG", "JPEG", (255, 255, 255), "JPEG"),
        ("WEBP", "JPEG", (0, 0, 0), "JPEG"),
        ("TIFF", "JPEG", (255, 0, 0), "JPEG"),
    ],
)
@pytest.mark.asyncio
async def test_image_compression_converter_transparency_handling(
    duckdb_instance,
    sample_transparent_image_bytes,
    input_format,
    output_format,
    background_color,
    expected_output_format,
):
    """Test transparency handling across formats with different background colors."""
    converter = ImageCompressionConverter(output_format=output_format, background_color=background_color)
    image_bytes = sample_transparent_image_bytes(format=input_format)
    image = Image.open(BytesIO(image_bytes))

    assert image.has_transparency_data  # before compression, the image should have transparency

    res_compressed_io, res_output_format = converter._compress_image(image, input_format, len(image_bytes))
    assert res_compressed_io
    assert res_output_format == expected_output_format

    output_image = Image.open(res_compressed_io)
    assert output_image.has_transparency_data is False  # after compression, the image should not have transparency


@pytest.mark.asyncio
async def test_image_compression_converter_skip(duckdb_instance, sample_image_bytes):
    """Test cases for skipping compression and fallback to original image."""

    # 1: Skip compression for small images
    converter = ImageCompressionConverter(min_compression_threshold=4567)
    small_image_bytes = sample_image_bytes(format="PNG", size=(50, 50))
    assert len(small_image_bytes) < 4567

    with patch("pyrit.prompt_converter.image_compression_converter.data_serializer_factory") as mock_factory:
        mock_serializer = AsyncMock()
        mock_serializer.read_data.return_value = small_image_bytes
        mock_factory.return_value = mock_serializer

        result = await converter.convert_async(prompt="small_image.png", input_type="image_path")
        assert result.output_text == "small_image.png"

        # Verify that compression was skipped - save_b64_image should not be called
        mock_serializer.save_b64_image.assert_not_called()

    # 2: Fallback to original when compression increases file size
    converter_fallback = ImageCompressionConverter(fallback_to_original=True, quality=100)
    large_image_bytes = sample_image_bytes(format="PNG", size=(500, 500))

    with patch("pyrit.prompt_converter.image_compression_converter.data_serializer_factory") as mock_factory:
        mock_serializer = AsyncMock()
        mock_serializer.read_data.return_value = large_image_bytes
        mock_factory.return_value = mock_serializer

        # Mock _compress_image to return larger size
        with patch.object(converter_fallback, "_compress_image") as mock_compress:
            larger_bytes = BytesIO(b"x" * (len(large_image_bytes) + 1000))
            mock_compress.return_value = (larger_bytes, "PNG")

            result = await converter_fallback.convert_async(prompt="test.png", input_type="image_path")
            assert result.output_text == "test.png"

            mock_serializer.save_b64_image.assert_not_called()
            mock_compress.assert_called_once()  # compression was attempted but resulted in larger size


@pytest.mark.asyncio
async def test_image_compression_converter_actually_compresses(duckdb_instance, sample_image_bytes):
    """Test that compression actually reduces file size for appropriate images."""
    converter = ImageCompressionConverter(compress_level=9)

    # Create a large image that should benefit from compression
    large_image_bytes = sample_image_bytes(format="PNG", size=(1024, 1024))
    original_size = len(large_image_bytes)
    image = Image.open(BytesIO(large_image_bytes))

    compressed_io, output_format = converter._compress_image(image, "PNG", original_size)
    compressed_size = len(compressed_io.getvalue())

    assert compressed_size < original_size


@pytest.mark.asyncio
async def test_image_compression_converter_url_format_conversion(duckdb_instance, sample_image_bytes):
    """Test successful compression of image from URL."""
    converter = ImageCompressionConverter(output_format="WEBP", min_compression_threshold=100)
    test_url = "https://example.com/test_image.jpeg"
    large_image_bytes = sample_image_bytes(format="JPEG", size=(2048, 2048))

    with patch("pyrit.prompt_converter.image_compression_converter.data_serializer_factory") as mock_factory:
        mock_serializer = AsyncMock()
        mock_serializer.file_extension = "jpeg"
        mock_serializer.value = "converted_image.webp"
        mock_serializer.save_b64_image = AsyncMock()
        mock_factory.return_value = mock_serializer

        with patch.object(converter, "_read_image_from_url") as mock_read_url:
            mock_read_url.return_value = large_image_bytes

            result = await converter.convert_async(prompt=test_url, input_type="url")

            assert result.output_text == "converted_image.webp"
            assert result.output_type == "image_path"
            # Verify file extension was updated to match WEBP output format
            assert mock_serializer.file_extension == "webp"
            mock_serializer.save_b64_image.assert_called_once()


@pytest.mark.asyncio
async def test_image_compression_converter_url_input_fallback_scenarios(duckdb_instance, sample_image_bytes):
    """Test URL input fallback scenarios (small image size)."""
    converter = ImageCompressionConverter(min_compression_threshold=5000, fallback_to_original=True)
    test_url = "https://example.com/small_image.png"
    small_image_bytes = sample_image_bytes(format="PNG", size=(100, 100))

    with patch("pyrit.prompt_converter.image_compression_converter.data_serializer_factory") as mock_factory:
        mock_serializer = AsyncMock()
        mock_serializer.file_extension = "png"
        mock_serializer.value = "fallback_image.png"
        mock_serializer.save_data = AsyncMock()
        mock_factory.return_value = mock_serializer

        with patch.object(converter, "_read_image_from_url") as mock_read_url:
            mock_read_url.return_value = small_image_bytes

            result = await converter.convert_async(prompt=test_url, input_type="url")

            assert result.output_text == "fallback_image.png"
            assert result.output_type == "image_path"
            mock_read_url.assert_called_once_with(test_url)
            mock_serializer.save_data.assert_called_once_with(small_image_bytes)
            mock_serializer.save_b64_image.assert_not_called()


@pytest.mark.asyncio
async def test_image_compression_converter_invalid_url():
    """Test handling of invalid URLs."""
    converter = ImageCompressionConverter()
    invalid_urls = ["ftp://example.com/image.png", "file:///local/path/image.png", "not-url", "example.com/image.png"]
    for invalid_url in invalid_urls:
        with pytest.raises(ValueError, match="Invalid URL"):
            await converter.convert_async(prompt=invalid_url, input_type="url")
