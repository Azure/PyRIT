# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.data_type_serializer import DataTypeSerializer
from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.pdf_converter import PDFConverter
from unit.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def pdf_converter_no_template():
    """A PDFConverter with no template path provided."""
    return PDFConverter(template_path=None)


@pytest.fixture
def pdf_converter_with_template(
    tmp_path,
):
    """A PDFConverter with a valid template file."""
    template_content = "This is a test template. Prompt: {{ prompt }}"
    template_file = tmp_path / "test_template.txt"
    template_file.write_text(template_content)
    converter = PDFConverter(template_path=str(template_file))

    yield converter
    if template_file.exists():
        os.remove(template_file)


@pytest.fixture
def pdf_converter_nonexistent_template():
    """A PDFConverter with a non-existent template path."""
    return PDFConverter(template_path="nonexistent_template_path.txt")


@pytest.mark.asyncio
async def test_convert_async_no_template(pdf_converter_no_template):
    """Test converting a prompt without a template."""
    prompt = "Hello, PDF!"

    # Mock the data serializer so no real I/O happens.
    with patch("pyrit.prompt_converter.pdf_converter.data_serializer_factory") as mock_factory:
        serializer_mock = MagicMock(spec=DataTypeSerializer)
        serializer_mock.value = "mock_url"
        mock_factory.return_value = serializer_mock

        result = await pdf_converter_no_template.convert_async(prompt=prompt)
        assert isinstance(result, ConverterResult)
        assert result.output_type == "url"
        assert result.output_text == "mock_url"

        # Check if serializer was called to save data
        serializer_mock.save_data.assert_called_once()
        # Check that the prompt was passed to the serializer
        mock_factory.assert_called_once_with(data_type="url", value=prompt)


@pytest.mark.asyncio
async def test_convert_async_with_template(pdf_converter_with_template):
    """Test converting a prompt using a provided template."""
    prompt = "TemplateTest"

    # Mock serializer
    with patch("pyrit.prompt_converter.pdf_converter.data_serializer_factory") as mock_factory:
        serializer_mock = MagicMock(spec=DataTypeSerializer)
        serializer_mock.value = "mock_url"
        mock_factory.return_value = serializer_mock

        result = await pdf_converter_with_template.convert_async(prompt=prompt)
        assert isinstance(result, ConverterResult)
        # The serializer value is "mock_url", so we just assert that is returned as output
        assert result.output_text == "mock_url"
        serializer_mock.save_data.assert_called_once()


@pytest.mark.asyncio
async def test_convert_async_nonexistent_template(pdf_converter_nonexistent_template):
    """Test behavior when the template file does not exist."""
    with pytest.raises(FileNotFoundError):
        await pdf_converter_nonexistent_template.convert_async(prompt="This will fail")


@pytest.mark.asyncio
async def test_convert_async_custom_font_and_size():
    """Test PDF generation with custom font and size parameters."""
    converter = PDFConverter(
        template_path=None,
        font_type="Courier",
        font_size=14,
        page_width=200,
        page_height=280,
    )

    prompt = "Custom font and size test."
    with patch("pyrit.prompt_converter.pdf_converter.data_serializer_factory") as mock_factory:
        serializer_mock = MagicMock(spec=DataTypeSerializer)
        serializer_mock.value = "mock_url"
        mock_factory.return_value = serializer_mock

        result = await converter.convert_async(prompt=prompt)
        assert isinstance(result, ConverterResult)
        assert result.output_text == "mock_url"
        serializer_mock.save_data.assert_called_once()


def test_input_supported(pdf_converter_no_template):
    """Test that only 'text' input types are supported."""
    assert pdf_converter_no_template.input_supported("text") is True
    assert pdf_converter_no_template.input_supported("image") is False
    assert pdf_converter_no_template.input_supported("text") is True


@pytest.mark.asyncio
async def test_convert_async_end_to_end_no_reader(tmp_path):
    prompt = "Test for PDF generation."
    pdf_file_path = tmp_path / "output.pdf"
    converter = PDFConverter(template_path=None)

    result = await converter.convert_async(prompt=prompt)
    with open(pdf_file_path, "wb") as pdf_file:
        pdf_file.write(result.output_text.encode("latin1"))

    assert pdf_file_path.exists()
    assert os.path.getsize(pdf_file_path) > 0
    pdf_file_path.unlink()
