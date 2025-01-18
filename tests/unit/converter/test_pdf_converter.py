# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models import DataTypeSerializer, SeedPrompt
from pyrit.prompt_converter import ConverterResult, PDFConverter


@pytest.fixture
def pdf_converter_no_template():
    """A PDFConverter with no template path provided."""
    return PDFConverter(prompt_template=None)


@pytest.fixture
def pdf_converter_with_template(
    tmp_path,
):
    """A PDFConverter with a valid template file."""
    template_content = "This is a test template. Prompt: {{ prompt }}"
    template_file = tmp_path / "test_template.txt"
    template_file.write_text(template_content)
    template = SeedPrompt(
        value=template_content,
        data_type="text",
        name="test_template",
        parameters=["prompt"],
    )
    converter = PDFConverter(prompt_template=template)

    yield converter
    if template_file.exists():
        template_file.unlink()


@pytest.fixture
def pdf_converter_nonexistent_template():
    """A PDFConverter with a non-existent template path."""
    return PDFConverter(prompt_template="nonexistent_prompt_template.txt")


@pytest.mark.asyncio
async def test_convert_async_no_template(pdf_converter_no_template):
    """Test converting a prompt without a template."""
    prompt = "Hello, PDF!"
    mock_pdf_bytes = BytesIO(b"mock_pdf_content")

    # Mock internal methods
    with (
        patch.object(pdf_converter_no_template, "_prepare_content", return_value=prompt) as mock_prepare,
        patch.object(pdf_converter_no_template, "_generate_pdf", return_value=mock_pdf_bytes) as mock_generate,
        patch.object(pdf_converter_no_template, "_serialize_pdf") as mock_serialize,
    ):

        serializer_mock = MagicMock()
        serializer_mock.value = "mock_url"
        mock_serialize.return_value = serializer_mock

        result = await pdf_converter_no_template.convert_async(prompt=prompt)

        # Assertions
        mock_prepare.assert_called_once_with(prompt)
        mock_generate.assert_called_once_with(prompt)
        mock_serialize.assert_called_once_with(mock_pdf_bytes, prompt)

        assert isinstance(result, ConverterResult)
        assert result.output_type == "url"
        assert result.output_text == "mock_url"


@pytest.mark.asyncio
async def test_convert_async_with_template(pdf_converter_with_template):
    """Test converting a prompt using a provided template."""
    prompt = {"prompt": "TemplateTest"}
    expected_rendered_content = "This is a test template. Prompt: TemplateTest"
    mock_pdf_bytes = BytesIO(b"mock_pdf_content")

    # Mock internal methods
    with (
        patch.object(
            pdf_converter_with_template, "_prepare_content", return_value=expected_rendered_content
        ) as mock_prepare,
        patch.object(pdf_converter_with_template, "_generate_pdf", return_value=mock_pdf_bytes) as mock_generate,
        patch.object(pdf_converter_with_template, "_serialize_pdf") as mock_serialize,
    ):

        serializer_mock = MagicMock()
        serializer_mock.value = "mock_url"
        mock_serialize.return_value = serializer_mock

        result = await pdf_converter_with_template.convert_async(prompt=prompt)

        # Assertions
        mock_prepare.assert_called_once_with(prompt)
        mock_generate.assert_called_once_with(expected_rendered_content)
        mock_serialize.assert_called_once_with(mock_pdf_bytes, expected_rendered_content)

        assert isinstance(result, ConverterResult)
        assert result.output_type == "url"
        assert result.output_text == "mock_url"


@pytest.mark.asyncio
async def test_convert_async_nonexistent_template(pdf_converter_nonexistent_template):
    """Test behavior when the template file does not exist."""
    with patch.object(pdf_converter_nonexistent_template, "_prepare_content", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            await pdf_converter_nonexistent_template.convert_async(prompt="This will fail")


@pytest.mark.asyncio
async def test_convert_async_custom_font_and_size():
    """Test PDF generation with custom font and size parameters."""
    converter = PDFConverter(
        prompt_template=None,
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


@pytest.mark.asyncio
async def test_convert_async_end_to_end_no_reader(tmp_path, duckdb_instance):
    prompt = "Test for PDF generation."
    pdf_file_path = tmp_path / "output.pdf"
    converter = PDFConverter(prompt_template=None)

    result = await converter.convert_async(prompt=prompt)
    with open(pdf_file_path, "wb") as pdf_file:
        pdf_file.write(result.output_text.encode("latin1"))

    assert pdf_file_path.exists()
    assert pdf_file_path.stat().st_size > 0
    pdf_file_path.unlink()
