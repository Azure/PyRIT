# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import MagicMock, patch

from pyrit.memory import MemoryInterface
from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.pdf_converter import PDFConverter
from pyrit.models.data_type_serializer import DataTypeSerializer


@pytest.fixture
def mock_memory():
    """Create a mock MemoryInterface using MagicMock."""
    memory_mock = MagicMock(spec=MemoryInterface)

    # Mock necessary attributes used by PDFConverter
    memory_mock.results_storage_io = MagicMock()
    memory_mock.results_path = "/mock/results/path"

    # Mock all abstract methods with default return values
    memory_mock._add_embeddings_to_memory.return_value = None
    memory_mock._get_prompt_pieces_by_orchestrator.return_value = []
    memory_mock._get_prompt_pieces_with_conversation_id.return_value = []
    memory_mock._init_storage_io.return_value = None
    memory_mock.add_request_pieces_to_memory.return_value = None
    memory_mock.dispose_engine.return_value = None
    memory_mock.get_all_embeddings.return_value = []
    memory_mock.get_all_prompt_pieces.return_value = []
    memory_mock.get_prompt_request_pieces_by_id.return_value = []
    memory_mock.insert_entries.return_value = None
    memory_mock.insert_entry.return_value = None
    memory_mock.query_entries.return_value = []
    memory_mock.update_entries.return_value = None

    return memory_mock


@pytest.fixture
def pdf_converter_no_template(mock_memory):
    """A PDFConverter with no template path provided."""
    return PDFConverter(template_path=None, memory=mock_memory)


@pytest.fixture
def pdf_converter_with_template(tmp_path, mock_memory):
    """A PDFConverter with a valid template file."""
    template_content = "This is a test template. Prompt: {{ prompt }}"
    template_file = tmp_path / "test_template.txt"
    template_file.write_text(template_content)
    return PDFConverter(template_path=str(template_file), memory=mock_memory)


@pytest.fixture
def pdf_converter_nonexistent_template(mock_memory):
    """A PDFConverter with a non-existent template path."""
    return PDFConverter(template_path="nonexistent_template_path.txt", memory=mock_memory)


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
        mock_factory.assert_called_once_with(data_type="url", value=prompt, memory=pdf_converter_no_template._memory)


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
async def test_convert_async_custom_font_and_size(mock_memory):
    """Test PDF generation with custom font and size parameters."""
    converter = PDFConverter(
        template_path=None,
        memory=mock_memory,
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
