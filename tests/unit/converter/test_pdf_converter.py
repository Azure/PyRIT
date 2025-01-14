# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fpdf import FPDF
from pypdf import PdfReader, PageObject

from pyrit.models import SeedPrompt, DataTypeSerializer
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


@pytest.fixture
def mock_pdf_path(tmp_path):
    """Create and return the path for a mock PDF with multiple pages."""
    pdf_path = tmp_path / "mock_test.pdf"

    # Create a multi-page PDF
    pdf = FPDF(format="A4")
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Page 1 content", ln=True, align="L")
    pdf.add_page()
    pdf.cell(200, 10, txt="Page 2 content", ln=True, align="L")

    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)

    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes.getvalue())

    return pdf_path


@pytest.mark.asyncio
async def test_injection_into_mock_pdf(mock_pdf_path):
    """Test injecting text into a generic mock PDF."""
    # Define injection items
    injection_items = [
        {
            "page": 0,
            "x": 50,
            "y": 700,
            "text": "Injected Text",
            "font_size": 12,
            "font": "Helvetica",
            "font_color": (255, 0, 0),
        }
    ]

    # Load the mock PDF
    with open(mock_pdf_path, "rb") as pdf_file:
        existing_pdf = BytesIO(pdf_file.read())

    # Create the PDFConverter with the mock PDF and injection items
    converter = PDFConverter(existing_pdf=existing_pdf, injection_items=injection_items)

    # Perform the injection
    result = await converter.convert_async(prompt="")

    # Get the local file path from the result
    modified_pdf_path = Path(result.output_text)
    assert modified_pdf_path.exists(), "Modified PDF file does not exist"

    # Open and validate the modified PDF
    with open(modified_pdf_path, "rb") as modified_pdf_file:
        modified_pdf = PdfReader(modified_pdf_file)
        page_content = modified_pdf.pages[0].extract_text()

    # Validate the injected text
    assert "Injected Text" in page_content, "Injected text not found on page 0"
    modified_pdf_path.unlink()  # Clean up after the test


@pytest.mark.asyncio
async def test_multiple_injections_into_mock_pdf(mock_pdf_path):
    """Test injecting text into multiple pages of a generic mock PDF."""
    # Define multiple injection items
    injection_items = [
        {
            "page": 0,
            "x": 50,
            "y": 700,
            "text": "Page 0 Injection",
            "font_size": 12,
            "font": "Helvetica",
            "font_color": (255, 0, 0),
        },
        {
            "page": 1,
            "x": 75,
            "y": 650,
            "text": "Page 1 Injection",
            "font_size": 10,
            "font": "Helvetica",
            "font_color": (0, 255, 0),
        },
    ]

    # Load the mock PDF
    with open(mock_pdf_path, "rb") as pdf_file:
        existing_pdf = BytesIO(pdf_file.read())

    # Create the PDFConverter with the mock PDF and multiple injection items
    converter = PDFConverter(existing_pdf=existing_pdf, injection_items=injection_items)

    # Perform the injections
    result = await converter.convert_async(prompt="")

    # Get the local file path from the result
    modified_pdf_path = Path(result.output_text)
    assert modified_pdf_path.exists(), "Modified PDF file does not exist"

    # Open and validate the modified PDF
    with open(modified_pdf_path, "rb") as modified_pdf_file:
        modified_pdf = PdfReader(modified_pdf_file)
        page_0_content = modified_pdf.pages[0].extract_text()
        page_1_content = modified_pdf.pages[1].extract_text()

    # Validate the injected text
    assert "Page 0 Injection" in page_0_content, "Injected text not found on page 0"
    assert "Page 1 Injection" in page_1_content, "Injected text not found on page 1"
    modified_pdf_path.unlink()  # Clean up after the test


@pytest.fixture
def dummy_page():
    """
    Create a dummy page with a known mediabox simulating an A4 page.
    An A4 page in points is approximately 595 x 842.
    """
    page = MagicMock(spec=PageObject)
    page.mediabox = [0, 0, 595, 842]
    return page


def test_inject_text_negative_x(dummy_page):
    """Test that a negative x coordinate raises a ValueError."""
    converter = PDFConverter(prompt_template=None)
    with pytest.raises(ValueError) as excinfo:
        converter._inject_text_into_page(
            page=dummy_page,
            x=-10,
            y=100,
            text="Test Text",
            font="Arial",
            font_size=12,
            font_color=(0, 0, 0),
        )
    assert "x_pos is less than 0" in str(excinfo.value)


def test_inject_text_x_exceeds_page_width(dummy_page):
    """Test that an x coordinate exceeding the A4 page width (595) raises a ValueError."""
    converter = PDFConverter(prompt_template=None)
    with pytest.raises(ValueError) as excinfo:
        converter._inject_text_into_page(
            page=dummy_page,
            x=600,  # 600 exceeds the A4 width of 595
            y=100,
            text="Test Text",
            font="Arial",
            font_size=12,
            font_color=(0, 0, 0),
        )
    assert "x_pos exceeds page width" in str(excinfo.value)


def test_inject_text_negative_y(dummy_page):
    """Test that a negative y coordinate raises a ValueError."""
    converter = PDFConverter(prompt_template=None)
    with pytest.raises(ValueError) as excinfo:
        converter._inject_text_into_page(
            page=dummy_page,
            x=100,
            y=-20,
            text="Test Text",
            font="Arial",
            font_size=12,
            font_color=(0, 0, 0),
        )
    assert "y_pos is less than 0" in str(excinfo.value)


def test_inject_text_y_exceeds_page_height(dummy_page):
    """Test that a y coordinate exceeding the A4 page height (842) raises a ValueError."""
    converter = PDFConverter(prompt_template=None)
    with pytest.raises(ValueError) as excinfo:
        converter._inject_text_into_page(
            page=dummy_page,
            x=100,
            y=850,  # 850 exceeds the A4 height of 842
            text="Test Text",
            font="Arial",
            font_size=12,
            font_color=(0, 0, 0),
        )
    assert "y_pos exceeds page height" in str(excinfo.value)
