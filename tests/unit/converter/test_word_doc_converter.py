from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document  # type: ignore[import-untyped]

from pyrit.prompt_converter import ConverterResult, WordDocConverter


@pytest.fixture
def word_doc_converter_no_template() -> WordDocConverter:
    """A WordDocConverter with no template or existing document."""
    return WordDocConverter(prompt_template=None)


def test_input_supported(word_doc_converter_no_template: WordDocConverter) -> None:
    """Test that only 'text' input types are supported."""
    assert word_doc_converter_no_template.input_supported("text") is True
    assert word_doc_converter_no_template.input_supported("image") is False


@pytest.mark.asyncio
async def test_convert_async_creates_new_docx(word_doc_converter_no_template: WordDocConverter) -> None:
    """convert_async should generate a new document when no existing_docx is configured."""
    prompt = "Hello, Word!"
    mock_doc_bytes = b"mock-docx-bytes"

    with (
        patch.object(word_doc_converter_no_template, "_prepare_content", return_value=prompt) as mock_prepare,
        patch.object(
            word_doc_converter_no_template,
            "_generate_new_docx",
            return_value=mock_doc_bytes,
        ) as mock_generate,
        patch.object(word_doc_converter_no_template, "_inject_into_existing_docx") as mock_inject,
        patch.object(word_doc_converter_no_template, "_serialize_docx_async") as mock_serialize,
    ):
        serializer_mock = MagicMock()
        serializer_mock.value = "mock_path.docx"
        mock_serialize.return_value = serializer_mock

        result = await word_doc_converter_no_template.convert_async(prompt=prompt)

    mock_prepare.assert_called_once_with(prompt)
    mock_generate.assert_called_once_with(prompt)
    mock_inject.assert_not_called()
    mock_serialize.assert_called_once_with(mock_doc_bytes)

    assert isinstance(result, ConverterResult)
    assert result.output_type == "binary_path"
    assert result.output_text == "mock_path.docx"


@pytest.mark.asyncio
async def test_convert_async_injects_into_existing_docx(tmp_path: Path) -> None:
    """convert_async should inject into an existing document when existing_docx is configured."""
    existing_docx_path = tmp_path / "template.docx"
    document = Document()
    document.add_paragraph("Prefix {{INJECTION_PLACEHOLDER}} Suffix")
    document.save(existing_docx_path)

    converter = WordDocConverter(existing_docx=existing_docx_path)

    prompt = "Injected content"
    mock_doc_bytes = b"modified-docx-bytes"

    with (
        patch.object(converter, "_prepare_content", return_value=prompt) as mock_prepare,
        patch.object(converter, "_generate_new_docx") as mock_generate,
        patch.object(converter, "_inject_into_existing_docx", return_value=mock_doc_bytes) as mock_inject,
        patch.object(converter, "_serialize_docx_async") as mock_serialize,
    ):
        serializer_mock = MagicMock()
        serializer_mock.value = "injected.docx"
        mock_serialize.return_value = serializer_mock

        result = await converter.convert_async(prompt=prompt)

    mock_prepare.assert_called_once_with(prompt)
    mock_generate.assert_not_called()
    mock_inject.assert_called_once_with(prompt)
    mock_serialize.assert_called_once_with(mock_doc_bytes)

    assert isinstance(result, ConverterResult)
    assert result.output_type == "binary_path"
    assert result.output_text == "injected.docx"


@pytest.mark.asyncio
async def test_convert_async_unsupported_input_type(word_doc_converter_no_template: WordDocConverter) -> None:
    """convert_async should raise for unsupported input types."""
    with pytest.raises(ValueError, match="Input type not supported"):
        await word_doc_converter_no_template.convert_async(prompt="ignored", input_type="image")


def test_inject_into_existing_docx_requires_existing_path(tmp_path: Path) -> None:
    """_inject_into_existing_docx should raise if called without an existing_docx path."""
    converter = WordDocConverter(existing_docx=None)

    with pytest.raises(ValueError, match="no 'existing_docx' path was provided"):
        converter._inject_into_existing_docx("content")


def test_placeholder_replaced_when_in_single_run(tmp_path: Path) -> None:
    """The placeholder should be replaced when it appears in a single run."""
    existing_docx_path = tmp_path / "placeholder_single_run.docx"
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Hello {{INJECTION_PLACEHOLDER}}!")
    document.save(existing_docx_path)

    converter = WordDocConverter(existing_docx=existing_docx_path)

    result_bytes = converter._inject_into_existing_docx("WORLD")
    new_doc = Document(BytesIO(result_bytes))

    assert "WORLD" in new_doc.paragraphs[0].text
    assert "{{INJECTION_PLACEHOLDER}}" not in new_doc.paragraphs[0].text


def test_placeholder_not_replaced_when_split_across_runs(tmp_path: Path) -> None:
    """The placeholder should not be replaced when split across multiple runs."""
    existing_docx_path = tmp_path / "placeholder_multi_run.docx"
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Hello {{")
    paragraph.add_run("INJECTION_PLACEHOLDER}}!")
    document.save(existing_docx_path)

    converter = WordDocConverter(existing_docx=existing_docx_path)

    result_bytes = converter._inject_into_existing_docx("WORLD")
    new_doc = Document(BytesIO(result_bytes))

    # Since the placeholder spans multiple runs, the content should be unchanged.
    assert new_doc.paragraphs[0].text == "Hello {{INJECTION_PLACEHOLDER}}!"

