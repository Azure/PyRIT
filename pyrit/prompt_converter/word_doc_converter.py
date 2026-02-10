# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from io import BytesIO
from pathlib import Path
from typing import Optional
from docx import Document
from docx.shared import Pt
from jinja2 import Template

from pyrit.common.logger import logger
from pyrit.identifiers import ConverterIdentifier
from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.models.data_type_serializer import DataTypeSerializer
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


class WordDocConverter(PromptConverter):
    """
    Convert a text prompt into a Word document file.

    Supports two modes:
    1. Direct generation: Create a new word document from the prompt text, applying the configured font settings.
    2. Template-based generation: Inject the prompt text into an existing .docx file that 
        contains jinja2 placeholders (e.g. {{prompt}}). The original file is read once at init and stored in memory, 
        so it is never modified, a new file is generated on each conversion.

    The output of both modes is a binary_path pointing to the newly created .docx file, serialized through PyRIT's data_serializer_factory.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("binary_path",)

    def __init__(
        self,
        *, 
        font_name: str = "Calibri", # Default font for direct generation mode.
        font_size: int = 12, # Default font size for direct generation mode (must be positive).
        existing_doc: Optional[Path] = None,
    ) -> None:
        """
        Initialize the wordDocConverter with font settings and a template document (optional).

        Args:
            font_name: Font family applied when generating a new document in direct mode.
                Ignored when ``existing_doc`` is provided (the existing file keeps its own
                fonts).  Default font is ``"Calibri"``.
            font_size: Font size in points for direct mode.  Must be a positive integer.
                Default font size is ``12``.
            existing_doc: Path to a ``.docx`` template file that contains jinja2
                placeholders (e.g. ``{{ prompt }}``).  The file is read once at init and
                stored in memory so the original is never touched.  Default is None.

        Raises:
            ValueError: If ``font_size`` is not a positive integer.
            FileNotFoundError: If ``existing_doc`` does not point to an existing file.
        """
        if font_size <= 0:
            raise ValueError(f"font_size must be a positive integer, got {font_size}.")

        self._font_name = font_name
        self._font_size = font_size

        self._existing_doc_path: Optional[Path] = existing_doc
        self._existing_doc_bytes: Optional[BytesIO] = None

        if existing_doc is not None:
            if not existing_doc.is_file():
                raise FileNotFoundError(f"Word document not found at: {existing_doc}")

            # Read the template once and keep the bytes in memory for repeated use without re-reading the file.
            with open(existing_doc, "rb") as doc_file:
                self._existing_doc_bytes = BytesIO(doc_file.read()) 


    def _build_identifier(self) -> ConverterIdentifier:
        """
        Build identifier with converter-specific parameters.

        Returns:
            ConverterIdentifier: Identifier containing font and template path info.
        """
        return self._create_identifier(
            converter_specific_params={
                "font_name": self._font_name,
                "font_size": self._font_size,
                "existing_doc_path": str(self._existing_doc_path) if self._existing_doc_path else None,
            }
        )


    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert text prompt into a Word document.

        If ``existing_doc`` was provided at init, the prompt is injected into that template by replacing jinja2 placeholders 
            (e.g. ``{{ prompt }}``) while preserving all original formatting. 
        If no template was provided, a new document is generated where each line of the prompt (split on ``\\n``)
            becomes a new paragraph with the configured font settings.

        Args:
            prompt: The text to embed in the Word document.
            input_type: Must be ``text``.

        Returns:
            ConverterResult: Contains the file path (``binary_path``) to the new ``.docx``.

        Raises:
            ValueError: If ``input_type`` is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if self._existing_doc_bytes:
            docx_bytes = self._render_template_docx(prompt)
        else:
            docx_bytes = self._generate_docx(prompt)

        serializer = await self._serialize_docx(docx_bytes)
        return ConverterResult(output_text=serializer.value, output_type="binary_path")


    def _generate_docx(self, content: str) -> bytes:
        """
        Create a new Word document from plain text.

        Each line (split on ``\\n``) becomes a separate paragraph.  The configured ``font_name`` and ``font_size`` are applied
            to the normal style so all paragraphs inherit them.

        Args:
            content: The text content.  Newlines create paragraph breaks.

        Returns:
            The ``.docx`` file content as raw bytes.
        """
        document = Document()

        # Apply font settings to the Normal style so every paragraph inherits them.
        style = document.styles["Normal"]
        font = style.font
        font.name = self._font_name
        font.size = Pt(self._font_size)

        # Add each line of the prompt as a new paragraph when splitting on newlines.
        for paragraph_text in content.split("\n"):
            document.add_paragraph(paragraph_text)

        # Save the document to a bytes buffer and return the raw bytes.
        docx_buffer = BytesIO()
        document.save(docx_buffer)
        docx_buffer.seek(0) # Rewind to read from the start.
        return docx_buffer.getvalue()

    def _render_template_docx(self, prompt: str) -> bytes:
        """
        Replace jinja2 placeholders in an existing Word document with the prompt.

        The method works on an in-memory copy of the original file, so it is
            safe to call repeatedly without mutating the template.

        Scanning locations:
            - Body paragraphs
            - Table cell paragraphs
            - Header paragraphs (all sections)
            - Footer paragraphs (all sections)

        Args:
            prompt: The text to inject into ``{{ prompt }}`` placeholders.

        Returns:
            The modified ``.docx`` file content as raw bytes.

        Raises:
            ValueError: If the in-memory template bytes are unavailable (should not
                happen when called through ``convert_async``).
        """
        if not self._existing_doc_bytes:
            raise ValueError("Existing document bytes are required for template-based generation.")

        # Rewind to read from the start of the stored bytes.
        self._existing_doc_bytes.seek(0)
        document = Document(self._existing_doc_bytes)

        template_vars = {"prompt": prompt}

        # Body paragraphs
        for paragraph in document.paragraphs:
            self._render_paragraph(paragraph, template_vars)

        # Table cells (each cell can contain multiple paragraphs)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        self._render_paragraph(paragraph, template_vars)

        # Headers and footers (one header/footer per section)
        for section in document.sections:
            for paragraph in section.header.paragraphs:
                self._render_paragraph(paragraph, template_vars)
            for paragraph in section.footer.paragraphs:
                self._render_paragraph(paragraph, template_vars)

        docx_buffer = BytesIO()
        document.save(docx_buffer)
        docx_buffer.seek(0)
        return docx_buffer.getvalue()

    def _render_paragraph(self, paragraph: "docx.text.paragraph.Paragraph", template_vars: dict) -> None:
        """
        Render jinja2 placeholders inside a single paragraph.

        Word internally splits paragraph text across multiple "runs" (text segments with their own formatting).  
            A placeholder like ``{{ prompt }}`` may be spread across several runs. 
        
        To handle this, render the entire paragraph text as one string, then write the rendered result back into the runs.
            This means if a placeholder is split across runs, the formatting of the first run will be applied to the 
            entire rendered text, and the subsequent runs will be cleared.

        If the paragraph contains no placeholders, it is left untouched.

        Args:
            paragraph: A python-docx ``Paragraph`` object to process.
            template_vars: Mapping of placeholder names to replacement values
                (e.g. ``{"prompt": "injected text"}``).
        """
        full_text = paragraph.text

        # Fast exit (skip paragraphs that have no jinja2 markers).
        if "{{" not in full_text or "}}" not in full_text:
            return

        try:
            template = Template(full_text)
            rendered_text = template.render(**template_vars)
        except Exception as e:
            logger.warning(f"Failed to render paragraph template: {e}")
            return

        # Nothing changed, leave the paragraph as-is.
        if rendered_text == full_text:
            return

        # Write rendered text back while preserving format of the first run.
        if paragraph.runs:
            first_run = paragraph.runs[0]
            # Clear all subsequent runs (their text is now part of rendered_text).
            for run in paragraph.runs[1:]:
                run.text = ""

            first_run.text = rendered_text
        else:
            paragraph.text = rendered_text


    async def _serialize_docx(self, docx_bytes: bytes) -> DataTypeSerializer:
        """
        Save the generated ``.docx`` bytes through PyRIT's data serializer.

        The serializer picks a unique filename and writes the bytes to the configured storage location (local disk by default).

        Args:
            docx_bytes: Raw content of the Word document.

        Returns:
            DataTypeSerializer: Serializer whose ``.value`` contains the output path.
        """
        docx_serializer = data_serializer_factory(
            category="prompt-memory-entries",
            data_type="binary_path",
            extension="docx",
        )

        await docx_serializer.save_data(docx_bytes)
        
        return docx_serializer
