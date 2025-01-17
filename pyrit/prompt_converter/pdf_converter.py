# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
from io import BytesIO
from typing import Optional

from fpdf import FPDF

from pyrit.common.logger import logger
from pyrit.models import PromptDataType, SeedPrompt, data_serializer_factory
from pyrit.prompt_converter import ConverterResult, PromptConverter


class PDFConverter(PromptConverter):
    """
    Converts a text prompt into a PDF file. Supports two modes:
    1. **Template-Based Generation**: If a `SeedPrompt` is provided, dynamic data can be injected into the
       template using the `SeedPrompt.render_template_value` method, and the resulting content is converted to a PDF.
    2. **Direct Text-Based Generation**: If no template is provided, the raw string prompt is converted directly
       into a PDF.

    Args:
        prompt_template (Optional[SeedPrompt], optional): A `SeedPrompt` object representing a template.
        font_type (Optional[str], optional): Font type for the PDF. Defaults to "Arial".
        font_size (Optional[int], optional): Font size for the PDF. Defaults to 12.
        page_width (Optional[int], optional): Width of the PDF page in mm. Defaults to 210 (A4 width).
        page_height (Optional[int], optional): Height of the PDF page in mm. Defaults to 297 (A4 height).
        column_width (Optional[int], optional): Width of each column in the PDF. Defaults to 0 (full page width).
        row_height (Optional[int], optional): Height of each row in the PDF. Defaults to 10.
    """

    def __init__(
        self,
        prompt_template: Optional[SeedPrompt] = None,
        font_type: Optional[str] = "Arial",
        font_size: Optional[int] = 12,
        page_width: Optional[int] = 210,
        page_height: Optional[int] = 297,
        column_width: Optional[int] = 0,
        row_height: Optional[int] = 10,
    ) -> None:
        self._prompt_template = prompt_template
        self._font_type = font_type
        self._font_size = font_size
        self._page_width = page_width
        self._page_height = page_height
        self._column_width = column_width
        self._row_height = row_height

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt into a PDF. If a template is provided, it injects the prompt into the template,
        otherwise, it generates a simple PDF with the prompt as the content.

        Args:
            prompt (str): The prompt to be embedded in the PDF.
            input_type (PromptDataType): The type of the input data (default: "text").

        Returns:
            ConverterResult: The result containing the full file path to the generated PDF.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Step 1: Prepare content
        content = self._prepare_content(prompt)

        # Step 2: Generate PDF
        pdf_bytes = self._generate_pdf(content)

        # Step 3: Serialize PDF
        pdf_serializer = await self._serialize_pdf(pdf_bytes, content)

        # Return the result
        return ConverterResult(output_text=pdf_serializer.value, output_type="url")

    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by the converter.

        Args:
            input_type (PromptDataType): The type of the input.

        Returns:
            bool: True if the input type is supported, False otherwise.
        """
        return input_type == "text"

    def _prepare_content(self, prompt: str) -> str:
        """
        Prepares the content for the PDF, either from a template or directly from the prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The prepared content.
        """
        if self._prompt_template:
            logger.debug(f"Preparing content with template: {self._prompt_template.value}")
            try:
                # Parse string prompt to dictionary
                dynamic_data = ast.literal_eval(prompt) if isinstance(prompt, str) else prompt
                logger.debug(f"Parsed dynamic data: {dynamic_data}")

                if not isinstance(dynamic_data, dict):
                    raise ValueError("Prompt must be a dictionary-compatible object after parsing.")

                # Use SeedPrompt's render_template_value for rendering
                rendered_content = self._prompt_template.render_template_value(**dynamic_data)
                logger.debug(f"Rendered content: {rendered_content}")
                return rendered_content

            except (ValueError, KeyError) as e:
                logger.error(f"Error rendering prompt: {e}")
                raise ValueError(f"Failed to render the prompt: {e}")

        # If no template is provided, return the raw prompt as content
        if isinstance(prompt, str):
            logger.debug("No template provided. Using raw prompt.")
            return prompt
        else:
            logger.error("Prompt must be a string when no template is provided.")
            raise ValueError("Prompt must be a string when no template is provided.")

    def _generate_pdf(self, content: str) -> BytesIO:
        """
        Generates the PDF content and writes it to a BytesIO buffer.

        Args:
            content (str): The text content to include in the PDF.

        Returns:
            BytesIO: The generated PDF in a BytesIO buffer.
        """
        pdf = FPDF(format=(self._page_width, self._page_height))  # Use custom page size
        pdf.add_page()
        pdf.set_font(self._font_type, size=self._font_size)  # Use custom font settings
        pdf.multi_cell(self._column_width, self._row_height, content)  # Use configurable cell dimensions

        pdf_bytes = BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes.seek(0)
        return pdf_bytes

    async def _serialize_pdf(self, pdf_bytes: BytesIO, content: str):
        """
        Serializes the generated PDF using a data serializer.

        Args:
            pdf_bytes (BytesIO): The BytesIO buffer containing the PDF.
            content (str): The original content of the PDF.

        Returns:
            DataTypeSerializer: The serializer object containing metadata about the saved file.
        """
        pdf_serializer = data_serializer_factory(
            category="prompt-memory-entries", data_type="url", value=content, extension="pdf"
        )
        await pdf_serializer.save_data(pdf_bytes.getvalue())
        return pdf_serializer
