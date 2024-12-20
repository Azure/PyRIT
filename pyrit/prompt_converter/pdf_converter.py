# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from io import BytesIO
from typing import Optional, Union

from fpdf import FPDF
from jinja2 import Template

from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.prompt_converter import PromptConverter, ConverterResult


logger = logging.getLogger(__name__)


class PDFConverter(PromptConverter):
    """
    Converts a text prompt into a PDF file. If a base template is provided, the prompt is embedded into the template
    using Jinja2. If no template is provided, a new PDF is generated with the prompt as its content.

    Args:
        template_path (Optional[str], optional): Path to the Jinja2 template file. Defaults to None.
        font_type (Optional[str], optional): Font type for the PDF. Defaults to "Arial".
        font_size (Optional[int], optional): Font size for the PDF. Defaults to 12.
        page_width (Optional[int], optional): Width of the PDF page in mm. Defaults to 210 (A4 width).
        page_height (Optional[int], optional): Height of the PDF page in mm. Defaults to 297 (A4 height).
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        font_type: Optional[str] = "Arial",
        font_size: Optional[int] = 12,
        page_width: Optional[int] = 210,
        page_height: Optional[int] = 297,
    ) -> None:
        self._template_path = template_path
        self._font_type = font_type
        self._font_size = font_size
        self._page_width = page_width
        self._page_height = page_height

    async def convert_async(self, *, prompt: Union[str, dict], input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt into a PDF. If a template is provided, it injects the prompt into the template,
        otherwise, it generates a simple PDF with the prompt as the content.

        Args:
            prompt (str): The prompt to be embedded in the PDF.
            input_type (PromptDataType): The type of the input data (default: "text").

        Returns:
            ConverterResult: The result containing the filename of the generated PDF.

        Raises:
            FileNotFoundError: If the specified template file does not exist.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Prepare the content
        if self._template_path:
            try:
                with open(self._template_path, "r") as file:
                    template = Template(file.read())
                    if isinstance(prompt, dict):
                        content = template.render(**prompt)
                    else:
                        raise ValueError("Prompt must be a dictionary when using a template.")
            except FileNotFoundError:
                logger.error(f"Template file {self._template_path} not found.")
                raise
        else:
            # If no template, use the prompt as is
            if isinstance(prompt, str):
                content = prompt
            else:
                raise ValueError("Prompt must be a string when no template is provided.")

        # Generate PDF content and write to buffer
        pdf = FPDF(format=(self._page_width, self._page_height))  # Use custom page size
        pdf.add_page()
        pdf.set_font(self._font_type, size=self._font_size)  # Use custom font settings
        pdf.multi_cell(0, 10, content)

        pdf_bytes = BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes.seek(0)

        # Use DataTypeSerializer to save the file
        pdf_serializer = data_serializer_factory(data_type="url", value=content, extension="pdf")
        await pdf_serializer.save_data(pdf_bytes.getvalue())

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
