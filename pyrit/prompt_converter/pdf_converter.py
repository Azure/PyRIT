# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from io import BytesIO
from typing import Optional

from fpdf import FPDF
from jinja2 import Template

from pyrit.memory import MemoryInterface
from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.prompt_converter import PromptConverter, ConverterResult


logger = logging.getLogger(__name__)


class PDFConverter(PromptConverter):
    """
    Converts a text prompt into a PDF file. If a base template is provided, the prompt is embedded into the template
    using Jinja2. If no template is provided, a new PDF is generated with the prompt as its content.

    Args:
        template_path (Optional[str], optional): Path to the Jinja2 template file. Defaults to None.
        memory (Optional[MemoryInterface], optional): Memory interface instance for data storage.
            If not provided, the default memory instance is used.
        font_type (Optional[str], optional): Font type for the PDF. Defaults to "Arial".
        font_size (Optional[int], optional): Font size for the PDF. Defaults to 12.
        page_width (Optional[int], optional): Width of the PDF page in mm. Defaults to 210 (A4 width).
        page_height (Optional[int], optional): Height of the PDF page in mm. Defaults to 297 (A4 height).
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        memory: Optional[MemoryInterface] = None,
        font_type: Optional[str] = "Arial",
        font_size: Optional[int] = 12,
        page_width: Optional[int] = 210,
        page_height: Optional[int] = 297,
    ) -> None:
        self._template_path = template_path
        self._memory = memory
        self._font_type = font_type
        self._font_size = font_size
        self._page_width = page_width
        self._page_height = page_height

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
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
        # Prepare the content
        if self._template_path:
            try:
                with open(self._template_path, "r") as file:
                    template = Template(file.read())
                    content = template.render(prompt=prompt)
            except FileNotFoundError:
                logger.error(f"Template file {self._template_path} not found.")
                raise
        else:
            content = prompt

        # Generate PDF content and write to buffer
        pdf = FPDF(format=(self._page_width, self._page_height))  # Use custom page size
        pdf.add_page()
        pdf.set_font(self._font_type, size=self._font_size)  # Use custom font settings
        pdf.multi_cell(0, 10, content)

        buffer = BytesIO()
        pdf.output(buffer)
        buffer.seek(0)

        # Use DataTypeSerializer to save the file
        serializer = data_serializer_factory(data_type="url", value=prompt, memory=self._memory)
        await serializer.save_data(buffer.getbuffer())

        # Return the result
        return ConverterResult(output_text=serializer.value, output_type="url")

    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by the converter.

        Args:
            input_type (PromptDataType): The type of the input.

        Returns:
            bool: True if the input type is supported, False otherwise.
        """
        return input_type == "text"
