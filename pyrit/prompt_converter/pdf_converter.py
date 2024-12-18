# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional
from jinja2 import Template
from fpdf import FPDF
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult

logger = logging.getLogger(__name__)

class PDFConverter(PromptConverter):
    """
    Converts a text prompt into a PDF file. If a base template is provided, the prompt is embedded into the template
    using Jinja2. If no template is provided, a new PDF is generated with the prompt as its content.

    Args:
        template_path (Optional[str], optional): Path to the Jinja2 template file. Defaults to None.
    """

    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt into a PDF. If a template is provided, it injects the prompt into the template,
        otherwise, it generates a simple PDF with the prompt as the content.

        Args:
            prompt (str): The prompt to be embedded in the PDF.
            input_type (PromptDataType): The type of the input data (default: "text").

        Returns:
            ConverterResult: The result containing the filename of the generated PDF.
        """
        output_file = "output.pdf"
        if self.template_path:
            try:
                with open(self.template_path, 'r') as file:
                    template = Template(file.read())
                    content = template.render(prompt=prompt)
            except FileNotFoundError:
                logger.error(f"Template file {self.template_path} not found.")
                raise
        else:
            content = prompt

        # Create the PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        pdf.output(output_file)

        logger.info(f"PDF created successfully: {output_file}")
        return ConverterResult(output_text=output_file, output_type="pdf")

    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by the converter.

        Args:
            input_type (PromptDataType): The type of the input.

        Returns:
            bool: True if the input type is supported, False otherwise.
        """
        return input_type == "text"

