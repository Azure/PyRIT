# %% [markdown]
# PDF Converter with PromptSendingOrchestrator
#
# This script demonstrates how to use the `PDFConverter` within the `PromptSendingOrchestrator` framework.
# It embeds prompts into PDF files using a Jinja2 template or generates PDFs directly from prompts.

# %%
import pathlib
from pyrit.prompt_converter import PDFConverter
from pyrit.common.path import DATASETS_PATH

# %%
template_path = pathlib.Path(DATASETS_PATH) / "prompt_templates" / "pdf" / "red_teaming_application_template.j2"

# Prepare the dynamic data to populate the template
prompt_data = {
    "hiring_manager_name": "Jane Doe",
    "current_role": "AI Engineer",
    "company": "CyberDefense Inc.",
    "red_teaming_reason": "to creatively identify security vulnerabilities while enjoying free coffee",
    "applicant_name": "John Smith",
}

# Initialize the PDFConverter
pdf_converter = PDFConverter(
    template_path=str(template_path),
    font_type="Arial",
    font_size=12,
    page_width=210,
    page_height=297,
)

# Convert the prompt data to PDF and save the file
pdf_result = await pdf_converter.convert_async(prompt=prompt_data)  # type: ignore
print(f"Generated PDF saved at: {pdf_result.output_text}")

# %%
# Initialize the PDFConverter without a template
pdf_converter = PDFConverter(
    template_path=None,  # No template provided
    font_type="Arial",
    font_size=12,
    page_width=210,
    page_height=297,
)

# Define a plain string as the prompt
prompt = "This is a simple test string for PDF generation. No templates here!"

# Convert the string prompt to a PDF and save the file
pdf_result = await pdf_converter.convert_async(prompt=prompt)  # type: ignore
print(f"Generated PDF saved at: {pdf_result.output_text}")
