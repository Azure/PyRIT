# %% [markdown]
# PDF Converter with Multiple Templates
#
# This script demonstrates how to use the `PDFConverter` to generate PDFs with supported templates.
# The `PDFConverter` supports dynamic data injection into text-based templates (e.g., `.j2`, `.txt`) or creates simple PDFs directly from string prompts.
#
# Key Features:
# - **Template-Based PDF Generation**: Populate placeholders in `.j2` or `.txt` templates with dynamic data and convert them into PDFs.
# - **Direct Prompt PDF Generation**: Generate PDFs from plain string prompts without using a template.
#
# **Note**: HTML and Markdown templates are not supported due to limitations in rendering by the FPDF library.

# %%
import pathlib
from pyrit.prompt_converter import PDFConverter
from pyrit.common.path import DATASETS_PATH

# %% Templates and Dynamic Data
template_folder = pathlib.Path(DATASETS_PATH) / "prompt_templates" / "pdf"
template_files = [
    template_folder / "red_teaming_application_template.j2",
    template_folder / "red_teaming_application_template.txt",
]

# Dynamic data for the templates
prompt_data = {
    "hiring_manager_name": "Jane Doe",
    "current_role": "AI Engineer",
    "company": "CyberDefense Inc.",
    "red_teaming_reason": "to creatively identify security vulnerabilities while enjoying free coffee",
    "applicant_name": "John Smith",
}

# %% Generate PDFs for Each Template
for template_file in template_files:
    print(f"\nUsing template: {template_file}")
    if template_file.exists():
        # Initialize the PDFConverter with the current template
        pdf_converter = PDFConverter(
            template_path=str(template_file),
            font_type="Arial",
            font_size=12,
            page_width=210,
            page_height=297,
        )

        # Generate the PDF
        pdf_result = await pdf_converter.convert_async(prompt=prompt_data)  # type: ignore
        print(f"Generated PDF saved at: {pdf_result.output_text}")
    else:
        print(f"Template file not found: {template_file}")

# %% Direct Prompt PDF Generation (No Template)
prompt = "This is a simple test string for PDF generation. No templates here!"

# Initialize the PDFConverter without a template
pdf_converter = PDFConverter(
    template_path=None,  # No template provided
    font_type="Arial",
    font_size=12,
    page_width=210,
    page_height=297,
)

# Generate the PDF from a plain string
pdf_result = await pdf_converter.convert_async(prompt=prompt)  # type: ignore
print(f"Generated PDF saved at: {pdf_result.output_text}")
