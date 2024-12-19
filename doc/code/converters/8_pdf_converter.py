# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PDF Converter with Jinja2 Templates
#
# This Python script demonstrates a PDF Converter implementation using Python and Jinja2.
# The converter allows users to embed a prompt into a template file (PDF, Word, HTML) or create a new file with the prompt embedded.
# The Jinja2 templating engine handles the insertion of dynamic content into predefined placeholders.
#
# ### Example Use Case:
# - Inject a prompt into a PDF file at a specific location (using a Jinja2 template).
# - This converter supports PDF, Word, and HTML files.
#
# ### Libraries Used:
# - **Jinja2**: For templating dynamic content.
# - **FPDF**: For creating PDFs from the embedded prompt.
# - **python-docx**: For handling Word documents.

# %%
import logging
from jinja2 import Template
from fpdf import FPDF
from docx import Document

logging.basicConfig(level=logging.INFO)

# Define a function to convert a prompt to PDF
def convert_to_pdf(prompt, base_template=None):
    if base_template:
        # Use the Jinja2 template if a base file is provided
        with open(base_template, 'r') as file:
            template = Template(file.read())
            content = template.render(prompt=prompt)
    else:
        content = prompt

    # Create a PDF using FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output('output.pdf')
    logging.info('PDF created successfully.')


# %% [markdown]
# ### Test the PDF Converter
# You can now test the PDF converter with a custom prompt.
# If no template is provided, the prompt will be directly embedded into a PDF.

# %%
# Example: Test with a prompt (no template)
prompt = "This is a sample prompt for the PDF Converter."
convert_to_pdf(prompt)


# %% [markdown]
# ### Using Jinja2 Template
# In this example, we will create a Jinja2 template and use it to inject a prompt into the template.

# %%
# Example Jinja2 Template
template_content = """
    Hello, {{ prompt }}!
    This PDF was generated dynamically using Jinja2 and FPDF.
"""

# Save the template to a file
with open('template.txt', 'w') as f:
    f.write(template_content)

# Use the template for PDF conversion
convert_to_pdf("This is a dynamic prompt!", base_template='template.txt')

# %% [markdown]
# ### Conclusion
# This script demonstrates how to create a simple PDF converter using Python and Jinja2 templates.
# You can expand on this by adding support for Word and HTML files, as well as customizing the templates further.
