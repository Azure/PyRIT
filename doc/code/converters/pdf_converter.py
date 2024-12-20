# %% [markdown]
# PDF Converter with PromptSendingOrchestrator
#
# This script demonstrates how to use the `PDFConverter` within the `PromptSendingOrchestrator` framework.
# It embeds prompts into PDF files using a Jinja2 template or generates PDFs directly from prompts.

# %%
from pyrit.common import default_values
from pyrit.prompt_target import TextTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter.pdf_converter import PDFConverter
from pyrit.memory import DuckDBMemory, CentralMemory
from pyrit.common.path import DATASETS_PATH

# %%
# Load environment variables
default_values.load_environment_files()

# Set up memory instance
memory_instance = DuckDBMemory()
CentralMemory.set_memory_instance(memory_instance)

# Define the path to the Jinja2 template
template_path = DATASETS_PATH / "prompt_templates/pdf/red_teaming_application_template.j2"

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
    template_path=str(template_path),  # Pass the template path as a string
    font_type="Arial",
    font_size=12,
    page_width=210,
    page_height=297,
)

# Pass the prompt data as a single JSON-formatted string to the converter
prompt_json = f"{prompt_data}"  # Converts the dictionary into a string representation for `prompt`

converters: list[PromptConverter] = [pdf_converter]
target = TextTarget()

# Use the PromptSendingOrchestrator to handle the PDF generation
with PromptSendingOrchestrator(objective_target=target, prompt_converters=converters) as orchestrator:
    await orchestrator.send_prompts_async(prompt_list=[prompt_json])  # type: ignore
