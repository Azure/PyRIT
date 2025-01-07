# %% [markdown]
# # PDF Converter with Multiple Modes:
#
# This script demonstrates the use of the `PDFConverter` for generating PDFs in two different modes:
#
# - **Template-Based PDF Generation**: Utilize a YAML template to render dynamic content into a PDF.
# - **Direct Prompt PDF Generation**: Convert plain string prompts into PDFs without using a template.
#
# The `PromptSendingOrchestrator` is used to handle the interaction with the `PDFConverter` and the mock `TextTarget` target system.
#
# ## Key Features
#
# 1. **Template-Based Generation**:
#    - Populate placeholders in a YAML-based template using dynamic data.
#    - Convert the rendered content into a PDF using the `PDFConverter`.
#
# 2. **String-Based Generation**:
#    - Accept plain string prompts directly.
#    - Generate PDFs from the provided string without using any template.

# %%
import pathlib
from pyrit.common import default_values
from pyrit.prompt_target import TextTarget
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import PDFConverter
from pyrit.models import SeedPrompt
from pyrit.common.path import DATASETS_PATH

# Load default environment values (e.g., API keys, endpoints)
default_values.load_environment_files()

# Define dynamic data for injection
prompt_data = {
    "hiring_manager_name": "Jane Doe",
    "current_role": "AI Engineer",
    "company": "CyberDefense Inc.",
    "red_teaming_reason": "to creatively identify security vulnerabilities while enjoying free coffee",
    "applicant_name": "John Smith",
}

# Load the YAML template for the PDF generation
template_path = (
    pathlib.Path(DATASETS_PATH) / "prompt_converters" / "pdf_converters" / "red_teaming_application_template.yaml"
)
if not template_path.exists():
    raise FileNotFoundError(f"Template file not found: {template_path}")

# Load the SeedPrompt from the YAML file
prompt_template = SeedPrompt.from_yaml_file(template_path)

# Initialize the Azure OpenAI chat target (or mock target if not needed)
prompt_target = TextTarget()

# Initialize the PDFConverter
pdf_converter = PDFConverter(
    prompt_template=prompt_template,
    font_type="Arial",
    font_size=12,
    page_width=210,
    page_height=297,
)

# Define a single prompt for the orchestrator
prompts = [str(prompt_data)]

# Initialize the orchestrator
with PromptSendingOrchestrator(
    objective_target=prompt_target,  # Target system (Azure OpenAI or other LLM target)
    prompt_converters=[pdf_converter],  # Attach the PDFConverter
    verbose=False,  # Set to True for detailed logging
) as orchestrator:
    await orchestrator.send_prompts_async(prompt_list=prompts)  # type: ignore

    await orchestrator.print_conversations_async()  # type: ignore

# %% [markdown]
# # Direct Prompt PDF Generation (No Template)

# %%
# Define a simple string prompt (no templates)
prompt = "This is a simple test string for PDF generation. No templates here!"

# Initialize the TextTarget (mock target for testing)
prompt_target = TextTarget()

# Initialize the PDFConverter without a template
pdf_converter = PDFConverter(
    prompt_template=None,  # No template provided
    font_type="Arial",
    font_size=12,
    page_width=210,
    page_height=297,
)

# Define the list of prompts as strings (required by PromptSendingOrchestrator)
prompts = [prompt]

# Initialize the orchestrator
with PromptSendingOrchestrator(
    objective_target=prompt_target,
    prompt_converters=[pdf_converter],
    verbose=False,
) as orchestrator:
    await orchestrator.send_prompts_async(prompt_list=prompts)  # type: ignore

    await orchestrator.print_conversations_async()  # type: ignore

# %%
