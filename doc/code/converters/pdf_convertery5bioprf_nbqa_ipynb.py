# %%NBQA-CELL-SEP52c935
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack import (
    AttackConverterConfig,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.models import SeedPrompt
from pyrit.prompt_converter import PDFConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import TextTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

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
pdf_converter = PromptConverterConfiguration.from_converters(
    converters=[
        PDFConverter(
            prompt_template=prompt_template,
            font_type="Arial",
            font_size=12,
            page_width=210,
            page_height=297,
        )
    ]
)

converter_config = AttackConverterConfig(
    request_converters=pdf_converter,
)

# Define a single prompt for the attack
prompt = str(prompt_data)

# Initialize the attack
attack = PromptSendingAttack(
    objective_target=prompt_target,  # Target system (Azure OpenAI or other LLM target)
    attack_converter_config=converter_config,  # Attach the PDFConverter
)

result = await attack.execute_async(objective=prompt)  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %%NBQA-CELL-SEP52c935
# Define a simple string prompt (no templates)
prompt = "This is a simple test string for PDF generation. No templates here!"

# Initialize the TextTarget (mock target for testing)
prompt_target = TextTarget()

# Initialize the PDFConverter without a template
pdf_converter = PromptConverterConfiguration.from_converters(
    converters=[
        PDFConverter(
            prompt_template=None,  # No template provided
            font_type="Arial",
            font_size=12,
            page_width=210,
            page_height=297,
        )
    ]
)

converter_config = AttackConverterConfig(
    request_converters=pdf_converter,
)

# Initialize the attack
attack = PromptSendingAttack(
    objective_target=prompt_target,
    attack_converter_config=converter_config,
)

result = await attack.execute_async(objective=prompt)  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %%NBQA-CELL-SEP52c935
import tempfile
from pathlib import Path

import requests

from pyrit.prompt_converter import PDFConverter
from pyrit.prompt_target import TextTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# This file isn't copied to our pipeline
url = "https://raw.githubusercontent.com/Azure/PyRIT/main/pyrit/datasets/prompt_converters/pdf_converters/fake_CV.pdf"

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    response = requests.get(url)
    tmp_file.write(response.content)

cv_pdf_path = Path(tmp_file.name)

# Define injection items
injection_items = [
    {
        "page": 0,
        "x": 50,
        "y": 700,
        "text": "Injected Text",
        "font_size": 12,
        "font": "Helvetica",
        "font_color": (255, 0, 0),
    },  # Red text
    {
        "page": 1,
        "x": 100,
        "y": 600,
        "text": "Confidential",
        "font_size": 10,
        "font": "Helvetica",
        "font_color": (0, 0, 255),
    },  # Blue text
]

# Define a simple string prompt (no templates)
prompt = "This is a simple test string for PDF generation. No templates here!"

# Initialize the TextTarget (mock target for testing)
prompt_target = TextTarget()

# Initialize the PDFConverter with the existing PDF and injection items
pdf_converter = PromptConverterConfiguration.from_converters(
    converters=[
        PDFConverter(
            prompt_template=None,  # No template provided
            font_type="Arial",
            font_size=12,
            page_width=210,
            page_height=297,
            existing_pdf=cv_pdf_path,  # Provide the existing PDF
            injection_items=injection_items,  # Provide the injection items
        )
    ]
)

converter_config = AttackConverterConfig(
    request_converters=pdf_converter,
)

# Initialize the attack
attack = PromptSendingAttack(
    objective_target=prompt_target,
    attack_converter_config=converter_config,
)

result = await attack.execute_async(objective=prompt)  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore
