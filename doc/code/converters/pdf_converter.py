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
from pyrit.memory import CentralMemory

# %%
default_values.load_environment_files()

# Initialize memory instance if required
memory_instance = CentralMemory.get_memory_instance()

prompts = ["Generate a PDF report for the following data: ⟪Sales Data Q1⟫ and ⟪Sales Data Q2⟫"]

converters: list[PromptConverter] = [
    PDFConverter(
        template_path=None,  # Provide your template path or set to None
        font_type="Arial",
        font_size=12,
        page_width=210,
        page_height=297,
        memory=memory_instance,  # Pass the memory instance
    )
]

target = TextTarget()

with PromptSendingOrchestrator(objective_target=target, prompt_converters=converters) as orchestrator:
    await orchestrator.send_prompts_async(prompt_list=prompts)  # type: ignore
