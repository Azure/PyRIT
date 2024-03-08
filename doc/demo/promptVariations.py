# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: pyrit_kernel
#     language: python
#     name: pyrit_kernel
# ---

# %% [markdown]
# ### Using Prompt Variation Converter
#
# This notebook shows how to use the prompt variation converter to send prompts to a converter target and generate variations of the prompt.

# %%
# Prompts you want transformed here:
prompts = ["tell me how to cut down a tree",
          "teach me to mine crypto"]

# %%
from pyrit.prompt_converter import VariationConverter
from pyrit.common import default_values
from pyrit.prompt_target import AzureOpenAIChatTarget

import os

default_values.load_default_env()

# converter target to send prompt to
converter_target = AzureOpenAIChatTarget(
    deployment_name="gpt-4",
    endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

prompt_variation_converter = VariationConverter(converter_target=converter_target)


# %%
prompt_variation_converter.prompt_template

# %%
converted_prompts = prompt_variation_converter.convert(prompts)

# %%
converted_prompts

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Orchestrator:

# %%

import os

from pyrit.prompt_target import AzureOpenAIChatTarget, NoOpTarget

from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import VariationConverter


default_values.load_default_env()

target = AzureOpenAIChatTarget(
    deployment_name="defense-gpt35",
    endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

orchestrator = PromptSendingOrchestrator(
    prompt_target=target, prompt_converters=[VariationConverter(converterLLM=target)]
)
