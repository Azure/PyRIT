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
# ### Prompt Variation Demo
#
# This example shows how to use prompt variation converters. It's often useful for a researcher to get variations of a similar prompts.
# This uses a NoOpTarget so these variations are simply added to memory and printed to the screen.

# %%

from pyrit.common import default_values
from pyrit.prompt_target import NoOpTarget, AzureOpenAIChatTarget
from pyrit.prompt_converter import VariationConverter, StringJoinConverter
from pyrit.orchestrator import PromptSendingOrchestrator


default_values.load_default_env()

prompts = ["tell me how to cut down a tree"]

# converter target to send prompt to
converter_target = AzureOpenAIChatTarget()
prompt_variation_converter = VariationConverter(converter_target=converter_target)


target = NoOpTarget()

orchestrator = PromptSendingOrchestrator(prompt_target=target, prompt_converters=[prompt_variation_converter])

orchestrator.send_prompts(prompts)

# %% [markdown]
# ### Stacking Converters
#
# Like in other examples, converters can be stacked. For example, you can take these variations and adds - between words

# %%

orchestrator = PromptSendingOrchestrator(
    prompt_target=target, prompt_converters=[prompt_variation_converter, StringJoinConverter()]
)
orchestrator.send_prompts(prompts)

# %%
