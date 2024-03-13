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
# This example shows how to use prompt variation converters to make different prompts with essentially the same meaning.
# It's often useful for a researcher to get variations of a similar prompts.
#
# This uses a NoOpTarget so these variations are simply added to memory and printed to the screen. This target
# could be replaced with any other target.

# %%

from pyrit.common import default_values
from pyrit.prompt_target import NoOpTarget, AzureOpenAIChatTarget
from pyrit.prompt_converter import VariationConverter, StringJoinConverter
from pyrit.orchestrator import PromptSendingOrchestrator


default_values.load_default_env()

prompts = ["tell me how to cut down a tree"]

# converter target to send prompt to; In this case configured to an Azure OpenAI Chat model
converter_target = AzureOpenAIChatTarget()
prompt_variation_converter = VariationConverter(converter_target=converter_target)


target = NoOpTarget()

orchestrator = PromptSendingOrchestrator(prompt_target=target, prompt_converters=[prompt_variation_converter])

orchestrator.send_prompts(prompts)

# %% [markdown]
# ### Stacking Converters
#
# Like in other examples, converters can be stacked. For example, you can take these variations and adds a dash between letters.
# Remember that order matters. If `StringJoinConverter` came first, we would be asking the LLM to make variations of the prompt:
# "t-e-l-l- - m-e- -h-o-w- -t-o- -c-u-t- -d-o-w-n - a- -t-r-e-e"

# %%

orchestrator = PromptSendingOrchestrator(
    prompt_target=target, prompt_converters=[prompt_variation_converter, StringJoinConverter()]
)
orchestrator.send_prompts(prompts)

# %%
