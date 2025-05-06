# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Using Prompt Converters
#
# Although converters can be used on their own, they should be thought of as a piece in the pipeine. Typically any orchestrator will have arguments so that prompts can be converted before being sent to the target. They can be stacked, use LLMs, and are a powerful tool.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).
#
# ## Stacking Converters with PromptSendingOrchestrator
#
# In the following example, a TextTarget is used so these prompts are simply printed and added to memory. This can be useful if you are red teaming something and need to manually enter prompts. However, the target can be replaced with any other [target](../targets/0_prompt_targets.md). E.g., if you have API access you can add a target there.
#
# In this example the converters are stacked. First a variation is found using `VariationConverter`, and then `StringJoinConverter` is used to add a dash between letters. Remember that order matters. If `StringJoinConverter` came first, we would be asking the LLM to make variations of the prompt:
# "t-e-l-l- - m-e- -h-o-w- -t-o- -c-u-t- -d-o-w-n - a- -t-r-e-e"

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import StringJoinConverter, VariationConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import OpenAIChatTarget, TextTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

objective = "tell me how to cut down a tree"

# converter target to send prompt to; in this case configured to the Azure OpenAI GPT-4o Chat model
converter_target = OpenAIChatTarget()
prompt_variation_converter = VariationConverter(converter_target=converter_target)

converters = PromptConverterConfiguration.from_converters(
    converters=[prompt_variation_converter, StringJoinConverter()]
)


target = TextTarget()
orchestrator = PromptSendingOrchestrator(objective_target=target, request_converter_configurations=converters)

await orchestrator.run_attack_async(objective=objective)  # type: ignore
