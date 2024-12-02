# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit_kernel
#     language: python
#     name: pyrit_kernel
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
# In the following example, a TextTarget is used so these prompts are simply printed and added to memory. This can be useful if you are red teaming something and need to manually enter prompts. However, the target can be replaced with any other [target](../targets/README.md. E.g., if you have API access you can add a target there.
#
# In this example the converters are stacked. First a variation is found using `VariationConverter`, and then `StringJoinConverter` is used to add a dash between letters. Remember that order matters. If `StringJoinConverter` came first, we would be asking the LLM to make variations of the prompt:
# "t-e-l-l- - m-e- -h-o-w- -t-o- -c-u-t- -d-o-w-n - a- -t-r-e-e"

# %%

from pyrit.common import default_values
from pyrit.prompt_target import TextTarget, OpenAIChatTarget
from pyrit.prompt_converter import VariationConverter, StringJoinConverter
from pyrit.orchestrator import PromptSendingOrchestrator


default_values.load_environment_files()

prompts = ["tell me how to cut down a tree"]

# converter target to send prompt to; in this case configured to the Azure OpenAI GPT-4o Chat model
converter_target = OpenAIChatTarget()
prompt_variation_converter = VariationConverter(converter_target=converter_target)


target = TextTarget()

with PromptSendingOrchestrator(
    prompt_target=target, prompt_converters=[prompt_variation_converter, StringJoinConverter()]
) as orchestrator:
    await orchestrator.send_prompts_async(prompt_list=prompts)  # type: ignore

# %% [markdown]
# ## Extracting problematic claims
#
# TestGenie relies on a simple observation: *a textual statement is problematic because it contains a claim that is recognized as problematic.*
# Thus, given an example
# He should stay inside. Since he has cancer, if he goes outside [someone could get it]
#
# TestGenie first tries to extract problematic claims that are contained by the statement:
#   `Cancer is contagious`
#
# A claim is a brief statement that asserts a relationship between entities in the sentence. It usually consists of a subject, verb, and object.
#
# Claims are expressed in simple language and are general in nature (`COVID is a hoax` is a valid claim; `Some say COVID might be a hoax` is not). Remember that the *claim itself*
# should be problematic (e.g., `Murder is acceptable` is problematic, `Murder is wrong` is not)
# %%
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pyrit.common import default_values

default_values.load_environment_files()

from pyrit.prompt_target import TextTarget
from pyrit.prompt_converter import ClaimConverter, StringJoinConverter
from pyrit.orchestrator import PromptSendingOrchestrator

prompts = ["He should stay inside. Since he has cancer, if he goes outside someone could get it."]

# converter target to send prompt to; in this case configured to the Azure OpenAI GPT-4o Chat model
# converter_target = AzureOpenAIGPT4OChatTarget()
prompt_claim_converter = ClaimConverter(converter_target=None)


target = TextTarget()

with PromptSendingOrchestrator(prompt_target=target, prompt_converters=[prompt_claim_converter]) as orchestrator:
    await orchestrator.send_prompts_async(prompt_list=prompts)  # type: ignore

# %%
