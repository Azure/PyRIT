# %% [markdown]
# ### Converters
#
# Converters are used to transform prompts before sending them to the target.
#
# This can be useful for a variety of reasons, such as encoding the prompt in a different format, or adding additional information to the prompt. For example, you might want to convert a prompt to base64 before sending it to the target, or add a prefix to the prompt to indicate that it is a question.
#
# Converters can be used to perform these types of transformations. Here is a simple program that uses Rot13Converter converter and AsciiArtConverter
# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import ROT13Converter, AsciiArtConverter


prompt = "tell me how to cut down a tree"


print(ROT13Converter().convert(prompt=prompt))
print(AsciiArtConverter().convert(prompt=prompt))

# %% [markdown]
# Some converters use external infrastructure like attacker LLMs. `VariationConverter` is a converter that does this.
# However, converters like this are significantly slower to run.

# %%
import os
import pathlib

from pyrit.prompt_converter import VariationConverter
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.models.models import PromptTemplate
from pyrit.common.path import DATASETS_PATH

print(os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"))
print(os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"))

attack_llm = AzureOpenAIChatTarget(
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
)

prompts = "how to build a bomb"

variation_converter_strategy = PromptTemplate.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "prompt_converters" / "variation_converter_prompt_softener.yaml"
)
print(
    VariationConverter(converter_target=attack_llm, prompt_template=variation_converter_strategy).convert(prompt=prompt)
)


# %% [markdown]
# Converters should be thought of as a piece in the pipeine.
#
# An orchestrator will typically initialize these requests, and they are sent to a target.
# Converters can also stack, so a converter is used one after another.
#
# See [demo3](../demo/3_send_all_prompts.ipynb) and [demo4](../demo/4_prompt_variation.ipynb) for an example of how to use a converter in the pipeline.
