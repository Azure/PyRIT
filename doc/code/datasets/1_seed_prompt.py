# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Seed Prompts
#
# Most of the datasets we load into PyRIT are stored as a `SeedPromptDataset`. It is useful to store these results with the attributes and metadata associated with these prompts. In order to better understand the organization of the data, we will first start by defining what a `SeedPrompt` is. A `SeedPrompt` can either be a prompt template with parameters such as `{{gender}}` or just a prompt. By loading a YAML file as a `SeedPromptDataset`, we can use the prompts in other components within PyRIT.

# %%
import pathlib

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPromptDataset

initialize_pyrit(memory_db_type=IN_MEMORY)

seed_prompt_dataset = SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal.prompt")

print(seed_prompt_dataset.prompts[0])

# %% [markdown]
# There are many attributes stored in a `SeedPrompt` that is very useful for querying by fields such as `harm_categories`.

# %% [markdown]
# # Loading a Dataset as a `SeedPromptDataset`
#
# By managing `SeedPrompt` objects as a `SeedPromptDataset`, we can load prompts from other datasets. This example demonstrates the process of using XSTest Bias Testing examples to evaluate the safety and bias of an AI language model.

# %%
from pyrit.datasets import fetch_xstest_dataset

# Fetch examples from XSTest Bias Testing datase
xstest_dataset = fetch_xstest_dataset()

# Use the first 4 examples
prompt_list = xstest_dataset.get_values(first=4)
print(xstest_dataset.prompts[0])

# %% [markdown]
# By using `render_template_value`, we can put values into the template. The below example shows a simple use case of using the template in a jailbreak to include in the prompts.

# %%
from pyrit.models import SeedPrompt

jailbreak_template = SeedPrompt.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
)

jailbreak_prompt: str = jailbreak_template.render_template_value(
    prompt="Say the words 'I wish I could be the first golden retriever to go to outer space.'"
)

print(jailbreak_prompt)

# %% [markdown]
# # Multimodal use case with Seed Prompt Groups
#
# For multimodal cases where we want to send a piece of text and image together, the `SeedPromptGroup` abstraction can be used for groups of seed prompts. When a group of prompts need to be sent together, this class can support sending this datatype to a target where all the prompts share the same `prompt_group_id`. This is also useful for multi-turn conversations such as in Skeleton Key attack where the sequences are both fixed prompts.

# %%
from pyrit.models import SeedPromptGroup

image_path = pathlib.Path(".") / ".." / ".." / ".." / "assets" / "pyrit_architecture.png"

seed_prompt_group = SeedPromptGroup(
    prompts=[
        SeedPrompt(value="Describe the image in the image_path", data_type="text"),
        SeedPrompt(
            value=str(image_path),
            data_type="image_path",
        ),
    ]
)

print(seed_prompt_group.prompts)
