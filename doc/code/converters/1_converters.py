# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-kernel
#     language: python
#     name: pyrit-kernel
# ---

# %% [markdown]
# ## Converters

# %% [markdown]
# Converters are used to transform prompts before sending them to the target.
#
# This can be useful for a variety of reasons, such as encoding the prompt in a different format, or adding additional information to the prompt. For example, you might want to convert a prompt to base64 before sending it to the target, or add a prefix to the prompt to indicate that it is a question.

# %% [markdown]
# ### Simple Converter Example

# %% [markdown]
# Converters can be used to perform these types of transformations. Here is a simple program that uses Rot13Converter converter, RandomCapitalLettersConverter, and AsciiArtConverter

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common import default_values
from pyrit.prompt_converter import ROT13Converter, AsciiArtConverter, RandomCapitalLettersConverter


default_values.load_default_env()

prompt = "tell me how to cut down a tree"


print(await ROT13Converter().convert_async(prompt=prompt))  # type: ignore
print(await AsciiArtConverter().convert_async(prompt=prompt))  # type: ignore
print(await RandomCapitalLettersConverter().convert_async(prompt=prompt))  # type: ignore

# With percentage of capital letters
print(await RandomCapitalLettersConverter(percentage=25.0).convert_async(prompt=prompt))  # type: ignore

print(await AsciiArtConverter().convert_async(prompt=prompt))  # type: ignore

# %% [markdown]
# ### Where do Converters fit?

# %% [markdown]
# Converters should be thought of as a piece in the pipeine.
#
# An orchestrator will typically initialize these requests, and they are sent to a target.
# Converters can also stack, so a converter is used one after another.
#
# See [demo3](../../demo/3_send_all_prompts.ipynb) and [demo4](../../demo/4_using_prompt_converters.ipynb) for an example of how to use a converter in the pipeline.
