# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Converters

# %% [markdown]
# Converters are used to transform prompts before sending them to the target.
#
# This can be useful for a variety of reasons, such as encoding the prompt in a different format, or adding additional information to the prompt. For example, you might want to convert a prompt to base64 before sending it to the target, or add a prefix to the prompt to indicate that it is a question.

# %% [markdown]
# ## Simple Converter Example

# %% [markdown]
# Converters can be used to perform these types of transformations. Here is a simple program that uses Rot13Converter converter, RandomCapitalLettersConverter, and AsciiArtConverter.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_converter import (
    AsciiArtConverter,
    BinaryConverter,
    RandomCapitalLettersConverter,
    ROT13Converter,
)

initialize_pyrit(memory_db_type=IN_MEMORY)

prompt = "tell me how to cut down a tree"


print(await ROT13Converter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await RandomCapitalLettersConverter(percentage=25.0).convert_tokens_async(prompt=prompt))  # type: ignore
print(await AsciiArtConverter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await BinaryConverter().convert_tokens_async(prompt=prompt))  # type: ignore


# %%
# Close connection
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
