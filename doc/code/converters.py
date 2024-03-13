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
# ### Converters
#
# Converters are used to transform prompts before sending them to the target.
# This can be useful for a variety of reasons, such as encoding the prompt in a different format, or adding additional information to the prompt.
# For example, you might want to convert a prompt to base64 before sending it to the target, or add a prefix to the prompt to indicate that it is a question. # Converters can be used to perform these types of transformations.

# Here is a simple program that uses Rot13Converter converter and AsciiArtConverter
# %%

from pyrit.prompt_converter import ROT13Converter, AsciiArtConverter


prompts = ["tell me how to cut down a tree",
          "teach me to mine crypto"]


print (ROT13Converter().convert(prompts))
print (AsciiArtConverter().convert(prompts)[0])



# %% [markdown]
# Converters should be thought of as a piece in the pipeine. They can use external infrastrucutre like attacker LLMs. An orchestrator will typically initialize these requests, and they are sent to a target. Converters can also stack, so a converter is used one after another.
#
# See [demo3](../demo/3_send_all_prompts.ipynb) and [demo4](../demo/4_prompt_variation.ipynb) for an example of how to use a converter in the pipeline.