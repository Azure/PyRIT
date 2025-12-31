# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Chat messages - optional
#
# This notebook gives an introduction to the concept of `ChatMessage` and `MessageNormalizer` and how it can be helpful as you start to work with different models.
#
#
# The main format PyRIT works with is the `MessagePiece` paradigm. Any time a user wants to store or retrieve a chat message, they will use the `MessagePiece` object. However, `ChatMessage` is very common, so there are a lot of times this is the most useful. Any `MessagePiece` object can be translated into a `ChatMessage` object.
#
# However, different models may require different formats. For example, certain models may use chatml, or may not support system messages. This is handled
# in from `MessageNormalizer` and its subclasses.
#
# Below is an example that converts a list of messages to chatml format using `TokenizerTemplateNormalizer`.
# %%
import asyncio

from pyrit.message_normalizer import TokenizerTemplateNormalizer
from pyrit.models import Message

messages = [
    Message.from_prompt(prompt="You are a helpful AI assistant", role="system"),
    Message.from_prompt(prompt="Hello, how are you?", role="user"),
    Message.from_prompt(prompt="I'm doing well, thanks for asking.", role="assistant"),
]

# Use the "chatml" alias to get a normalizer that applies ChatML format
normalizer = TokenizerTemplateNormalizer.from_model("chatml")
chatml_messages = asyncio.run(normalizer.normalize_string_async(messages))
# chatml_messages is a string in chatml format

print(chatml_messages)

# %% [markdown]
# The `TokenizerTemplateNormalizer` supports various model aliases like "chatml", "llama3", "phi3", "mistral", and more.
# You can also pass a full HuggingFace model name to use its chat template.

# %% [markdown]
# To see how to use this in action, check out the [aml endpoint](../targets/3_non_open_ai_chat_targets.ipynb) notebook. It takes a `message_normalizer` parameter so that an AML model can support various chat message formats.

# %% [markdown]
# Besides chatml, there are many other chat templates that a model might be trained on. The `TokenizerTemplateNormalizer.from_model()` factory
# method can load the tokenizer for a HuggingFace model and apply its chat template. In the example below, we use the "mistral" alias.
# Note that this template only adds `[INST]` and `[/INST]` tokens to the user messages for instruction fine-tuning.
# %%
import os

from pyrit.message_normalizer import TokenizerTemplateNormalizer

messages = [
    Message.from_prompt(prompt="Hello, how are you?", role="user"),
    Message.from_prompt(prompt="I'm doing well, thanks for asking.", role="assistant"),
    Message.from_prompt(prompt="What is your favorite food?", role="user"),
]

# Use the "mistral" alias - requires HUGGINGFACE_TOKEN for gated models
tokenizer_normalizer = TokenizerTemplateNormalizer.from_model("mistral")

tokenizer_template_messages = asyncio.run(tokenizer_normalizer.normalize_string_async(messages))
print(tokenizer_template_messages)
