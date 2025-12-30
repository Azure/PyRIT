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
# Below is an example that converts a list of messages to chatml format and back.
# %%
import asyncio

from pyrit.message_normalizer import ChatMLNormalizer
from pyrit.models import Message

messages = [
    Message.from_prompt(prompt="You are a helpful AI assistant", role="system"),
    Message.from_prompt(prompt="Hello, how are you?", role="user"),
    Message.from_prompt(prompt="I'm doing well, thanks for asking.", role="assistant"),
]

normalizer = ChatMLNormalizer()
chatml_messages = asyncio.run(normalizer.normalize_string_async(messages))
# chatml_messages is a string in chatml format

print(chatml_messages)

# %% [markdown]
#
# If you wish you load a chatml-format conversation, you can use the `from_chatml` method in the `ChatMLNormalizer`. This will return a list of `Message` objects that you can then use.

# %%
chat_messages = normalizer.from_chatml(
    """\
    <|im_start|>system
    You are a helpful AI assistant<|im_end|>
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm doing well, thanks for asking.<|im_end|>"""
)

print(chat_messages)

# %% [markdown]
# To see how to use this in action, check out the [aml endpoint](../targets/3_non_open_ai_chat_targets.ipynb) notebook. It takes a `message_normalizer` parameter so that an AML model can support various chat message formats.

# %% [markdown]
# Besides chatml, there are many other chat templates that a model might be trained on. If you would like to apply the template stored in a Hugging Face tokenizer,
# you can utilize `TokenizerTemplateNormalizer`. In the example below, we load the tokenizer for Mistral-7B-Instruct-v0.1 and apply its chat template to
# the messages. Note that this template only adds `[INST]` and `[/INST]` tokens to the user messages for instruction fine-tuning.
# %%
import os

from transformers import AutoTokenizer

from pyrit.message_normalizer import TokenizerTemplateNormalizer

messages = [
    Message.from_prompt(prompt="Hello, how are you?", role="user"),
    Message.from_prompt(prompt="I'm doing well, thanks for asking.", role="assistant"),
    Message.from_prompt(prompt="What is your favorite food?", role="user"),
]

# Load the tokenizer. If you are not logged in via CLI (huggingface-cli login), you can pass in your access token here
# via the HUGGINGFACE_TOKEN environment variable to access the gated model.
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1", token=os.environ.get("HUGGINGFACE_TOKEN")
)

# create the normalizer and pass in the tokenizer
tokenizer_normalizer = TokenizerTemplateNormalizer(tokenizer=tokenizer)

tokenizer_template_messages = asyncio.run(tokenizer_normalizer.normalize_string_async(messages))
print(tokenizer_template_messages)
