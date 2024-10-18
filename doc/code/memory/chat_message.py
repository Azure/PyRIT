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
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Introduction
#
# This notebook gives an introduction to the concept of `ChatMessage` and `ChatMessageNormalizer` and how it can be helpful as you start to work with different models.
#
#
# The main format PyRIT works with is the `PromptRequestPiece` paradigm. Any time a user wants to store or retrieve a chat message, they will use the `PromptRequestPiece` object. However, `ChatMessage` is very common, so there are a lot of times this is the most useful. Any `PromptRequestPiece` object can be translated into a `ChatMessage` object.
#
# However, different models may require different formats. For example, certain models may use chatml, or may not support system messages. This is handled
# in from `ChatMessageNormalizer` and its subclasses.
#
# Below is an example that converts a list of chat messages to chatml format and back.
# %%
from pyrit.models import ChatMessage
from pyrit.chat_message_normalizer import ChatMessageNormalizerChatML

messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant"),
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thanks for asking."),
]

normalizer = ChatMessageNormalizerChatML()
chatml_messages = normalizer.normalize(messages)
# chatml_messages is a string in chatml format

print(chatml_messages)

# %% [markdown]
#
# If you wish you load a chatml-format conversation, you can use the `from_chatml` method in the `ChatMessageNormalizerChatML`. This will return a list of `ChatMessage` objects that you can then use.

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
# To see how to use this in action, check out the [aml endpoint](../targets/4_non_open_ai_chat_targets.ipynb) notebook. It takes a `chat_message_normalizer` parameter so that an AML model can support various chat message formats.

# %% [markdown]
# Besides chatml, there are many other chat templates that a model might be trained on. If you would like to apply the template stored in a Hugging Face tokenizer,
# you can utilize `ChatMessageNormalizerTokenizerTemplate`. In the example below, we load the tokenizer for Mistral-7B-Instruct-v0.1 and apply its chat template to
# the messages. Note that this template only adds `[INST]` and `[/INST]` tokens to the user messages for instruction fine-tuning.

# %%
from pyrit.chat_message_normalizer import ChatMessageNormalizerTokenizerTemplate
from transformers import AutoTokenizer

messages = [
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thanks for asking."),
    ChatMessage(role="user", content="What is your favorite food?"),
]

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# create the normalizer and pass in the tokenizer
tokenizer_normalizer = ChatMessageNormalizerTokenizerTemplate(tokenizer)

tokenizer_template_messages = tokenizer_normalizer.normalize(messages)
print(tokenizer_template_messages)
