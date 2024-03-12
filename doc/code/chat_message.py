# %% [markdown]
# ### Introduction
#
# This notebook gives an introduction to the concept of `ChatMessage` and `ChatMessageNormalizer` and how it can be helpful as you start to work with different models.
#
#
# The main format PyRIT works with is the `ChatMessage` paradigm. Any time a user wants to store or retrieve a chat message, they will use the `ChatMessage` object.
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
# To see how to use this in action, check out the [aml endpoint](./aml_endpoints.ipynb) notebook. It takes a `chat_message_normalizer` parameter so that an AML model can support various chat message formats.
