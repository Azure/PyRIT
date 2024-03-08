#!/usr/bin/env python
# coding: utf-8

# ### Introduction
#
# This Jupyter notebook gives an introduction to the concept of `ChatMessagesDataset` and how they can be helpful as you start to work with more and more data.
#
# ### What is ChatMessagesDataset?
#
# The chat messages dataset is a way to store information about multi-turn conversations in a structured way. It is a collection of `ChatMessage` that can be used to train and evaluate models for conversational AI. Although can also he used for other purposes, such as storing data for chatbots, or for analyzing conversations later on.
#
# The added benefits of using `ChatMessagesDataset` are that you can more easily manage and manipulate the data, and you can use the built-in functionality that `pydantic` offers to validate the data and ensure that it is in the correct format. Additionally, you can use the `name` and `description` fields to store metadata about the dataset, such as where it came from, who created it, and what it is used for.
#
#
# Below is an example of how to create a `ChatMessagesDataset` and add some `ChatMessage` to it.

# In[1]:


from pyrit.models import ChatMessagesDataset, ChatMessage

dataset = ChatMessagesDataset(
    name="demo-dataset",
    description="dataset for demo purposes",
    list_of_chat_messages=[
        [
            ChatMessage(role="system", content="You are a helpful AI assistant"),
            ChatMessage(role="user", content="Hello, how are you?"),
            ChatMessage(role="assistant", content="I'm doing well, thank you for asking."),
        ],
        [
            ChatMessage(role="system", content="You are a helpful AI assistant"),
            ChatMessage(role="user", content="How are you?"),
            ChatMessage(
                role="assistant",
                content="I'm just a computer program, so I don't have feelings, but thank you for asking! How can I assist you today?.",
            ),
        ],
    ],
)

print(dataset.model_dump(exclude_none=True))


# another thing you can do with the `ChatMessagesDataset` is to save it to a file and load it from a file. This is useful for storing the data in a more permanent way, and for sharing it with others.
#
# Below is an example of how to load a `ChatMessagesDataset` after it has been read from a file.

# In[11]:


data = {
    "name": "demo-dataset",
    "description": "dataset for demo purposes",
    "list_of_chat_messages": [
        [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks for asking."},
        ],
        [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Good, you?"},
        ],
    ],
}

my_dataset = ChatMessagesDataset.model_validate(data)
my_dataset


# Another things that datasets allow is the easily conversion to other formats. For example, the `list_of_chat_messages` field in the `ChatMessagesDataset` is of type `list[ChatMessages]`. This means that it is compatible with the different `ChatMessageNormalizer` derived objects. To normalize to [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) you can use `ChatMessageNormalizerChatML` normalizer.

# In[3]:


from pyrit.chat_message_normalizer import ChatMessageNormalizerChatML

normalizer = ChatMessageNormalizerChatML()
print(normalizer.normalize(my_dataset.list_of_chat_messages[0]))


# If you wish you load a chatml-format conversation, you can use the `from_chatml` method in the `ChatMessageNormalizerChatML`. This will return a list of `ChatMessage` objects that you can then use.

# In[4]:


normalizer.from_chatml(
    """\
    <|im_start|>system
    You are a helpful AI assistant<|im_end|>
    <|im_start|>user
    Hello, how are you?<|im_end|>
    <|im_start|>assistant
    I'm doing well, thanks for asking.<|im_end|>"""
)
