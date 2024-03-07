#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# This Jupyter notebook gives an introduction to the concept of `ChatMessagesDataset` and how they can be helpful as you start to work with more and more data.
# 
# ### What is ChatMessagesDataset?
# 
# The chat messages dataset is a way to store information about multi-turn conversations in a structured way. It is a collection of `ChatMessages` that can be used to train and evaluate models for conversational AI. Although can also he used for other purposes, such as storing data for chatbots, or for analyzing conversations later on.
# 
# The added benefits of using `ChatMessagesDataset` are that you can more easily manage and manipulate the data, and you can use the built-in functionality that `pydantic` offers to validate the data and ensure that it is in the correct format. Additionally, you can use the `name` and `description` fields to store metadata about the dataset, such as where it came from, who created it, and what it is used for.
# 
# 
# Below is an example of how to create a `ChatMessagesDataset` and add some `ChatMessages` to it.

# In[5]:


from pyrit.models import ChatMessagesDataset, ChatMessages, ChatMessage

dataset = ChatMessagesDataset(
    name="demo-dataset",
    description="dataset for demo purposes",
    list_of_chat_messages=[
        ChatMessages(messages=[
            ChatMessage(role="system", content="You are a helpful AI assistant"),
            ChatMessage(role="user", content="Hello, how are you?"),
            ChatMessage(role="assistant", content="I'm doing well, thank you for asking."),
        ]),
        ChatMessages(messages=[
            ChatMessage(role="system", content="You are a helpful AI assistant"),
            ChatMessage(role="user", content="How are you?"),
            ChatMessage(role="assistant", content="I'm just a computer program, so I don't have feelings, but thank you for asking! How can I assist you today?."),
        ]),
    ]
)

print(dataset.model_dump(exclude_none=True))


# another thing you can do with the `ChatMessagesDataset` is to save it to a file and load it from a file. This is useful for storing the data in a more permanent way, and for sharing it with others.
# 
# Below is an example of how to load a `ChatMessagesDataset` after it has been read from a file.

# In[9]:


data = {
  "name": "demo-dataset",
  "description": "dataset for demo purposes",
  "list_of_chat_messages": [
    {
      "messages": [
        { "role": "system", "content": "You are a helpful AI assistant" },
        { "role": "user", "content": "Hello, how are you?" },
        { "role": "assistant", "content": "I'm doing well, thanks for asking." }
      ]
    },
    {
      "messages": [
        { "role": "system", "content": "You are a helpful AI assistant" },
        { "role": "user", "content": "How are you?" },
        { "role": "assistant", "content": "Good, you?" }
      ]
    }
  ]
}

my_dataset = ChatMessagesDataset(**data)
my_dataset


# Another things that datasets allow is the easily conversion to other formats. For example, the `list_of_chat_messages` field in the `ChatMessagesDataset` is of type `list[ChatMessages]`. This means that each of the entries can be converted into a format such as [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) that can be used when interacting with large language models at a low-level. This functionality is available via the `to_chatml()` method.

# In[13]:


print(my_dataset.list_of_chat_messages[0])


# In[14]:


print(my_dataset.list_of_chat_messages[0].to_chatml())


# In[ ]:




