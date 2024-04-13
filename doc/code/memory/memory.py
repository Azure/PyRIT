# %% [markdown]
# The `pyrit.memory` module provides functionality to keep track of the conversation history. In a nutshell, this can be used as follows

# %% [markdown]
# The PyRIT DuckDB database comprises of two primary tables: `ConversationStore` and `EmbeddingStore`.
#
# ### **ConversationStore** Table
# The `ConversationStore` table is designed to efficiently store and manage conversational data, with each field tailored to capture specific aspects of the conversation with the LLM model:
#
# | Field            | Type          | Description                                                                                   |
# |------------------|---------------|-----------------------------------------------------------------------------------------------|
# | uuid             | UUID          | A unique identifier for each conversation entry, serving as the primary key.                  |
# | role             | String        | Indicates the origin of the message within the conversation (e.g., "user", "assistant", "system"). |
# | content          | String        | The actual text content of the conversation entry.                                            |
# | conversation_id  | String        | Groups related conversation entries. Linked to a specific LLM model, it aggregates all related conversations under a single identifier. In multi-turn interactions involving two models, there will be two distinct conversation_ids. |
# | timestamp        | DateTime      | The creation or log timestamp of the conversation entry, defaulting to the current UTC time.  |
# | normalizer_id    | String        | Groups messages within a prompt_normalizer, aiding in organizing conversation flows.         |
# | sha256           | String        | An optional SHA-256 hash of the content for integrity verification.                           |
# | labels           | ARRAY(String) | An array of labels for categorizing or filtering conversation entries.                        |
# | idx_conversation_id | Index       | An index on the `conversation_id` column to enhance query performance, particularly for retrieving conversation histories based on conversation_id. |
#
# ### **EmbeddingStore** Table
# The EmbeddingStore table focuses on storing embeddings associated with the conversational data. Its structure includes:
#
# | Field          | Type          | Description                                                                                   |
# |----------------|---------------|-----------------------------------------------------------------------------------------------|
# | uuid           | UUID          | The primary key, which is a foreign key referencing the UUID in the ConversationStore table. |
# | embedding      | ARRAY(String)          | An array of floats representing the embedding vector.       |
# | embedding_type | String        | The name or type of the embedding, indicating the model or method used. |
#

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4
from pyrit.memory import DuckDBMemory
from pyrit.models.models import ChatMessage

conversation_id = str(uuid4())

memory = DuckDBMemory()
message_list = [
    ChatMessage(role="user", content="Hi, chat bot! This is my initial prompt."),
    ChatMessage(role="assistant", content="Nice to meet you! This is my response."),
]
next_message = ChatMessage(role="user", content="Wonderful! This is my second prompt to the chat bot.")
message_list.append(next_message)
memory.add_chat_messages_to_memory(conversations=message_list, conversation_id=conversation_id)


# To retrieve the items from memory

memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

# %%
memory = DuckDBMemory()
message_list = [
    ChatMessage(role="user", content="Hi, chat bot! This is my initial prompt."),
    ChatMessage(role="assistant", content="Nice to meet you! This is my response."),
]
next_message = ChatMessage(role="user", content="Wonderful! This is my second prompt to the chat bot.")
message_list.append(next_message)
memory.add_chat_messages_to_memory(conversations=message_list, conversation_id=conversation_id)


# %%
# To retrieve the items from memory
memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

# %%
# update based on conversation_id
update_fileds = {"content": "this is updated field"}
memory.update_entries_by_conversation_id(conversation_id=conversation_id, update_fields=update_fileds)


# %%
# To retrieve the items from memory
memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

# %%
# Cleanup memory resources
memory.dispose_engine()

# %%
