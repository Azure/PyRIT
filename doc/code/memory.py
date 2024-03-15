# %% [markdown]
# The `pyrit.memory` module provides functionality to keep track of the conversation history. In a nutshell, this can be used as follows

# %%
from uuid import uuid4
from pyrit.memory import DuckDBMemory
from pyrit.models import ChatMessage

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

memory.get_chat_messages_with_conversation_id(conversation_id="11111")

# %%
# update by conversation_id
update_fileds = {"content": "this is updtaed field"}
memory.update_entries_by_conversation_id(conversation_id="11111", update_fields=update_fileds)

# %%
# To retrieve the items from memory
memory.get_chat_messages_with_conversation_id(conversation_id="11111")

# %%
# Cleanup memory resources
memory.dispose_engine()

# %%



