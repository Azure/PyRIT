# %% [markdown]
# The `pyrit.memory` module provides functionality to keep track of the conversation history. In a nutshell, this can be used as follows

# %%

from pyrit.memory import FileMemory
from pyrit.models import ChatMessage

memory = FileMemory()
message_list = [
    ChatMessage(role="user", content="Hi, chat bot! This is my initial prompt."),
    ChatMessage(role="assistant", content="Nice to meet you! This is my response."),
]
next_message = ChatMessage(role="user", content="Wonderful! This is my second prompt to the chat bot.")
message_list.append(next_message)
memory.add_chat_messages_to_memory(conversations=message_list, conversation_id="11111")


# To retrieve the items from memory

memory.get_chat_messages_with_conversation_id(conversation_id="11111")
