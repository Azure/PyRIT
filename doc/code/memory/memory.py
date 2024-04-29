# %% [markdown]
# The memory module is the primary way pyrit keeps track of requests and responses to targets. The schema is found in `memory_models.py` and can be programatically viewed as follows
#

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory import DuckDBMemory

memory = DuckDBMemory()

memory.print_schema()

# %% [markdown]
#
# The `pyrit.memory` module provides functionality to keep track of the conversation history. In a nutshell, this can be used as follows:

# %%


from uuid import uuid4
from pyrit.models import PromptRequestPiece, PromptRequestResponse

conversation_id = str(uuid4())

message_list = [
    PromptRequestPiece(
        role="user", original_value="Hi, chat bot! This is my initial prompt.", conversation_id=conversation_id
    ),
    PromptRequestPiece(
        role="assistant", original_value="Nice to meet you! This is my response.", conversation_id=conversation_id
    ),
    PromptRequestPiece(
        role="user",
        original_value="Wonderful! This is my second prompt to the chat bot!",
        conversation_id=conversation_id,
    ),
]

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))


entries = memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

for entry in entries:
    print(entry)


# %%

# update based on conversation_id
update_fileds = {"converted_prompt_text": "this is updated field"}
memory.update_entries_by_conversation_id(conversation_id=conversation_id, update_fields=update_fileds)


prompt_entries = memory.get_conversation(conversation_id=conversation_id)

for prompt_entry in prompt_entries:
    print(prompt_entry)

# %%
# Cleanup memory resources
memory.dispose_engine()

# %%
