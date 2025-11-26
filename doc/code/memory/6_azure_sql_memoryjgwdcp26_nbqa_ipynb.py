# %%NBQA-CELL-SEP52c935
from pyrit.memory import CentralMemory
from pyrit.setup import AZURE_SQL, initialize_pyrit

initialize_pyrit(memory_db_type=AZURE_SQL)

memory = CentralMemory.get_memory_instance()
memory.print_schema()  # type: ignore

# %%NBQA-CELL-SEP52c935
from uuid import uuid4

from pyrit.models import Message, MessagePiece

conversation_id = str(uuid4())

message_list = [
    MessagePiece(
        role="user", original_value="Hi, chat bot! This is my initial prompt.", conversation_id=conversation_id
    ),
    MessagePiece(
        role="assistant", original_value="Nice to meet you! This is my response.", conversation_id=conversation_id
    ),
    MessagePiece(
        role="user",
        original_value="Wonderful! This is my second prompt to the chat bot!",
        conversation_id=conversation_id,
    ),
]

memory.add_message_to_memory(request=Message([message_list[0]]))
memory.add_message_to_memory(request=Message([message_list[1]]))
memory.add_message_to_memory(request=Message([message_list[2]]))


entries = memory.get_conversation(conversation_id=conversation_id)

for entry in entries:
    print(entry)


# Cleanup memory resources
memory.dispose_engine()
