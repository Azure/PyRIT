# %%NBQA-CELL-SEP52c935
from uuid import uuid4

from pyrit.common.path import DB_DATA_PATH
from pyrit.memory import CentralMemory
from pyrit.models import Message, MessagePiece
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

conversation_id = str(uuid4())

print(conversation_id)

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

sqlite_memory = CentralMemory.get_memory_instance()
sqlite_memory.add_message_to_memory(request=Message([message_list[0]]))
sqlite_memory.add_message_to_memory(request=Message([message_list[1]]))
sqlite_memory.add_message_to_memory(request=Message([message_list[2]]))

entries = sqlite_memory.get_conversation(conversation_id=conversation_id)

for entry in entries:
    print(entry)

# Define file path for export
json_file_path = DB_DATA_PATH / "conversation_and_scores_json_example.json"
# csv_file_path = DB_DATA_PATH / "conversation_and_scores_csv_example.csv"

# Export the data to a JSON file
conversation_with_scores = sqlite_memory.export_conversations(file_path=json_file_path, export_type="json")
print(f"Exported conversation with scores to JSON: {json_file_path}")

# Export the data to a CSV file
# conversation_with_scores = sqlite_memory.export_conversations(file_path=csv_file_path, export_type="csv")
# print(f"Exported conversation with scores to CSV: {csv_file_path}")

# %%NBQA-CELL-SEP52c935
import pandas as pd  # type: ignore

df = pd.read_json(json_file_path)
df.head(1)
