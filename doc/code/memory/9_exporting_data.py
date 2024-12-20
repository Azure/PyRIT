#!/usr/bin/env python
# coding: utf-8

# # 9. Exporting Data Example
#
# This notebook shows all the different ways to export data from memory. This first example exports all conversations from local memory with their respective score values in a JSON format.

# In[ ]:


from pyrit.memory import DuckDBMemory, CentralMemory
from pyrit.common import default_values
from uuid import uuid4
from pyrit.common.path import RESULTS_PATH

default_values.load_environment_files()

memory = DuckDBMemory()
CentralMemory.set_memory_instance(memory)

from pyrit.models import PromptRequestPiece, PromptRequestResponse

conversation_id = str(uuid4())

print(conversation_id)

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

memory = DuckDBMemory()

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))

entries = memory.get_conversation(conversation_id=conversation_id)

for entry in entries:
    print(entry)

# Define file path for export
json_file_path = RESULTS_PATH / "conversation_and_scores_json_example.json"
csv_file_path = RESULTS_PATH / "conversation_and_scores_csv_example.csv"

# # Export the data to a JSON file
conversation_with_scores = memory.export_all_conversations(file_path=json_file_path, export_type="json")
print(f"Exported conversation with scores to JSON: {json_file_path}")

# Export the data to a CSV file
# conversation_with_scores = memory.export_all_conversations(file_path=csv_file_path, export_type="csv")
# print(f"Exported conversation with scores to CSV: {csv_file_path}")


# ## Importing Data as NumPy DataFrame
#
# You can use the exported JSON or CSV files to import the data as a NumPy DataFrame. This can be useful for various data manipulation and analysis tasks.

# In[ ]:


import pandas as pd  # type: ignore

df = pd.read_json(json_file_path)
df.head(1)


# In[ ]:
