# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 9. Exporting Data Example
#
# This notebook shows different ways to export data from memory. This first example exports all conversations from local DuckDB memory with their respective score values in a JSON format. The data can currently be exported both as JSON file or a CSV file that will be saved in your results folder within PyRIT. The CSV export is commented out below. In this example, all conversations are exported, but by using other export functions from `memory_interface`, we can export by specific labels and other methods.

# %%
from uuid import uuid4

from pyrit.common import DUCK_DB, initialize_pyrit
from pyrit.common.path import DB_DATA_PATH
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse

initialize_pyrit(memory_db_type=DUCK_DB)

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

duckdb_memory = CentralMemory.get_memory_instance()
duckdb_memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
duckdb_memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
duckdb_memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))

entries = duckdb_memory.get_conversation(conversation_id=conversation_id)

for entry in entries:
    print(entry)

# Define file path for export
json_file_path = DB_DATA_PATH / "conversation_and_scores_json_example.json"
# csv_file_path = DB_DATA_PATH / "conversation_and_scores_csv_example.csv"

# Export the data to a JSON file
conversation_with_scores = duckdb_memory.export_conversations(file_path=json_file_path, export_type="json")
print(f"Exported conversation with scores to JSON: {json_file_path}")

# Export the data to a CSV file
# conversation_with_scores = duckdb_memory.export_conversations(file_path=csv_file_path, export_type="csv")
# print(f"Exported conversation with scores to CSV: {csv_file_path}")

# Cleanup memory resources
duckdb_memory.dispose_engine()

# %% [markdown]
# You can also use the exported JSON or CSV files to import the data as a NumPy DataFrame. This can be useful for various data manipulation and analysis tasks.

# %%
import pandas as pd  # type: ignore

df = pd.read_json(json_file_path)
df.head(1)
