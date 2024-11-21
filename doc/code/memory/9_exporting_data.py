# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
# # 9. Exporting Data Example
#
# This notebook shows all the different ways to export data from memory. This first example exports all conversations from Azure SQL memory with their respective score values in a JSON format. Without using the database query editor, these export functions allow for a quick approach to exporting data from memory.

# %%
from pyrit.memory.azure_sql_memory import AzureSQLMemory
from pyrit.common import default_values
from pathlib import Path

default_values.load_environment_files()

memory = AzureSQLMemory()

# Define file path for export
json_file_path = Path("conversation_and_scores_json_example")
# csv_file_path = Path(\"conversation_and_scores_csv_example\")

# Export the data to a JSON file
conversation_with_scores = memory.export_all_conversations_with_scores(file_path=json_file_path, export_type="json")
print(f"Exported conversation with scores to JSON: {json_file_path}")

# Export the data to a CSV file
# conversation_with_scores = memory.export_all_conversations_with_scores(file_path=json_file_path, export_type="csv")
# print(f"Exported conversation with scores to CSV: {csv_file_path}")

# %% [markdown]
# ## Importing Data as NumPy DataFrame
#
# You can use the exported JSON or CSV files to import the data as a NumPy DataFrame. This can be useful for various data manipulation and analysis tasks.

# %%
import pandas as pd  # type: ignore

df = pd.read_json(json_file_path)
df.head()
