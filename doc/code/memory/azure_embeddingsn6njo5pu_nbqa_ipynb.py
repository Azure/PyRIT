# %%NBQA-CELL-SEP52c935
from pprint import pprint

from pyrit.embedding.azure_text_embedding import AzureTextEmbedding
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

input_text = "hello"
ada_embedding_engine = AzureTextEmbedding()
embedding_response = ada_embedding_engine.generate_text_embedding(text=input_text)

pprint(embedding_response, width=280, compact=True)

# %%NBQA-CELL-SEP52c935
embedding_response.to_json()

# %%NBQA-CELL-SEP52c935
from pyrit.common.path import DB_DATA_PATH

saved_embedding_path = embedding_response.save_to_file(directory_path=DB_DATA_PATH)
saved_embedding_path

# %%NBQA-CELL-SEP52c935
from pyrit.common.path import DB_DATA_PATH

saved_embedding_path = embedding_response.save_to_file(directory_path=DB_DATA_PATH)
saved_embedding_path
