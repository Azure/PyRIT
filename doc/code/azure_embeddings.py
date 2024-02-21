# %% [markdown]
# ## Azure OpenAI Embeddings
#
# Similarly to the [Azure OpenAI Completions](./azure_completions.ipynb) endpoint, PyRIT also allows to get embeddings. The embedding response is a wrapper for the OpenAI embedding API.

# %%


from pprint import pprint

from pyrit.embedding.azure_text_embedding import AzureTextEmbedding
from pyrit.common import default_values

default_values.load_default_env()

input_text = "hello"
ada_embedding_engine = AzureTextEmbedding()
embedding_response = ada_embedding_engine.generate_text_embedding(text=input_text)

pprint(embedding_response, width=280, compact=True)

# %% [markdown]
#
# ### Embeddings Serialization
#
# All the PyRIT's embeddings are easily serializable. This allows you to easily save and load embeddings for later use, and be able to inspect the value of the embeddings offline (since
# embeddings are stored as JSON objects).
#

# %% [markdown]
# To view the json of an embedding

# %%
embedding_response.to_json()

# %% [markdown]
# To save an embedding to disk

# %%
from pyrit.common.path import RESULTS_PATH

saved_embedding_path = embedding_response.save_to_file(directory_path=RESULTS_PATH)
saved_embedding_path

# %% [markdown]
# To load an embedding from disk


# %%
from pyrit.common.path import RESULTS_PATH

saved_embedding_path = embedding_response.save_to_file(directory_path=RESULTS_PATH)
saved_embedding_path
