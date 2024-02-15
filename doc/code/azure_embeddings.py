# %% [markdown]
# ## Azure OpenAI Embeddings
#
# Similarly to the [Azure OpenAI Completions](./azure_completions.ipynb) endpoint, PyRIT also allows to get embeddings. The embedding response is a wrapper for the OpenAI embedding API.

# %%

from pyrit.embedding.azure_text_embedding import AzureTextEmbedding

input_text = "hello"
ada_embedding_engine = AzureTextEmbedding(
    api_key=api_key,
    api_base=api_base,
    model=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
embedding_response = ada_embedding_engine.generate_text_embedding(text=input_text)

pprint(embedding_response, width=280, compact=True)