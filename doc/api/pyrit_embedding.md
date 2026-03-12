# pyrit.embedding

Embedding module for PyRIT to provide OpenAI text embedding class.

## `class OpenAITextEmbedding(EmbeddingSupport)`

Text embedding class that works with both Azure OpenAI and platform OpenAI endpoints.
Uses the AsyncOpenAI client under the hood for both providers since they share the same API.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `api_key` | `Optional[str | Callable[[], str | Awaitable[str]]]` | The API key (string) or token provider (callable) for authentication. For Azure with Entra auth, pass get_azure_openai_auth(endpoint) from pyrit.auth. Defaults to OPENAI_EMBEDDING_KEY environment variable. Defaults to `None`. |
| `endpoint` | `Optional[str]` | The API endpoint URL. For Azure: https://{resource}.openai.azure.com/openai/v1 For platform OpenAI: https://api.openai.com/v1 Defaults to OPENAI_EMBEDDING_ENDPOINT environment variable. Defaults to `None`. |
| `model_name` | `Optional[str]` | The model/deployment name (e.g., "text-embedding-3-small"). Defaults to OPENAI_EMBEDDING_MODEL environment variable. Defaults to `None`. |

**Methods:**

#### `generate_text_embedding(text: str, kwargs: Any = {}) → EmbeddingResponse`

Generate text embedding synchronously by calling the async method.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The text to generate the embedding for |
| `**kwargs` | `Any` | Additional arguments to pass to the embeddings API Defaults to `{}`. |

**Returns:**

- `EmbeddingResponse` — The embedding response

#### `generate_text_embedding_async(text: str, kwargs: Any = {}) → EmbeddingResponse`

Generate text embedding asynchronously.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The text to generate the embedding for |
| `**kwargs` | `Any` | Additional arguments to pass to the embeddings API Defaults to `{}`. |

**Returns:**

- `EmbeddingResponse` — The embedding response
