# pyrit.memory

Provide functionality for storing and retrieving conversation history and embeddings.

This package defines the core `MemoryInterface` and concrete implementations for different storage backends.

## `class AttackResultEntry(Base)`

Represents the attack result data in the database.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `entry` | `AttackResult` | The attack result object to convert into a database entry. |

**Methods:**

#### `filter_json_serializable_metadata(metadata: dict[str, Any]) → dict[str, Any]`

Filter a dictionary to only include JSON-serializable values.

This function iterates through the metadata dictionary and keeps only
values that can be serialized to JSON, discarding any non-serializable objects.

| Parameter | Type | Description |
|---|---|---|
| `metadata` | `dict[str, Any]` | Dictionary with potentially non-serializable values |

**Returns:**

- `dict[str, Any]` — Dictionary with only JSON-serializable values

#### `get_attack_result() → AttackResult`

Convert this database entry back into an AttackResult object.

**Returns:**

- `AttackResult` — The reconstructed attack result including related conversations and scores.

## `class AzureSQLMemory(MemoryInterface)`

A class to manage conversation memory using Azure SQL Server as the backend database. It leverages SQLAlchemy Base
models for creating tables and provides CRUD operations to interact with the tables.

This class encapsulates the setup of the database connection, table creation based on SQLAlchemy models,
and session management to perform database operations.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `connection_string` | `Optional[str]` | The connection string for the Azure Sql Database. If not provided, it falls back to the 'AZURE_SQL_DB_CONNECTION_STRING' environment variable. Defaults to `None`. |
| `results_container_url` | `Optional[str]` | The URL to an Azure Storage Container. If not provided, it falls back to the 'AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL' environment variable. Defaults to `None`. |
| `results_sas_token` | `Optional[str]` | The Shared Access Signature (SAS) token for the storage container. If not provided, falls back to the 'AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN' environment variable. Defaults to `None`. |
| `verbose` | `bool` | Whether to enable verbose logging for the database engine. Defaults to False. Defaults to `False`. |

**Methods:**

#### `add_message_pieces_to_memory(message_pieces: Sequence[MessagePiece]) → None`

Insert a list of message pieces into the memory storage.

#### `dispose_engine() → None`

Dispose the engine and clean up resources.

#### `get_all_embeddings() → Sequence[EmbeddingDataEntry]`

Fetch all entries from the specified table and returns them as model instances.

**Returns:**

- `Sequence[EmbeddingDataEntry]` — Sequence[EmbeddingDataEntry]: A sequence of EmbeddingDataEntry instances representing all stored embeddings.

#### get_conversation_stats

```python
get_conversation_stats(conversation_ids: Sequence[str]) → dict[str, ConversationStats]
```

Azure SQL implementation: lightweight aggregate stats per conversation.

Executes a single SQL query that returns message count (distinct
sequences), a truncated last-message preview, the first non-empty
labels dict, and the earliest timestamp for each conversation_id.

| Parameter | Type | Description |
|---|---|---|
| `conversation_ids` | `Sequence[str]` | The conversation IDs to query. |

**Returns:**

- `dict[str, ConversationStats]` — Mapping from conversation_id to ConversationStats.

#### `get_session() → Session`

Provide a session for database operations.

**Returns:**

- `Session` — A new SQLAlchemy session bound to the configured engine.

#### `get_unique_attack_class_names() → list[str]`

Azure SQL implementation: extract unique class_name values from
the atomic_attack_identifier JSON column.

**Returns:**

- `list[str]` — Sorted list of unique attack class name strings.

#### `get_unique_converter_class_names() → list[str]`

Azure SQL implementation: extract unique converter class_name values
from the request_converter_identifiers array in the atomic_attack_identifier
JSON column.

**Returns:**

- `list[str]` — Sorted list of unique converter class name strings.

#### `reset_database() → None`

Drop and recreate existing tables.

## `class CentralMemory`

Provide a centralized memory instance across the framework.
The provided memory instance will be reused for future calls.

**Methods:**

#### `get_memory_instance() → MemoryInterface`

Return a centralized memory instance.

**Returns:**

- `MemoryInterface` — The singleton memory instance.

**Raises:**

- `ValueError` — If the central memory instance has not been set.

#### `set_memory_instance(passed_memory: MemoryInterface) → None`

Set a provided memory instance as the central instance for subsequent calls.

| Parameter | Type | Description |
|---|---|---|
| `passed_memory` | `MemoryInterface` | The memory instance to set as the central instance. |

## `class EmbeddingDataEntry(Base)`

Represents the embedding data associated with conversation entries in the database.
Each embedding is linked to a specific conversation entry via an id.

## `class MemoryEmbedding`

The MemoryEmbedding class is responsible for encoding the memory embeddings.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `embedding_model` | `Optional[EmbeddingSupport]` | The embedding model used to generate text embeddings. If not provided, a ValueError is raised. Defaults to `None`. |

**Methods:**

#### `generate_embedding_memory_data(message_piece: MessagePiece) → EmbeddingDataEntry`

Generate metadata for a message piece.

| Parameter | Type | Description |
|---|---|---|
| `message_piece` | `MessagePiece` | the message piece for which to generate a text embedding |

**Returns:**

- `EmbeddingDataEntry` — The generated metadata.

**Raises:**

- `ValueError` — If the message piece is not of type text.

## `class MemoryExporter`

Handles the export of data to various formats, currently supporting only JSON format.
This class utilizes the strategy design pattern to select the appropriate export format.

**Methods:**

#### export_data

```python
export_data(data: list[MessagePiece], file_path: Optional[Path] = None, export_type: str = 'json') → None
```

Export the provided data to a file in the specified format.

| Parameter | Type | Description |
|---|---|---|
| `data` | `list[MessagePiece]` | The data to be exported, as a list of MessagePiece instances. |
| `file_path` | `str` | The full path, including the file name, where the data will be exported. Defaults to `None`. |
| `export_type` | `(str, Optional)` | The format for exporting data. Defaults to "json". Defaults to `'json'`. |

**Raises:**

- `ValueError` — If no file_path is provided or if the specified export format is not supported.

#### `export_to_csv(data: list[MessagePiece], file_path: Optional[Path] = None) → None`

Export the provided data to a CSV file at the specified file path.
Each item in the data list, representing a row from the table,
is converted to a dictionary before being written to the file.

| Parameter | Type | Description |
|---|---|---|
| `data` | `list[MessagePiece]` | The data to be exported, as a list of MessagePiece instances. |
| `file_path` | `Path` | The full path, including the file name, where the data will be exported. Defaults to `None`. |

**Raises:**

- `ValueError` — If no file_path is provided.

#### export_to_json

```python
export_to_json(data: list[MessagePiece], file_path: Optional[Path] = None) → None
```

Export the provided data to a JSON file at the specified file path.
Each item in the data list, representing a row from the table,
is converted to a dictionary before being written to the file.

| Parameter | Type | Description |
|---|---|---|
| `data` | `list[MessagePiece]` | The data to be exported, as a list of MessagePiece instances. |
| `file_path` | `Path` | The full path, including the file name, where the data will be exported. Defaults to `None`. |

**Raises:**

- `ValueError` — If no file_path is provided.

#### export_to_markdown

```python
export_to_markdown(data: list[MessagePiece], file_path: Optional[Path] = None) → None
```

Export the provided data to a Markdown file at the specified file path.
Each item in the data list is converted to a dictionary and formatted as a table.

| Parameter | Type | Description |
|---|---|---|
| `data` | `list[MessagePiece]` | The data to be exported, as a list of MessagePiece instances. |
| `file_path` | `Path` | The full path, including the file name, where the data will be exported. Defaults to `None`. |

**Raises:**

- `ValueError` — If no file_path is provided or if there is no data to export.

## `class MemoryInterface(abc.ABC)`

Abstract interface for conversation memory storage systems.

This interface defines the contract for storing and retrieving chat messages
and conversation history. Implementations can use different storage backends
such as files, databases, or cloud storage services.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `embedding_model` | `Optional[Any]` | If set, this includes embeddings in the memory entries which are extremely useful for comparing chat messages and similarities, but also includes overhead. Defaults to `None`. |

**Methods:**

#### `add_attack_results_to_memory(attack_results: Sequence[AttackResult]) → None`

Insert a list of attack results into the memory storage.
The database model automatically calculates objective_sha256 for consistency.

**Raises:**

- `SQLAlchemyError` — If the database transaction fails.

#### add_attack_results_to_scenario

```python
add_attack_results_to_scenario(scenario_result_id: str, atomic_attack_name: str, attack_results: Sequence[AttackResult]) → bool
```

Add attack results to an existing scenario result in memory.

This method efficiently updates a scenario result by appending new attack results
to a specific atomic attack name without requiring a full retrieve-modify-save cycle.

| Parameter | Type | Description |
|---|---|---|
| `scenario_result_id` | `str` | The ID of the scenario result to update. |
| `atomic_attack_name` | `str` | The name of the atomic attack to add results for. |
| `attack_results` | `Sequence[AttackResult]` | The attack results to add. |

**Returns:**

- `bool` — True if the update was successful, False otherwise.

#### `add_message_pieces_to_memory(message_pieces: Sequence[MessagePiece]) → None`

Insert a list of message pieces into the memory storage.

#### `add_message_to_memory(request: Message) → None`

Insert a list of message pieces into the memory storage.

Automatically updates the sequence to be the next number in the conversation.
If necessary, generates embedding data for applicable entries

| Parameter | Type | Description |
|---|---|---|
| `request` | `MessagePiece` | The message piece to add to the memory. |

#### add_scenario_results_to_memory

```python
add_scenario_results_to_memory(scenario_results: Sequence[ScenarioResult]) → None
```

Insert a list of scenario results into the memory storage.

| Parameter | Type | Description |
|---|---|---|
| `scenario_results` | `Sequence[ScenarioResult]` | Sequence of ScenarioResult objects to store in the database. |

#### `add_scores_to_memory(scores: Sequence[Score]) → None`

Insert a list of scores into the memory storage.

#### add_seed_datasets_to_memory_async

```python
add_seed_datasets_to_memory_async(datasets: Sequence[SeedDataset], added_by: str) → None
```

Insert a list of seed datasets into the memory storage.

| Parameter | Type | Description |
|---|---|---|
| `datasets` | `Sequence[SeedDataset]` | A list of seed datasets to insert. |
| `added_by` | `str` | The user who added the datasets. |

#### add_seed_groups_to_memory_async

```python
add_seed_groups_to_memory_async(prompt_groups: Sequence[SeedGroup], added_by: Optional[str] = None) → None
```

Insert a list of seed groups into the memory storage.

| Parameter | Type | Description |
|---|---|---|
| `prompt_groups` | `Sequence[SeedGroup]` | A list of prompt groups to insert. |
| `added_by` | `str` | The user who added the prompt groups. Defaults to `None`. |

**Raises:**

- `ValueError` — If a seed group does not have at least one seed.
- `ValueError` — If seed group IDs are inconsistent within the same seed group.

#### add_seeds_to_memory_async

```python
add_seeds_to_memory_async(seeds: Sequence[Seed], added_by: Optional[str] = None) → None
```

Insert a list of seeds into the memory storage.

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Seed]` | A list of seeds to insert. |
| `added_by` | `str` | The user who added the seeds. Defaults to `None`. |

**Raises:**

- `ValueError` — If the 'added_by' attribute is not set for each prompt.

#### `cleanup() → None`

Ensure cleanup on process exit.

#### `disable_embedding() → None`

Disable embedding functionality for the memory interface.

Sets the memory_embedding attribute to None, disabling any embedding operations.

#### `dispose_engine() → None`

Dispose the engine and clean up resources.

#### `duplicate_conversation(conversation_id: str) → str`

Duplicate a conversation for reuse.

This can be useful when an attack strategy requires branching out from a particular point in the conversation.
One cannot continue both branches with the same conversation ID since that would corrupt
the memory. Instead, one needs to duplicate the conversation and continue with the new conversation ID.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID with existing conversations. |

**Returns:**

- `str` — The uuid for the new conversation.

#### `duplicate_conversation_excluding_last_turn(conversation_id: str) → str`

Duplicate a conversation, excluding the last turn. In this case, last turn is defined as before the last
user request (e.g. if there is half a turn, it just removes that half).

This can be useful when an attack strategy requires back tracking the last prompt/response pair.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID with existing conversations. |

**Returns:**

- `str` — The uuid for the new conversation.

#### duplicate_messages

```python
duplicate_messages(messages: Sequence[Message]) → tuple[str, Sequence[MessagePiece]]
```

Duplicate messages with a new conversation ID.

Each duplicated piece gets a fresh ``id`` and ``timestamp`` while
preserving ``original_prompt_id`` for tracking lineage.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `Sequence[Message]` | The messages to duplicate. |

**Returns:**

- `tuple[str, Sequence[MessagePiece]]` — Tuple of (new_conversation_id, duplicated_message_pieces).

#### `enable_embedding(embedding_model: Optional[Any] = None) → None`

Enable embedding functionality for the memory interface.

| Parameter | Type | Description |
|---|---|---|
| `embedding_model` | `Optional[Any]` | Optional embedding model to use. If not provided, attempts to create a default embedding model from environment variables. Defaults to `None`. |

**Raises:**

- `ValueError` — If no embedding model is provided and required environment

#### export_conversations

```python
export_conversations(attack_id: Optional[str | uuid.UUID] = None, conversation_id: Optional[str | uuid.UUID] = None, prompt_ids: Optional[Sequence[str] | Sequence[uuid.UUID]] = None, labels: Optional[dict[str, str]] = None, sent_after: Optional[datetime] = None, sent_before: Optional[datetime] = None, original_values: Optional[Sequence[str]] = None, converted_values: Optional[Sequence[str]] = None, data_type: Optional[str] = None, not_data_type: Optional[str] = None, converted_value_sha256: Optional[Sequence[str]] = None, file_path: Optional[Path] = None, export_type: str = 'json') → Path
```

Export conversation data with the given inputs to a specified file.
    Defaults to all conversations if no filters are provided.

| Parameter | Type | Description |
|---|---|---|
| `attack_id` | `Optional[str | uuid.UUID]` | The ID of the attack. Defaults to None. Defaults to `None`. |
| `conversation_id` | `Optional[str | uuid.UUID]` | The ID of the conversation. Defaults to None. Defaults to `None`. |
| `prompt_ids` | `Optional[Sequence[str] | Sequence[uuid.UUID]]` | A list of prompt IDs. Defaults to None. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of labels. Defaults to None. Defaults to `None`. |
| `sent_after` | `Optional[datetime]` | Filter for prompts sent after this datetime. Defaults to None. Defaults to `None`. |
| `sent_before` | `Optional[datetime]` | Filter for prompts sent before this datetime. Defaults to None. Defaults to `None`. |
| `original_values` | `Optional[Sequence[str]]` | A list of original values. Defaults to None. Defaults to `None`. |
| `converted_values` | `Optional[Sequence[str]]` | A list of converted values. Defaults to None. Defaults to `None`. |
| `data_type` | `Optional[str]` | The data type to filter by. Defaults to None. Defaults to `None`. |
| `not_data_type` | `Optional[str]` | The data type to exclude. Defaults to None. Defaults to `None`. |
| `converted_value_sha256` | `Optional[Sequence[str]]` | A list of SHA256 hashes of converted values. Defaults to None. Defaults to `None`. |
| `file_path` | `Optional[Path]` | The path to the file where the data will be exported. Defaults to None. Defaults to `None`. |
| `export_type` | `str` | The format of the export. Defaults to "json". Defaults to `'json'`. |

**Returns:**

- `Path` — The path to the exported file.

#### `get_all_embeddings() → Sequence[EmbeddingDataEntry]`

Load all EmbeddingData from the memory storage handler.

#### get_attack_results

```python
get_attack_results(attack_result_ids: Optional[Sequence[str]] = None, conversation_id: Optional[str] = None, objective: Optional[str] = None, objective_sha256: Optional[Sequence[str]] = None, outcome: Optional[str] = None, attack_class: Optional[str] = None, converter_classes: Optional[Sequence[str]] = None, targeted_harm_categories: Optional[Sequence[str]] = None, labels: Optional[dict[str, str]] = None) → Sequence[AttackResult]
```

Retrieve a list of AttackResult objects based on the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `attack_result_ids` | `Optional[Sequence[str]]` | A list of attack result IDs. Defaults to None. Defaults to `None`. |
| `conversation_id` | `Optional[str]` | The conversation ID to filter by. Defaults to None. Defaults to `None`. |
| `objective` | `Optional[str]` | The objective to filter by (substring match). Defaults to None. Defaults to `None`. |
| `objective_sha256` | `Optional[Sequence[str]]` | A list of objective SHA256 hashes to filter by. Defaults to None. Defaults to `None`. |
| `outcome` | `Optional[str]` | The outcome to filter by (success, failure, undetermined). Defaults to None. Defaults to `None`. |
| `attack_class` | `Optional[str]` | Filter by exact attack class_name in attack_identifier. Defaults to None. Defaults to `None`. |
| `converter_classes` | `Optional[Sequence[str]]` | Filter by converter class names. Returns only attacks that used ALL specified converters (AND logic, case-insensitive). Defaults to None. Defaults to `None`. |
| `targeted_harm_categories` | `Optional[Sequence[str]]` |  A list of targeted harm categories to filter results by. These targeted harm categories are associated with the prompts themselves, meaning they are harm(s) we're trying to elicit with the prompt, not necessarily one(s) that were found in the response. By providing a list, this means ALL categories in the list must be present. Defaults to None. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of memory labels to filter results by. These labels are associated with the prompts themselves, used for custom tagging and tracking. Defaults to None. Defaults to `None`. |

**Returns:**

- `Sequence[AttackResult]` — Sequence[AttackResult]: A list of AttackResult objects that match the specified filters.

#### `get_conversation(conversation_id: str) → MutableSequence[Message]`

Retrieve a list of Message objects that have the specified conversation ID.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID to match. |

**Returns:**

- `MutableSequence[Message]` — MutableSequence[Message]: A list of chat memory entries with the specified conversation ID.

#### get_conversation_stats

```python
get_conversation_stats(conversation_ids: Sequence[str]) → dict[str, ConversationStats]
```

Return lightweight aggregate statistics for one or more conversations.

Computes per-conversation message count (distinct sequence numbers),
a truncated last-message preview, the first non-empty labels dict,
and the earliest message timestamp using efficient SQL aggregation
instead of loading full pieces.

| Parameter | Type | Description |
|---|---|---|
| `conversation_ids` | `Sequence[str]` | The conversation IDs to query. |

**Returns:**

- `dict[str, ConversationStats]` — Mapping from conversation_id to ConversationStats.
- `dict[str, ConversationStats]` — Conversations with no pieces are omitted from the result.

#### get_message_pieces

```python
get_message_pieces(attack_id: Optional[str | uuid.UUID] = None, role: Optional[str] = None, conversation_id: Optional[str | uuid.UUID] = None, prompt_ids: Optional[Sequence[str | uuid.UUID]] = None, labels: Optional[dict[str, str]] = None, prompt_metadata: Optional[dict[str, Union[str, int]]] = None, sent_after: Optional[datetime] = None, sent_before: Optional[datetime] = None, original_values: Optional[Sequence[str]] = None, converted_values: Optional[Sequence[str]] = None, data_type: Optional[str] = None, not_data_type: Optional[str] = None, converted_value_sha256: Optional[Sequence[str]] = None) → Sequence[MessagePiece]
```

Retrieve a list of MessagePiece objects based on the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `attack_id` | `Optional[str | uuid.UUID]` | The ID of the attack. Defaults to None. Defaults to `None`. |
| `role` | `Optional[str]` | The role of the prompt. Defaults to None. Defaults to `None`. |
| `conversation_id` | `Optional[str | uuid.UUID]` | The ID of the conversation. Defaults to None. Defaults to `None`. |
| `prompt_ids` | `Optional[Sequence[str] | Sequence[uuid.UUID]]` | A list of prompt IDs. Defaults to None. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of labels. Defaults to None. Defaults to `None`. |
| `prompt_metadata` | `Optional[dict[str, Union[str, int]]]` | The metadata associated with the prompt. Defaults to None. Defaults to `None`. |
| `sent_after` | `Optional[datetime]` | Filter for prompts sent after this datetime. Defaults to None. Defaults to `None`. |
| `sent_before` | `Optional[datetime]` | Filter for prompts sent before this datetime. Defaults to None. Defaults to `None`. |
| `original_values` | `Optional[Sequence[str]]` | A list of original values. Defaults to None. Defaults to `None`. |
| `converted_values` | `Optional[Sequence[str]]` | A list of converted values. Defaults to None. Defaults to `None`. |
| `data_type` | `Optional[str]` | The data type to filter by. Defaults to None. Defaults to `None`. |
| `not_data_type` | `Optional[str]` | The data type to exclude. Defaults to None. Defaults to `None`. |
| `converted_value_sha256` | `Optional[Sequence[str]]` | A list of SHA256 hashes of converted values. Defaults to None. Defaults to `None`. |

**Returns:**

- `Sequence[MessagePiece]` — Sequence[MessagePiece]: A list of MessagePiece objects that match the specified filters.

**Raises:**

- `Exception` — If there is an error retrieving the prompts,
an exception is logged and an empty list is returned.

#### get_prompt_scores

```python
get_prompt_scores(attack_id: Optional[str | uuid.UUID] = None, role: Optional[str] = None, conversation_id: Optional[str | uuid.UUID] = None, prompt_ids: Optional[Sequence[str | uuid.UUID]] = None, labels: Optional[dict[str, str]] = None, prompt_metadata: Optional[dict[str, Union[str, int]]] = None, sent_after: Optional[datetime] = None, sent_before: Optional[datetime] = None, original_values: Optional[Sequence[str]] = None, converted_values: Optional[Sequence[str]] = None, data_type: Optional[str] = None, not_data_type: Optional[str] = None, converted_value_sha256: Optional[Sequence[str]] = None) → Sequence[Score]
```

Retrieve scores attached to message pieces based on the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `attack_id` | `Optional[str | uuid.UUID]` | The ID of the attack. Defaults to None. Defaults to `None`. |
| `role` | `Optional[str]` | The role of the prompt. Defaults to None. Defaults to `None`. |
| `conversation_id` | `Optional[str | uuid.UUID]` | The ID of the conversation. Defaults to None. Defaults to `None`. |
| `prompt_ids` | `Optional[Sequence[str] | Sequence[uuid.UUID]]` | A list of prompt IDs. Defaults to None. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of labels. Defaults to None. Defaults to `None`. |
| `prompt_metadata` | `Optional[dict[str, Union[str, int]]]` | The metadata associated with the prompt. Defaults to None. Defaults to `None`. |
| `sent_after` | `Optional[datetime]` | Filter for prompts sent after this datetime. Defaults to None. Defaults to `None`. |
| `sent_before` | `Optional[datetime]` | Filter for prompts sent before this datetime. Defaults to None. Defaults to `None`. |
| `original_values` | `Optional[Sequence[str]]` | A list of original values. Defaults to None. Defaults to `None`. |
| `converted_values` | `Optional[Sequence[str]]` | A list of converted values. Defaults to None. Defaults to `None`. |
| `data_type` | `Optional[str]` | The data type to filter by. Defaults to None. Defaults to `None`. |
| `not_data_type` | `Optional[str]` | The data type to exclude. Defaults to None. Defaults to `None`. |
| `converted_value_sha256` | `Optional[Sequence[str]]` | A list of SHA256 hashes of converted values. Defaults to None. Defaults to `None`. |

**Returns:**

- `Sequence[Score]` — Sequence[Score]: A list of scores extracted from the message pieces.

#### `get_request_from_response(response: Message) → Message`

Retrieve the request that produced the given response.

| Parameter | Type | Description |
|---|---|---|
| `response` | `Message` | The response message object to match. |

**Returns:**

- `Message` — The corresponding message object.

**Raises:**

- `ValueError` — If the response is not from an assistant role or has no preceding request.

#### get_scenario_results

```python
get_scenario_results(scenario_result_ids: Optional[Sequence[str]] = None, scenario_name: Optional[str] = None, scenario_version: Optional[int] = None, pyrit_version: Optional[str] = None, added_after: Optional[datetime] = None, added_before: Optional[datetime] = None, labels: Optional[dict[str, str]] = None, objective_target_endpoint: Optional[str] = None, objective_target_model_name: Optional[str] = None) → Sequence[ScenarioResult]
```

Retrieve a list of ScenarioResult objects based on the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `scenario_result_ids` | `Optional[Sequence[str]]` | A list of scenario result IDs. Defaults to None. Defaults to `None`. |
| `scenario_name` | `Optional[str]` | The scenario name to filter by (substring match). Defaults to None. Defaults to `None`. |
| `scenario_version` | `Optional[int]` | The scenario version to filter by. Defaults to None. Defaults to `None`. |
| `pyrit_version` | `Optional[str]` | The PyRIT version to filter by. Defaults to None. Defaults to `None`. |
| `added_after` | `Optional[datetime]` | Filter for scenarios completed after this datetime. Defaults to None. Defaults to `None`. |
| `added_before` | `Optional[datetime]` | Filter for scenarios completed before this datetime. Defaults to None. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of memory labels to filter by. Defaults to None. Defaults to `None`. |
| `objective_target_endpoint` | `Optional[str]` | Filter for scenarios where the objective_target_identifier has an endpoint attribute containing this value (case-insensitive). Defaults to None. Defaults to `None`. |
| `objective_target_model_name` | `Optional[str]` | Filter for scenarios where the objective_target_identifier has a model_name attribute containing this value (case-insensitive). Defaults to None. Defaults to `None`. |

**Returns:**

- `Sequence[ScenarioResult]` — Sequence[ScenarioResult]: A list of ScenarioResult objects that match the specified filters.

#### get_scores

```python
get_scores(score_ids: Optional[Sequence[str]] = None, score_type: Optional[str] = None, score_category: Optional[str] = None, sent_after: Optional[datetime] = None, sent_before: Optional[datetime] = None) → Sequence[Score]
```

Retrieve a list of Score objects based on the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `score_ids` | `Optional[Sequence[str]]` | A list of score IDs to filter by. Defaults to `None`. |
| `score_type` | `Optional[str]` | The type of the score to filter by. Defaults to `None`. |
| `score_category` | `Optional[str]` | The category of the score to filter by. Defaults to `None`. |
| `sent_after` | `Optional[datetime]` | Filter for scores sent after this datetime. Defaults to `None`. |
| `sent_before` | `Optional[datetime]` | Filter for scores sent before this datetime. Defaults to `None`. |

**Returns:**

- `Sequence[Score]` — Sequence[Score]: A list of Score objects that match the specified filters.

#### `get_seed_dataset_names() → Sequence[str]`

Return a list of all seed dataset names in the memory storage.

**Returns:**

- `Sequence[str]` — Sequence[str]: A list of unique dataset names.

#### get_seed_groups

```python
get_seed_groups(value: Optional[str] = None, value_sha256: Optional[Sequence[str]] = None, dataset_name: Optional[str] = None, dataset_name_pattern: Optional[str] = None, data_types: Optional[Sequence[str]] = None, harm_categories: Optional[Sequence[str]] = None, added_by: Optional[str] = None, authors: Optional[Sequence[str]] = None, groups: Optional[Sequence[str]] = None, source: Optional[str] = None, seed_type: Optional[SeedType] = None, is_objective: Optional[bool] = None, parameters: Optional[Sequence[str]] = None, metadata: Optional[dict[str, Union[str, int]]] = None, prompt_group_ids: Optional[Sequence[uuid.UUID]] = None, group_length: Optional[Sequence[int]] = None) → Sequence[SeedGroup]
```

Retrieve groups of seed prompts based on the provided filtering criteria.

| Parameter | Type | Description |
|---|---|---|
| `value` | `(Optional[str], Optional)` | The value to match by substring. Defaults to `None`. |
| `value_sha256` | `(Optional[Sequence[str]], Optional)` | SHA256 hash of value to filter seed groups by. Defaults to `None`. |
| `dataset_name` | `(Optional[str], Optional)` | Name of the dataset to match exactly. Defaults to `None`. |
| `dataset_name_pattern` | `(Optional[str], Optional)` | A pattern to match dataset names using SQL LIKE syntax. Supports wildcards: % (any characters) and _ (single character). Examples: "harm%" matches names starting with "harm", "%test%" matches names containing "test". If both dataset_name and dataset_name_pattern are provided, dataset_name takes precedence. Defaults to `None`. |
| `data_types` | `(Optional[Sequence[str]], Optional)` | List of data types to filter seed prompts by Defaults to `None`. |
| `harm_categories` | `(Optional[Sequence[str]], Optional)` | List of harm categories to filter seed prompts by. Defaults to `None`. |
| `added_by` | `(Optional[str], Optional)` | The user who added the seed groups to filter by. Defaults to `None`. |
| `authors` | `(Optional[Sequence[str]], Optional)` | List of authors to filter seed groups by. Defaults to `None`. |
| `groups` | `(Optional[Sequence[str]], Optional)` | List of groups to filter seed groups by. Defaults to `None`. |
| `source` | `(Optional[str], Optional)` | The source from which the seed prompts originated. Defaults to `None`. |
| `seed_type` | `(Optional[SeedType], Optional)` | The type of seed to filter by ("prompt", "objective", or "simulated_conversation"). Defaults to `None`. |
| `is_objective` | `bool` | Deprecated in 0.13.0. Use seed_type="objective" instead. Defaults to `None`. |
| `parameters` | `(Optional[Sequence[str]], Optional)` | List of parameters to filter by. Defaults to `None`. |
| `metadata` | `(Optional[dict[str, Union[str, int]]], Optional)` | A free-form dictionary for tagging prompts with custom metadata. Defaults to `None`. |
| `prompt_group_ids` | `(Optional[Sequence[uuid.UUID]], Optional)` | List of prompt group IDs to filter by. Defaults to `None`. |
| `group_length` | `(Optional[Sequence[int]], Optional)` | The number of seeds in the group to filter by. Defaults to `None`. |

**Returns:**

- `Sequence[SeedGroup]` — Sequence[SeedGroup]: A list of `SeedGroup` objects that match the filtering criteria.

#### get_seeds

```python
get_seeds(value: Optional[str] = None, value_sha256: Optional[Sequence[str]] = None, dataset_name: Optional[str] = None, dataset_name_pattern: Optional[str] = None, data_types: Optional[Sequence[str]] = None, harm_categories: Optional[Sequence[str]] = None, added_by: Optional[str] = None, authors: Optional[Sequence[str]] = None, groups: Optional[Sequence[str]] = None, source: Optional[str] = None, seed_type: Optional[SeedType] = None, is_objective: Optional[bool] = None, parameters: Optional[Sequence[str]] = None, metadata: Optional[dict[str, Union[str, int]]] = None, prompt_group_ids: Optional[Sequence[uuid.UUID]] = None) → Sequence[Seed]
```

Retrieve a list of seed prompts based on the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `value` | `str` | The value to match by substring. If None, all values are returned. Defaults to `None`. |
| `value_sha256` | `str` | The SHA256 hash of the value to match. If None, all values are returned. Defaults to `None`. |
| `dataset_name` | `str` | The dataset name to match exactly. If None, all dataset names are considered. Defaults to `None`. |
| `dataset_name_pattern` | `str` | A pattern to match dataset names using SQL LIKE syntax. Supports wildcards: % (any characters) and _ (single character). Examples: "harm%" matches names starting with "harm", "%test%" matches names containing "test". If both dataset_name and dataset_name_pattern are provided, dataset_name takes precedence. Defaults to `None`. |
| `data_types` | `Optional[Sequence[str], Optional` | List of data types to filter seed prompts by (e.g., text, image_path). Defaults to `None`. |
| `harm_categories` | `Sequence[str]` | A list of harm categories to filter by. If None, Defaults to `None`. |
| `added_by` | `str` | The user who added the prompts. Defaults to `None`. |
| `authors` | `Sequence[str]` | A list of authors to filter by. Note that this filters by substring, so a query for "Adam Jones" may not return results if the record is "A. Jones", "Jones, Adam", etc. If None, all authors are considered. Defaults to `None`. |
| `groups` | `Sequence[str]` | A list of groups to filter by. If None, all groups are considered. Defaults to `None`. |
| `source` | `str` | The source to filter by. If None, all sources are considered. Defaults to `None`. |
| `seed_type` | `SeedType` | The type of seed to filter by ("prompt", "objective", or "simulated_conversation"). Defaults to `None`. |
| `is_objective` | `bool` | Deprecated in 0.13.0. Use seed_type="objective" instead. Defaults to `None`. |
| `parameters` | `Sequence[str]` | A list of parameters to filter by. Specifying parameters effectively returns prompt templates instead of prompts. Defaults to `None`. |
| `metadata` | `dict[str, str | int]` | A free-form dictionary for tagging prompts with custom metadata. Defaults to `None`. |
| `prompt_group_ids` | `Sequence[uuid.UUID]` | A list of prompt group IDs to filter by. Defaults to `None`. |

**Returns:**

- `Sequence[Seed]` — Sequence[SeedPrompt]: A list of prompts matching the criteria.

**Raises:**

- `ValueError` — If both 'seed_type' and deprecated 'is_objective' parameters are specified.

#### `get_session() → Any`

Provide a SQLAlchemy session for transactional operations.

**Returns:**

- `Any` — A SQLAlchemy session bound to the engine.

#### `get_unique_attack_class_names() → list[str]`

Return sorted unique attack class names from all stored attack results.

Extracts class_name from the attack_identifier JSON column via a
database-level DISTINCT query.

**Returns:**

- `list[str]` — Sorted list of unique attack class name strings.

#### `get_unique_attack_labels() → dict[str, list[str]]`

Return all unique label key-value pairs across attack results.

Labels live on ``PromptMemoryEntry.labels`` (the established SDK
path).  This method JOINs with ``AttackResultEntry`` to scope the
query to conversations that belong to an attack, applies DISTINCT
to reduce duplicate label dicts, then aggregates unique key-value
pairs in Python.

**Returns:**

- `dict[str, list[str]]` — dict[str, list[str]]: Mapping of label keys to sorted lists of
- `dict[str, list[str]]` — unique values.

#### `get_unique_converter_class_names() → list[str]`

Return sorted unique converter class names used across all attack results.

Extracts class_name values from the request_converter_identifiers array
within the attack_identifier JSON column via a database-level query.

**Returns:**

- `list[str]` — Sorted list of unique converter class name strings.

#### `print_schema() → None`

Print the schema of all tables in the database.

#### `update_attack_result(conversation_id: str, update_fields: dict[str, Any]) → bool`

Update specific fields of an existing AttackResultEntry identified by conversation_id.

This method queries for the raw database entry by conversation_id and updates
the specified fields in place, avoiding the creation of duplicate rows.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID of the attack result to update. |
| `update_fields` | `dict[str, Any]` | A dictionary of column names to new values. Valid fields include 'adversarial_chat_conversation_ids', 'pruned_conversation_ids', 'outcome', 'attack_metadata', etc. |

**Returns:**

- `bool` — True if the update was successful, False if the entry was not found.

**Raises:**

- `ValueError` — If update_fields is empty.

#### update_attack_result_by_id

```python
update_attack_result_by_id(attack_result_id: str, update_fields: dict[str, Any]) → bool
```

Update specific fields of an existing AttackResultEntry identified by its primary key.

| Parameter | Type | Description |
|---|---|---|
| `attack_result_id` | `str` | The UUID primary key of the AttackResultEntry. |
| `update_fields` | `dict[str, Any]` | Column names to new values. |

**Returns:**

- `bool` — True if the update was successful, False if the entry was not found.

#### update_labels_by_conversation_id

```python
update_labels_by_conversation_id(conversation_id: str, labels: dict[str, Any]) → bool
```

Update the labels of prompt entries in memory for a given conversation ID.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID of the entries to be updated. |
| `labels` | `dict` | New dictionary of labels. |

**Returns:**

- `bool` — True if the update was successful, False otherwise.

#### update_prompt_entries_by_conversation_id

```python
update_prompt_entries_by_conversation_id(conversation_id: str, update_fields: dict[str, Any]) → bool
```

Update prompt entries for a given conversation ID with the specified field values.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID of the entries to be updated. |
| `update_fields` | `dict` | A dictionary of field names and their new values (ex. {"labels": {"test": "value"}}) |

**Returns:**

- `bool` — True if the update was successful, False otherwise.

**Raises:**

- `ValueError` — If update_fields is empty or not provided.

#### update_prompt_metadata_by_conversation_id

```python
update_prompt_metadata_by_conversation_id(conversation_id: str, prompt_metadata: dict[str, Union[str, int]]) → bool
```

Update the metadata of prompt entries in memory for a given conversation ID.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID of the entries to be updated. |
| `prompt_metadata` | `dict[str, str | int]` | New metadata. |

**Returns:**

- `bool` — True if the update was successful, False otherwise.

#### update_scenario_run_state

```python
update_scenario_run_state(scenario_result_id: str, scenario_run_state: str) → bool
```

Update the run state of an existing scenario result.

| Parameter | Type | Description |
|---|---|---|
| `scenario_result_id` | `str` | The ID of the scenario result to update. |
| `scenario_run_state` | `str` | The new state for the scenario (e.g., "CREATED", "IN_PROGRESS", "COMPLETED", "FAILED"). |

**Returns:**

- `bool` — True if the update was successful, False otherwise.

## `class PromptMemoryEntry(Base)`

Represents the prompt data.

Because of the nature of database and sql alchemy, type ignores are abundant :)

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `entry` | `MessagePiece` | The message piece to convert into a database entry. |

**Methods:**

#### `get_message_piece() → MessagePiece`

Convert this database entry back into a MessagePiece object.

**Returns:**

- `MessagePiece` — The reconstructed message piece with all its data and scores.

## `class SQLiteMemory(MemoryInterface)`

A memory interface that uses SQLite as the backend database.

This class provides functionality to insert, query, and manage conversation data
using SQLite. It supports both file-based and in-memory databases.

Note: this is replacing the old DuckDB implementation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `db_path` | `Optional[Union[Path, str]]` | Path to the SQLite database file. Defaults to "pyrit.db". Defaults to `None`. |
| `verbose` | `bool` | Whether to enable verbose logging. Defaults to False. Defaults to `False`. |

**Methods:**

#### `add_message_pieces_to_memory(message_pieces: Sequence[MessagePiece]) → None`

Insert a list of message pieces into the memory storage.

#### `dispose_engine() → None`

Dispose the engine and close all connections.

#### `export_all_tables(export_type: str = 'json') → None`

Export all table data using the specified exporter.

Iterate over all tables, retrieves their data, and exports each to a file named after the table.

| Parameter | Type | Description |
|---|---|---|
| `export_type` | `str` | The format to export the data in (defaults to "json"). Defaults to `'json'`. |

#### export_conversations

```python
export_conversations(attack_id: Optional[str | uuid.UUID] = None, conversation_id: Optional[str | uuid.UUID] = None, prompt_ids: Optional[Sequence[str] | Sequence[uuid.UUID]] = None, labels: Optional[dict[str, str]] = None, sent_after: Optional[datetime] = None, sent_before: Optional[datetime] = None, original_values: Optional[Sequence[str]] = None, converted_values: Optional[Sequence[str]] = None, data_type: Optional[str] = None, not_data_type: Optional[str] = None, converted_value_sha256: Optional[Sequence[str]] = None, file_path: Optional[Path] = None, export_type: str = 'json') → Path
```

Export conversations and their associated scores from the database to a specified file.

**Returns:**

- `Path` — The path to the exported file.

#### `get_all_embeddings() → Sequence[EmbeddingDataEntry]`

Fetch all entries from the specified table and returns them as model instances.

**Returns:**

- `Sequence[EmbeddingDataEntry]` — Sequence[EmbeddingDataEntry]: A sequence of EmbeddingDataEntry instances representing all stored embeddings.

#### `get_all_table_models() → list[type[Base]]`

Return a list of all table models used in the database by inspecting the Base registry.

**Returns:**

- `list[type[Base]]` — list[Base]: A list of SQLAlchemy model classes.

#### get_conversation_stats

```python
get_conversation_stats(conversation_ids: Sequence[str]) → dict[str, ConversationStats]
```

SQLite implementation: lightweight aggregate stats per conversation.

Executes a single SQL query that returns message count (distinct
sequences), a truncated last-message preview, the first non-empty
labels dict, and the earliest timestamp for each conversation_id.

| Parameter | Type | Description |
|---|---|---|
| `conversation_ids` | `Sequence[str]` | The conversation IDs to query. |

**Returns:**

- `dict[str, ConversationStats]` — Mapping from conversation_id to ConversationStats.

#### `get_session() → Session`

Provide a SQLAlchemy session for transactional operations.

**Returns:**

- `Session` — A SQLAlchemy session bound to the engine.

#### `get_unique_attack_class_names() → list[str]`

SQLite implementation: extract unique class_name values from
the atomic_attack_identifier JSON column.

**Returns:**

- `list[str]` — Sorted list of unique attack class name strings.

#### `get_unique_converter_class_names() → list[str]`

SQLite implementation: extract unique converter class_name values
from the request_converter_identifiers array in the atomic_attack_identifier
JSON column.

**Returns:**

- `list[str]` — Sorted list of unique converter class name strings.

#### `print_schema() → None`

Print the schema of all tables in the SQLite database.

#### `reset_database() → None`

Drop and recreates all tables in the database.

## `class SeedEntry(Base)`

Represents the raw prompt or prompt template data as found in open datasets.

Note: This is different from the PromptMemoryEntry which is the processed prompt data.
SeedPrompt merely reflects basic prompts before plugging into attacks,
running through models with corresponding attack strategies, and applying converters.
PromptMemoryEntry captures the processed prompt data before and after the above steps.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `entry` | `Seed` | The seed object to convert into a database entry. |

**Methods:**

#### `get_seed() → Seed`

Convert this database entry back into a Seed object.

**Returns:**

- `Seed` — The reconstructed seed object (SeedPrompt, SeedObjective, or SeedSimulatedConversation)
