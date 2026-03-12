# pyrit.models

Public model exports for PyRIT core data structures and helpers.

## Functions

### construct_response_from_request

```python
construct_response_from_request(request: MessagePiece, response_text_pieces: list[str], response_type: PromptDataType = 'text', prompt_metadata: Optional[dict[str, Union[str, int]]] = None, error: PromptResponseError = 'none') → Message
```

Construct a response message from a request message piece.

| Parameter | Type | Description |
|---|---|---|
| `request` | `MessagePiece` | Source request message piece. |
| `response_text_pieces` | `list[str]` | Response values to include. |
| `response_type` | `PromptDataType` | Data type for original and converted response values. Defaults to `'text'`. |
| `prompt_metadata` | `Optional[Dict[str, Union[str, int]]]` | Additional metadata to merge. Defaults to `None`. |
| `error` | `PromptResponseError` | Error classification for the response. Defaults to `'none'`. |

**Returns:**

- `Message` — Constructed response message.

### data_serializer_factory

```python
data_serializer_factory(data_type: PromptDataType, value: Optional[str] = None, extension: Optional[str] = None, category: AllowedCategories) → DataTypeSerializer
```

Create a DataTypeSerializer instance.

| Parameter | Type | Description |
|---|---|---|
| `data_type` | `str` | The type of the data (e.g., 'text', 'image_path', 'audio_path'). |
| `value` | `str` | The data value to be serialized. Defaults to `None`. |
| `extension` | `Optional[str]` | The file extension, if applicable. Defaults to `None`. |
| `category` | `AllowedCategories` | The category or context for the data (e.g., 'seed-prompt-entries'). |

**Returns:**

- `DataTypeSerializer` — An instance of the appropriate serializer.

**Raises:**

- `ValueError` — If the category is not provided or invalid.

### `get_all_harm_definitions() → dict[str, HarmDefinition]`

Load all harm definitions from the standard harm_definition directory.

This function scans the HARM_DEFINITION_PATH directory for all YAML files
and loads each one as a HarmDefinition.

**Returns:**

- `dict[str, HarmDefinition]` — Dict[str, HarmDefinition]: A dictionary mapping category names to their
HarmDefinition objects. The keys are the category names from the YAML files
(e.g., "violence", "hate_speech").

**Raises:**

- `ValueError` — If any YAML file in the directory is invalid.

### group_conversation_message_pieces_by_sequence

```python
group_conversation_message_pieces_by_sequence(message_pieces: Sequence[MessagePiece]) → MutableSequence[Message]
```

Example:
>>> message_pieces = [
>>>     MessagePiece(conversation_id=1, sequence=1, text="Given this list of creatures, which is your
>>>     favorite:"),
>>>     MessagePiece(conversation_id=1, sequence=2, text="Good question!"),
>>>     MessagePiece(conversation_id=1, sequence=1, text="Raccoon, Narwhal, or Sloth?"),
>>>     MessagePiece(conversation_id=1, sequence=2, text="I'd have to say raccoons are my favorite!"),
>>> ]
>>> grouped_responses = group_conversation_message_pieces(message_pieces)
... [
...     Message(message_pieces=[
...         MessagePiece(conversation_id=1, sequence=1, text="Given this list of creatures, which is your
...         favorite:"),
...         MessagePiece(conversation_id=1, sequence=1, text="Raccoon, Narwhal, or Sloth?")
...     ]),
...     Message(message_pieces=[
...         MessagePiece(conversation_id=1, sequence=2, text="Good question!"),
...         MessagePiece(conversation_id=1, sequence=2, text="I'd have to say raccoons are my favorite!")
...     ])
... ]

| Parameter | Type | Description |
|---|---|---|
| `message_pieces` | `Sequence[MessagePiece]` | A list of MessagePiece objects representing individual message pieces. |

**Returns:**

- `MutableSequence[Message]` — MutableSequence[Message]: A list of Message objects representing grouped message
pieces. This is ordered by the sequence number.

**Raises:**

- `ValueError` — If the conversation ID of any message piece does not match the conversation ID of the first
message piece.

### group_message_pieces_into_conversations

```python
group_message_pieces_into_conversations(message_pieces: Sequence[MessagePiece]) → list[list[Message]]
```

Example:
>>> message_pieces = [
>>>     MessagePiece(conversation_id="conv1", sequence=1, text="Hello"),
>>>     MessagePiece(conversation_id="conv2", sequence=1, text="Hi there"),
>>>     MessagePiece(conversation_id="conv1", sequence=2, text="How are you?"),
>>>     MessagePiece(conversation_id="conv2", sequence=2, text="I'm good"),
>>> ]
>>> conversations = group_message_pieces_into_conversations(message_pieces)
>>> # Returns a list of 2 conversations:
>>> # [
>>> #   [Message(seq=1), Message(seq=2)],  # conv1
>>> #   [Message(seq=1), Message(seq=2)]   # conv2
>>> # ]

| Parameter | Type | Description |
|---|---|---|
| `message_pieces` | `Sequence[MessagePiece]` | A list of MessagePiece objects from potentially different conversations. |

**Returns:**

- `list[list[Message]]` — list[list[Message]]: A list of conversations, where each conversation is a list
of Message objects grouped by sequence.

### `sort_message_pieces(message_pieces: list[MessagePiece]) → list[MessagePiece]`

Group by conversation_id.
Order conversations by the earliest timestamp within each conversation_id.
Within each conversation, order messages by sequence.

| Parameter | Type | Description |
|---|---|---|
| `message_pieces` | `list[MessagePiece]` | Message pieces to sort. |

**Returns:**

- `list[MessagePiece]` — list[MessagePiece]: Sorted message pieces.

## `class AttackOutcome(str, Enum)`

Enum representing the possible outcomes of an attack.

Inherits from ``str`` so that values serialize naturally in Pydantic
models and REST responses without a dedicated mapping function.

## `class AttackResult(StrategyResult)`

Base class for all attack results.

**Methods:**

#### `get_active_conversation_ids() → set[str]`

Return the main conversation ID plus pruned (user-visible) related conversation IDs.

Excludes adversarial chat conversations which are internal implementation details.

**Returns:**

- `set[str]` — set[str]: Main + pruned conversation IDs.

#### `get_all_conversation_ids() → set[str]`

Return the main conversation ID plus all related conversation IDs.

**Returns:**

- `set[str]` — set[str]: All conversation IDs associated with this attack.

#### `get_attack_strategy_identifier() → Optional[ComponentIdentifier]`

Return the attack strategy identifier from the composite atomic identifier.

This is the non-deprecated replacement for the ``attack_identifier`` property.
Extracts and returns the ``"attack"`` child from ``atomic_attack_identifier``.

**Returns:**

- `Optional[ComponentIdentifier]` — Optional[ComponentIdentifier]: The attack strategy identifier, or ``None`` if
``atomic_attack_identifier`` is not set.

#### get_conversations_by_type

```python
get_conversations_by_type(conversation_type: ConversationType) → list[ConversationReference]
```

Return all related conversations of the requested type.

| Parameter | Type | Description |
|---|---|---|
| `conversation_type` | `ConversationType` | The type of conversation to filter by. |

**Returns:**

- `list[ConversationReference]` — A list of related conversations matching the specified type.

#### `get_pruned_conversation_ids() → list[str]`

Return IDs of pruned (branched) conversations only.

**Returns:**

- `list[str]` — list[str]: Pruned conversation IDs.

#### `includes_conversation(conversation_id: str) → bool`

Check whether a conversation belongs to this attack (main or any related).

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID to check. |

**Returns:**

- `bool` — True if the conversation is part of this attack.

## `class AudioPathDataTypeSerializer(DataTypeSerializer)`

Serializer for audio path values stored on disk.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | Data category folder name. |
| `prompt_text` | `Optional[str]` | Optional existing audio path. Defaults to `None`. |
| `extension` | `Optional[str]` | Optional audio extension. Defaults to `None`. |

**Methods:**

#### `data_on_disk() → bool`

Indicate whether this serializer persists data on disk.

**Returns:**

- `bool` — Always True for audio path serializers.

## `class AzureBlobStorageIO(StorageIO)`

Implementation of StorageIO for Azure Blob Storage.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `container_url` | `Optional[str]` | Azure Blob container URL. Defaults to `None`. |
| `sas_token` | `Optional[str]` | Optional SAS token. Defaults to `None`. |
| `blob_content_type` | `SupportedContentType` | Blob content type for uploads. Defaults to `SupportedContentType.PLAIN_TEXT`. |

**Methods:**

#### `create_directory_if_not_exists(directory_path: Union[Path, str]) → None`

Log a no-op directory creation for Azure Blob Storage.

| Parameter | Type | Description |
|---|---|---|
| `directory_path` | `Union[Path, str]` | Requested directory path. |

#### `is_file(path: Union[Path, str]) → bool`

Check whether the path refers to a file (blob) in Azure Blob Storage.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[Path, str]` | Blob URL or path to test. |

**Returns:**

- `bool` — True when the blob exists and has non-zero content size.

#### `parse_blob_url(file_path: str) → tuple[str, str]`

Parse a blob URL to extract the container and blob name.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Full blob URL. |

**Returns:**

- `tuple[str, str]` — tuple[str, str]: Container name and blob name.

**Raises:**

- `ValueError` — If file_path is not a valid blob URL.

#### `path_exists(path: Union[Path, str]) → bool`

Check whether a given path exists in the Azure Blob Storage container.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[Path, str]` | Blob URL or path to test. |

**Returns:**

- `bool` — True when the path exists.

#### `read_file(path: Union[Path, str]) → bytes`

Asynchronously reads the content of a file (blob) from Azure Blob Storage.

If the provided `path` is a full URL
(e.g., "https://account.blob.core.windows.net/container/dir1/dir2/sample.png"),
it extracts the relative blob path (e.g., "dir1/dir2/sample.png") to correctly access the blob.
If a relative path is provided, it will use it as-is.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | The path to the file (blob) in Azure Blob Storage.                     This can be either a full URL or a relative path. |

**Returns:**

- `bytes` — The content of the file (blob) as bytes.

**Raises:**

- `Exception` — If there is an error in reading the blob file, an exception will be logged
    and re-raised.

#### `write_file(path: Union[Path, str], data: bytes) → None`

Write data to Azure Blob Storage at the specified path.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | The full Azure Blob Storage URL |
| `data` | `bytes` | The data to write. |

## `class BinaryPathDataTypeSerializer(DataTypeSerializer)`

Serializer for generic binary path values stored on disk.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | The category or context for the data. |
| `prompt_text` | `Optional[str]` | The binary file path or identifier. Defaults to `None`. |
| `extension` | `Optional[str]` | The file extension, defaults to 'bin'. Defaults to `None`. |

**Methods:**

#### `data_on_disk() → bool`

Indicate whether this serializer persists data on disk.

**Returns:**

- `bool` — Always True for binary path serializers.

## `class ChatMessage(BaseModel)`

Represents a chat message for API consumption.

The content field can be:
- A simple string for single-part text messages
- A list of dicts for multipart messages (e.g., text + images)

**Methods:**

#### `from_json(json_str: str) → ChatMessage`

Deserialize a ChatMessage from a JSON string.

| Parameter | Type | Description |
|---|---|---|
| `json_str` | `str` | A JSON string representation of a ChatMessage. |

**Returns:**

- `ChatMessage` — A ChatMessage instance.

#### `to_dict() → dict[str, Any]`

Convert the ChatMessage to a dictionary.

**Returns:**

- `dict[str, Any]` — A dictionary representation of the message, excluding None values.

#### `to_json() → str`

Serialize the ChatMessage to a JSON string.

**Returns:**

- `str` — A JSON string representation of the message.

## `class ChatMessageListDictContent(ChatMessage)`

Deprecated: Use ChatMessage instead.

This class exists for backward compatibility and will be removed in a future version.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `**data` | `Any` | Keyword arguments accepted by ChatMessage. Defaults to `{}`. |

## `class ChatMessagesDataset(BaseModel)`

Represents a dataset of chat messages.

## `class ConversationReference`

Immutable reference to a conversation that played a role in the attack.

## `class ConversationStats`

Lightweight aggregate statistics for a conversation.

Used to build attack summaries without loading full message pieces.

## `class ConversationType(Enum)`

Types of conversations that can be associated with an attack.

## `class DataTypeSerializer(abc.ABC)`

Abstract base class for data type normalizers.

Responsible for reading and saving multi-modal data types to local disk or Azure Storage Account.

**Methods:**

#### `data_on_disk() → bool`

Indicate whether the data is stored on disk.

**Returns:**

- `bool` — True when data is persisted on disk.

#### `get_data_filename(file_name: Optional[str] = None) → Union[Path, str]`

Generate or retrieve a unique filename for the data file.

| Parameter | Type | Description |
|---|---|---|
| `file_name` | `Optional[str]` | Optional file name override. Defaults to `None`. |

**Returns:**

- `Union[Path, str]` — Union[Path, str]: Full storage path for the generated data file.

**Raises:**

- `TypeError` — If the serializer is not configured for on-disk data.
- `RuntimeError` — If required data subdirectory information is missing.

#### `get_extension(file_path: str) → str | None`

Get the file extension from the file path.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Input file path. |

**Returns:**

- `str | None` — str | None: File extension (including dot) or None if unavailable.

#### `get_mime_type(file_path: str) → str | None`

Get the MIME type of the file path.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Input file path. |

**Returns:**

- `str | None` — str | None: MIME type if detectable; otherwise None.

#### `get_sha256() → str`

Compute SHA256 hash for this serializer's current value.

**Returns:**

- `str` — Hex digest of the computed SHA256 hash.

**Raises:**

- `FileNotFoundError` — If on-disk data path does not exist.
- `ValueError` — If in-memory data cannot be converted to bytes.

#### `read_data() → bytes`

Read data from storage.

**Returns:**

- `bytes` — The data read from storage.

**Raises:**

- `TypeError` — If the serializer does not represent on-disk data.
- `RuntimeError` — If no value is set.
- `FileNotFoundError` — If the referenced file does not exist.

#### `read_data_base64() → str`

Read data from storage and return it as a base64 string.

**Returns:**

- `str` — Base64-encoded data.

#### `save_b64_image(data: str | bytes, output_filename: str = None) → None`

Save a base64-encoded image to storage.

| Parameter | Type | Description |
|---|---|---|
| `data` | `str | bytes` | string or bytes with base64 data |
| `output_filename` | `(optional, str)` | filename to store image as. Defaults to UUID if not provided Defaults to `None`. |

#### `save_data(data: bytes, output_filename: Optional[str] = None) → None`

Save data to storage.

| Parameter | Type | Description |
|---|---|---|
| `data` | `bytes` | bytes: The data to be saved. |
| `output_filename` | `(optional, str)` | filename to store data as. Defaults to UUID if not provided Defaults to `None`. |

#### save_formatted_audio

```python
save_formatted_audio(data: bytes, num_channels: int = 1, sample_width: int = 2, sample_rate: int = 16000, output_filename: Optional[str] = None) → None
```

Save PCM16 or similarly formatted audio data to storage.

| Parameter | Type | Description |
|---|---|---|
| `data` | `bytes` | bytes with audio data |
| `output_filename` | `(optional, str)` | filename to store audio as. Defaults to UUID if not provided Defaults to `None`. |
| `num_channels` | `(optional, int)` | number of channels in audio data. Defaults to 1 Defaults to `1`. |
| `sample_width` | `(optional, int)` | sample width in bytes. Defaults to 2 Defaults to `2`. |
| `sample_rate` | `(optional, int)` | sample rate in Hz. Defaults to 16000 Defaults to `16000`. |

## `class DiskStorageIO(StorageIO)`

Implementation of StorageIO for local disk storage.

**Methods:**

#### `create_directory_if_not_exists(path: Union[Path, str]) → None`

Asynchronously creates a directory if it doesn't exist on the local disk.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | The directory path to create. |

#### `is_file(path: Union[Path, str]) → bool`

Check whether the given path is a file (not a directory).

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | The path to check. |

**Returns:**

- `bool` — True if the path is a file, False otherwise.

#### `path_exists(path: Union[Path, str]) → bool`

Check whether a path exists on the local disk.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | The path to check. |

**Returns:**

- `bool` — True if the path exists, False otherwise.

#### `read_file(path: Union[Path, str]) → bytes`

Asynchronously reads a file from the local disk.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[Path, str]` | The path to the file. |

**Returns:**

- `bytes` — The content of the file.

#### `write_file(path: Union[Path, str], data: bytes) → None`

Asynchronously writes data to a file on the local disk.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | The path to the file. |
| `data` | `bytes` | The content to write to the file. |

## `class EmbeddingData(BaseModel)`

Single embedding vector payload with index and object metadata.

## `class EmbeddingResponse(BaseModel)`

Embedding API response containing vectors, model metadata, and usage.

**Methods:**

#### `load_from_file(file_path: Path) → EmbeddingResponse`

Load the embedding response from disk.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `Path` | The path to load the file from. |

**Returns:**

- `EmbeddingResponse` — The loaded embedding response.

#### `save_to_file(directory_path: Path) → str`

Save the embedding response to disk and return the path of the new file.

| Parameter | Type | Description |
|---|---|---|
| `directory_path` | `Path` | The path to save the file to. |

**Returns:**

- `str` — The full path to the file that was saved.

#### `to_json() → str`

Serialize this embedding response to JSON.

**Returns:**

- `str` — JSON-encoded embedding response.

## `class EmbeddingSupport(ABC)`

Protocol-like interface for classes that generate text embeddings.

**Methods:**

#### `generate_text_embedding(text: str, kwargs: object = {}) → EmbeddingResponse`

Generate text embedding synchronously.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The text to generate the embedding for |
| `**kwargs` | `object` | Additional arguments to pass to the function. Defaults to `{}`. |

**Returns:**

- `EmbeddingResponse` — The embedding response

#### generate_text_embedding_async

```python
generate_text_embedding_async(text: str, kwargs: object = {}) → EmbeddingResponse
```

Generate text embedding asynchronously.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The text to generate the embedding for |
| `**kwargs` | `object` | Additional arguments to pass to the function. Defaults to `{}`. |

**Returns:**

- `EmbeddingResponse` — The embedding response

## `class EmbeddingUsageInformation(BaseModel)`

Token usage metadata returned by an embedding API.

## `class ErrorDataTypeSerializer(DataTypeSerializer)`

Serializer for error payloads stored as in-memory text.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt_text` | `str` | Error payload text. |

**Methods:**

#### `data_on_disk() → bool`

Indicate whether this serializer persists data on disk.

**Returns:**

- `bool` — Always False for error serializers.

## `class HarmDefinition`

A harm definition loaded from a YAML file.

This class represents the structured content of a harm definition YAML file,
which includes the version, category name, and scale descriptions that define
how to score content for this harm category.

**Methods:**

#### `from_yaml(harm_definition_path: Union[str, Path]) → HarmDefinition`

Load and validate a harm definition from a YAML file.

The function first checks if the path is a simple filename (e.g., "violence.yaml")
and if so, looks for it in the standard HARM_DEFINITION_PATH directory.
Otherwise, it treats the path as a full or relative path.

| Parameter | Type | Description |
|---|---|---|
| `harm_definition_path` | `Union[str, Path]` | Path to the harm definition YAML file. Can be a simple filename like "violence.yaml" which will be resolved relative to the standard harm_definition directory, or a full path. |

**Returns:**

- `HarmDefinition` — The loaded harm definition.

**Raises:**

- `FileNotFoundError` — If the harm definition file does not exist.
- `ValueError` — If the YAML file is invalid or missing required fields.

#### `get_scale_description(score_value: str) → Optional[str]`

Get the description for a specific score value.

| Parameter | Type | Description |
|---|---|---|
| `score_value` | `str` | The score value to look up (e.g., "1", "2"). |

**Returns:**

- `Optional[str]` — The description for the score value, or None if not found.

#### `validate_category(category: str, check_exists: bool = False) → bool`

Validate a harm category name.

Validates that the category name follows the naming convention (lowercase letters
and underscores only) and optionally checks if it exists in the standard
harm definitions.

| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | The category name to validate. |
| `check_exists` | `bool` | If True, also verify the category exists in get_all_harm_definitions(). Defaults to False. Defaults to `False`. |

**Returns:**

- `bool` — True if the category is valid (and exists if check_exists is True),
- `bool` — False otherwise.

## `class ImagePathDataTypeSerializer(DataTypeSerializer)`

Serializer for image path values stored on disk.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | Data category folder name. |
| `prompt_text` | `Optional[str]` | Optional existing image path. Defaults to `None`. |
| `extension` | `Optional[str]` | Optional image extension. Defaults to `None`. |

**Methods:**

#### `data_on_disk() → bool`

Indicate whether this serializer persists data on disk.

**Returns:**

- `bool` — Always True for image path serializers.

## `class Message`

Represents a message in a conversation, for example a prompt or a response to a prompt.

This is a single request to a target. It can contain multiple message pieces.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `message_pieces` | `Sequence[MessagePiece]` | Pieces belonging to the same message turn. |
| `skip_validation` | `Optional[bool]` | Whether to skip consistency validation. Defaults to `False`. |

**Methods:**

#### `duplicate_message() → Message`

Create a deep copy of this message with new IDs and timestamp for all message pieces.

This is useful when you need to reuse a message template but want fresh IDs
to avoid database conflicts (e.g., during retry attempts).

The original_prompt_id is intentionally kept the same to track the origin.
Generates a new timestamp to reflect when the duplicate is created.

**Returns:**

- `Message` — A new Message with deep-copied message pieces, new IDs, and fresh timestamp.

#### flatten_to_message_pieces

```python
flatten_to_message_pieces(messages: Sequence[Message]) → MutableSequence[MessagePiece]
```

Flatten messages into a single list of message pieces.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `Sequence[Message]` | Messages to flatten. |

**Returns:**

- `MutableSequence[MessagePiece]` — MutableSequence[MessagePiece]: Flattened message pieces.

#### from_prompt

```python
from_prompt(prompt: str, role: ChatMessageRole, prompt_metadata: Optional[dict[str, Union[str, int]]] = None) → Message
```

Build a single-piece message from prompt text.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | Prompt text. |
| `role` | `ChatMessageRole` | Role assigned to the message piece. |
| `prompt_metadata` | `Optional[Dict[str, Union[str, int]]]` | Optional prompt metadata. Defaults to `None`. |

**Returns:**

- `Message` — Constructed message instance.

#### `from_system_prompt(system_prompt: str) → Message`

Build a message from a system prompt.

| Parameter | Type | Description |
|---|---|---|
| `system_prompt` | `str` | System instruction text. |

**Returns:**

- `Message` — Constructed system-role message.

#### `get_all_values(messages: Sequence[Message]) → list[str]`

Return all converted values across the provided messages.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `Sequence[Message]` | Messages to aggregate. |

**Returns:**

- `list[str]` — list[str]: Flattened list of converted values.

#### `get_piece(n: int = 0) → MessagePiece`

Return the nth message piece.

| Parameter | Type | Description |
|---|---|---|
| `n` | `int` | Zero-based index of the piece to return. Defaults to `0`. |

**Returns:**

- `MessagePiece` — Selected message piece.

**Raises:**

- `ValueError` — If the message has no pieces.
- `IndexError` — If the index is out of bounds.

#### get_piece_by_type

```python
get_piece_by_type(data_type: Optional[PromptDataType] = None, original_value_data_type: Optional[PromptDataType] = None, converted_value_data_type: Optional[PromptDataType] = None) → Optional[MessagePiece]
```

Return the first message piece matching the given data type, or None.

| Parameter | Type | Description |
|---|---|---|
| `data_type` | `Optional[PromptDataType]` | Alias for converted_value_data_type (for convenience). Defaults to `None`. |
| `original_value_data_type` | `Optional[PromptDataType]` | The original_value_data_type to filter by. Defaults to `None`. |
| `converted_value_data_type` | `Optional[PromptDataType]` | The converted_value_data_type to filter by. Defaults to `None`. |

**Returns:**

- `Optional[MessagePiece]` — The first matching MessagePiece, or None if no match is found.

#### get_pieces_by_type

```python
get_pieces_by_type(data_type: Optional[PromptDataType] = None, original_value_data_type: Optional[PromptDataType] = None, converted_value_data_type: Optional[PromptDataType] = None) → list[MessagePiece]
```

Return all message pieces matching the given data type.

| Parameter | Type | Description |
|---|---|---|
| `data_type` | `Optional[PromptDataType]` | Alias for converted_value_data_type (for convenience). Defaults to `None`. |
| `original_value_data_type` | `Optional[PromptDataType]` | The original_value_data_type to filter by. Defaults to `None`. |
| `converted_value_data_type` | `Optional[PromptDataType]` | The converted_value_data_type to filter by. Defaults to `None`. |

**Returns:**

- `list[MessagePiece]` — A list of matching MessagePiece objects (may be empty).

#### `get_value(n: int = 0) → str`

Return the converted value of the nth message piece.

| Parameter | Type | Description |
|---|---|---|
| `n` | `int` | Zero-based index of the piece to read. Defaults to `0`. |

**Returns:**

- `str` — Converted value of the selected message piece.

**Raises:**

- `IndexError` — If the index is out of bounds.

#### `get_values() → list[str]`

Return the converted values of all message pieces.

**Returns:**

- `list[str]` — list[str]: Converted values for all message pieces.

#### `is_error() → bool`

Check whether any message piece indicates an error.

**Returns:**

- `bool` — True when any piece has a non-none error flag or error data type.

#### `set_response_not_in_database() → None`

Set that the prompt is not in the database.

This is needed when we're scoring prompts or other things that have not been sent by PyRIT

#### `set_simulated_role() → None`

Set the role of all message pieces to simulated_assistant.

This marks the message as coming from a simulated conversation
rather than an actual target response.

#### `to_dict() → dict[str, object]`

Convert the message to a dictionary representation.

**Returns:**

- `dict[str, object]` — A dictionary with 'role', 'converted_value', 'conversation_id', 'sequence',
and 'converted_value_data_type' keys.

#### `validate() → None`

Validate that all message pieces are internally consistent.

**Raises:**

- `ValueError` — If piece collection is empty or contains mismatched conversation IDs,
sequence numbers, roles, or missing converted values.

## `class MessagePiece`

Represents a piece of a message to a target.

This class represents a single piece of a message that will be sent
to a target. Since some targets can handle multiple pieces (e.g., text and images),
requests are composed of lists of MessagePiece objects.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `role` | `ChatMessageRole` | The role of the prompt (system, assistant, user). |
| `original_value` | `str` | The text of the original prompt. If prompt is an image, it's a link. |
| `original_value_sha256` | `Optional[str]` | The SHA256 hash of the original prompt data. Defaults to None. Defaults to `None`. |
| `converted_value` | `Optional[str]` | The text of the converted prompt. If prompt is an image, it's a link. Defaults to None. Defaults to `None`. |
| `converted_value_sha256` | `Optional[str]` | The SHA256 hash of the converted prompt data. Defaults to None. Defaults to `None`. |
| `id` | `Optional[uuid.UUID | str]` | The unique identifier for the memory entry. Defaults to None (auto-generated). Defaults to `None`. |
| `conversation_id` | `Optional[str]` | The identifier for the conversation which is associated with a single target. Defaults to None. Defaults to `None`. |
| `sequence` | `int` | The order of the conversation within a conversation_id. Defaults to -1. Defaults to `-1`. |
| `labels` | `Optional[dict[str, str]]` | The labels associated with the memory entry. Several can be standardized. Defaults to None. Defaults to `None`. |
| `prompt_metadata` | `Optional[dict[str, Union[str, int]]]` | The metadata associated with the prompt. This can be specific to any scenarios. Because memory is how components talk with each other, this can be component specific. e.g. the URI from a file uploaded to a blob store, or a document type you want to upload. Defaults to None. Defaults to `None`. |
| `converter_identifiers` | `Optional[list[Union[ComponentIdentifier, dict[str, str]]]]` | The converter identifiers for the prompt. Can be ComponentIdentifier objects or dicts (deprecated, will be removed in 0.14.0). Defaults to None. Defaults to `None`. |
| `prompt_target_identifier` | `Optional[Union[ComponentIdentifier, dict[str, Any]]]` | The target identifier for the prompt. Defaults to None. Defaults to `None`. |
| `attack_identifier` | `Optional[Union[ComponentIdentifier, dict[str, str]]]` | The attack identifier for the prompt. Defaults to None. Defaults to `None`. |
| `scorer_identifier` | `Optional[Union[ComponentIdentifier, dict[str, str]]]` | The scorer identifier for the prompt. Can be a ComponentIdentifier or a dict (deprecated, will be removed in 0.13.0). Defaults to None. Defaults to `None`. |
| `original_value_data_type` | `PromptDataType` | The data type of the original prompt (text, image). Defaults to "text". Defaults to `'text'`. |
| `converted_value_data_type` | `Optional[PromptDataType]` | The data type of the converted prompt (text, image). Defaults to "text". Defaults to `None`. |
| `response_error` | `PromptResponseError` | The response error type. Defaults to "none". Defaults to `'none'`. |
| `originator` | `Originator` | The originator of the prompt. Defaults to "undefined". Defaults to `'undefined'`. |
| `original_prompt_id` | `Optional[uuid.UUID]` | The original prompt id. It is equal to id unless it is a duplicate. Defaults to None. Defaults to `None`. |
| `timestamp` | `Optional[datetime]` | The timestamp of the memory entry. Defaults to None (auto-generated). Defaults to `None`. |
| `scores` | `Optional[list[Score]]` | The scores associated with the prompt. Defaults to None. Defaults to `None`. |
| `targeted_harm_categories` | `Optional[list[str]]` | The harm categories associated with the prompt. Defaults to None. Defaults to `None`. |

**Methods:**

#### `get_role_for_storage() → ChatMessageRole`

Get the actual stored role, including simulated_assistant.

Use this when duplicating messages or preserving role information
for storage. For API calls or comparisons, use api_role instead.

**Returns:**

- `ChatMessageRole` — The actual role stored (may be simulated_assistant).

#### `has_error() → bool`

Check if the message piece has an error.

**Returns:**

- `bool` — True when the response_error is not "none".

#### `is_blocked() → bool`

Check if the message piece is blocked.

**Returns:**

- `bool` — True when the response_error is "blocked".

#### `set_piece_not_in_database() → None`

Set that the prompt is not in the database.

This is needed when we're scoring prompts or other things that have not been sent by PyRIT

#### `set_sha256_values_async() → None`

Compute SHA256 hash values for original and converted payloads.
It should be called after object creation if `original_value` and `converted_value` are set.

Note, this method is async due to the blob retrieval. And because of that, we opted
to take it out of main and setter functions. The disadvantage is that it must be explicitly called.

#### `to_dict() → dict[str, object]`

Convert this message piece to a dictionary representation.

**Returns:**

- `dict[str, object]` — dict[str, object]: Dictionary representation suitable for serialization.

#### `to_message() → Message`

Convert this message piece into a Message.

**Returns:**

- `Message` — A Message containing this piece.

## `class NextMessageSystemPromptPaths(enum.Enum)`

Enum for predefined next message generation system prompt paths.

## `class QuestionAnsweringDataset(BaseModel)`

Represents a dataset for question answering.

## `class QuestionAnsweringEntry(BaseModel)`

Represents a question model.

**Methods:**

#### `get_correct_answer_text() → str`

Get the text of the correct answer.

**Returns:**

- `str` — Text corresponding to the configured correct answer index.

**Raises:**

- `ValueError` — If no choice matches the configured correct answer.

## `class QuestionChoice(BaseModel)`

Represents a choice for a question.

## `class ScaleDescription`

A single scale description entry from a harm definition.

## `class ScenarioIdentifier`

Scenario result class for aggregating results from multiple AtomicAttacks.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Name of the scenario. |
| `description` | `str` | Description of the scenario. Defaults to `''`. |
| `scenario_version` | `int` | Version of the scenario. Defaults to `1`. |
| `init_data` | `Optional[dict]` | Initialization data. Defaults to `None`. |
| `pyrit_version` | `Optional[str]` | PyRIT version string. If None, uses current version. Defaults to `None`. |

## `class ScenarioResult`

Scenario result class for aggregating scenario results.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `scenario_identifier` | `ScenarioIdentifier` | Identifier for the executed scenario. |
| `objective_target_identifier` | `Union[Dict[str, Any], TargetIdentifier]` | Target identifier. |
| `attack_results` | `dict[str, List[AttackResult]]` | Results grouped by atomic attack name. |
| `objective_scorer_identifier` | `Union[Dict[str, Any], ScorerIdentifier]` | Objective scorer identifier. |
| `scenario_run_state` | `ScenarioRunState` | Current scenario run state. Defaults to `'CREATED'`. |
| `labels` | `Optional[dict[str, str]]` | Optional labels. Defaults to `None`. |
| `completion_time` | `Optional[datetime]` | Optional completion timestamp. Defaults to `None`. |
| `number_tries` | `int` | Number of run attempts. Defaults to `0`. |
| `id` | `Optional[uuid.UUID]` | Optional scenario result ID. Defaults to `None`. |
| `objective_scorer` | `Optional[Scorer]` | Deprecated scorer object parameter. Defaults to `None`. |

**Methods:**

#### `get_objectives(atomic_attack_name: Optional[str] = None) → list[str]`

Get the list of unique objectives for this scenario.

| Parameter | Type | Description |
|---|---|---|
| `atomic_attack_name` | `Optional[str]` | Name of specific atomic attack to include. If None, includes objectives from all atomic attacks. Defaults to None. Defaults to `None`. |

**Returns:**

- `list[str]` — List[str]: Deduplicated list of objectives.

#### `get_scorer_evaluation_metrics() → Optional[ScorerMetrics]`

Get the evaluation metrics for the scenario's scorer from the scorer evaluation registry.

**Returns:**

- `Optional[ScorerMetrics]` — The evaluation metrics object, or None if not found.

#### `get_strategies_used() → list[str]`

Get the list of strategies used in this scenario.

**Returns:**

- `list[str]` — List[str]: Atomic attack strategy names present in the results.

#### `normalize_scenario_name(scenario_name: str) → str`

Normalize a scenario name to match the stored class name format.

Converts CLI-style snake_case names (e.g., "foundry" or "content_harms") to
PascalCase class names (e.g., "Foundry" or "ContentHarms") for database queries.
If the input is already in PascalCase or doesn't match the snake_case pattern,
it is returned unchanged.

This is the inverse of ScenarioRegistry._class_name_to_scenario_name().

| Parameter | Type | Description |
|---|---|---|
| `scenario_name` | `str` | The scenario name to normalize. |

**Returns:**

- `str` — The normalized scenario name suitable for database queries.

#### `objective_achieved_rate(atomic_attack_name: Optional[str] = None) → int`

Get the success rate of this scenario.

| Parameter | Type | Description |
|---|---|---|
| `atomic_attack_name` | `Optional[str]` | Name of specific atomic attack to calculate rate for. If None, calculates rate across all atomic attacks. Defaults to None. Defaults to `None`. |

**Returns:**

- `int` — Success rate as a percentage (0-100).

## `class Score`

Represents a normalized score generated by a scorer component.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `score_value` | `str` | Normalized score value. |
| `score_value_description` | `str` | Human-readable score value description. |
| `score_type` | `ScoreType` | Score type (true_false or float_scale). |
| `score_rationale` | `str` | Rationale for the score. |
| `message_piece_id` | `str | uuid.UUID` | ID of the scored message piece. |
| `id` | `Optional[uuid.UUID | str]` | Optional score ID. Defaults to `None`. |
| `score_category` | `Optional[List[str]]` | Optional score categories. Defaults to `None`. |
| `score_metadata` | `Optional[Dict[str, Union[str, int, float]]]` | Optional metadata. Defaults to `None`. |
| `scorer_class_identifier` | `Union[ScorerIdentifier, Dict[str, Any]]` | Scorer identifier. |
| `timestamp` | `Optional[datetime]` | Optional creation timestamp. Defaults to `None`. |
| `objective` | `Optional[str]` | Optional task objective. Defaults to `None`. |

**Methods:**

#### `get_value() → bool | float`

Return the value of the score based on its type.

If the score type is "true_false", it returns True if the score value is "true" (case-insensitive),
otherwise it returns False.

If the score type is "float_scale", it returns the score value as a float.

**Returns:**

- `bool | float` — bool | float: Parsed score value.

**Raises:**

- `ValueError` — If the score type is unknown.

#### `to_dict() → dict[str, Any]`

Convert this score to a dictionary.

**Returns:**

- `dict[str, Any]` — Dict[str, Any]: Serialized score payload.

#### `validate(scorer_type: str, score_value: str) → None`

Validate score value against scorer type constraints.

| Parameter | Type | Description |
|---|---|---|
| `scorer_type` | `str` | Scorer type to validate against. |
| `score_value` | `str` | Raw score value. |

**Raises:**

- `ValueError` — If value is incompatible with scorer type constraints.

## `class Seed(YamlLoadable)`

Represents seed data with various attributes and metadata.

**Methods:**

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → Seed
```

Load a Seed from a YAML file and validate that it contains specific parameters.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the template. |
| `required_parameters` | `list[str]` | List of parameter names that must exist in the template. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. If None, a default message is used. Defaults to `None`. |

**Returns:**

- `Seed` — The loaded and validated seed of the specific subclass type.

#### `render_template_value(kwargs: Any = {}) → str`

Render self.value as a template with provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Any` | Key-value pairs to replace in the SeedPrompt value. Defaults to `{}`. |

**Returns:**

- `str` — A new prompt with the parameters applied.

**Raises:**

- `ValueError` — If parameters are missing or invalid in the template.

#### `render_template_value_silent(kwargs: Any = {}) → str`

Render self.value as a template with provided parameters. For parameters in the template
that are not provided as kwargs here, this function will leave them as is instead of raising an error.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Any` | Key-value pairs to replace in the SeedPrompt value. Defaults to `{}`. |

**Returns:**

- `str` — A new prompt with the parameters applied.

**Raises:**

- `ValueError` — If parameters are missing or invalid in the template.

#### `set_sha256_value_async() → None`

Compute the SHA256 hash value asynchronously.
It should be called after prompt `value` is serialized to text,
as file paths used in the `value` may have changed from local to memory storage paths.

Note, this method is async due to the blob retrieval. And because of that, we opted
to take it out of main and setter functions. The disadvantage is that it must be explicitly called.

## `class SeedAttackGroup(SeedGroup)`

A group of seeds for use in attack scenarios.

This class extends SeedGroup with attack-specific validation:
- Requires exactly one SeedObjective (not optional like in SeedGroup)

All other functionality (simulated conversation, prepended conversation,
next_message, etc.) is inherited from SeedGroup.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Union[Seed, dict[str, Any]]]` | Sequence of seeds. Must include exactly one SeedObjective. |

**Methods:**

#### `validate() → None`

Validate the seed attack group state.

Extends SeedGroup validation to require exactly one objective.

**Raises:**

- `ValueError` — If validation fails.

## `class SeedAttackTechniqueGroup(SeedGroup)`

A group of seeds representing a general attack technique.

This class extends SeedGroup with technique-specific validation:
- Requires all seeds to have is_general_technique=True

All other functionality (simulated conversation, prepended conversation,
next_message, etc.) is inherited from SeedGroup.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Union[Seed, dict[str, Any]]]` | Sequence of seeds. All seeds must have is_general_technique=True. |

**Methods:**

#### `validate() → None`

Validate the seed attack technique group state.

Extends SeedGroup validation to require all seeds to be general strategies.

**Raises:**

- `ValueError` — If validation fails.

## `class SeedDataset(YamlLoadable)`

SeedDataset manages seed prompts plus optional top-level defaults.
Prompts are stored as a Sequence[Seed], so references to prompt properties
are straightforward (e.g. ds.seeds[0].value).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Optional[Union[Sequence[dict[str, Any]], Sequence[Seed]]]` | List of seed dictionaries or Seed objects. Defaults to `None`. |
| `data_type` | `Optional[PromptDataType]` | Default data type for seeds. Defaults to `'text'`. |
| `name` | `Optional[str]` | Name of the dataset. Defaults to `None`. |
| `dataset_name` | `Optional[str]` | Dataset name for categorization. Defaults to `None`. |
| `harm_categories` | `Optional[Sequence[str]]` | List of harm categories. Defaults to `None`. |
| `description` | `Optional[str]` | Description of the dataset. Defaults to `None`. |
| `authors` | `Optional[Sequence[str]]` | List of authors. Defaults to `None`. |
| `groups` | `Optional[Sequence[str]]` | List of groups. Defaults to `None`. |
| `source` | `Optional[str]` | Source of the dataset. Defaults to `None`. |
| `date_added` | `Optional[datetime]` | Date when the dataset was added. Defaults to `None`. |
| `added_by` | `Optional[str]` | User who added the dataset. Defaults to `None`. |
| `seed_type` | `Optional[SeedType]` | The type of seeds in this dataset ("prompt", "objective", or "simulated_conversation"). Defaults to `None`. |
| `is_objective` | `bool` | Deprecated in 0.13.0. Use seed_type="objective" instead. Defaults to `False`. |

**Methods:**

#### `from_dict(data: dict[str, Any]) → SeedDataset`

Build a SeedDataset by merging top-level defaults into each item in `seeds`.

| Parameter | Type | Description |
|---|---|---|
| `data` | `Dict[str, Any]` | Dataset payload with top-level defaults and seed entries. |

**Returns:**

- `SeedDataset` — Constructed dataset with merged defaults.

**Raises:**

- `ValueError` — If any seed entry includes a pre-set prompt_group_id.

#### get_random_values

```python
get_random_values(number: PositiveInt, harm_categories: Optional[Sequence[str]] = None) → Sequence[str]
```

Extract and return random prompt values from the dataset.

| Parameter | Type | Description |
|---|---|---|
| `number` | `int` | The number of random prompt values to return. |
| `harm_categories` | `Optional[Sequence[str]]` | If provided, only prompts containing at least one of these harm categories are included. Defaults to `None`. |

**Returns:**

- `Sequence[str]` — Sequence[str]: A list of prompt values.

#### get_values

```python
get_values(first: Optional[PositiveInt] = None, last: Optional[PositiveInt] = None, harm_categories: Optional[Sequence[str]] = None) → Sequence[str]
```

Extract and return prompt values from the dataset.

| Parameter | Type | Description |
|---|---|---|
| `first` | `Optional[int]` | If provided, values from the first N prompts are included. Defaults to `None`. |
| `last` | `Optional[int]` | If provided, values from the last N prompts are included. Defaults to `None`. |
| `harm_categories` | `Optional[Sequence[str]]` | If provided, only prompts containing at least one of these harm categories are included. Defaults to `None`. |

**Returns:**

- `Sequence[str]` — Sequence[str]: A list of prompt values.

#### group_seed_prompts_by_prompt_group_id

```python
group_seed_prompts_by_prompt_group_id(seeds: Sequence[Seed]) → Sequence[SeedGroup]
```

Group the given list of seeds by prompt_group_id and create
SeedGroup or SeedAttackGroup instances.

For each group, this method first attempts to create a SeedAttackGroup
(which has attack-specific properties like objective). If validation fails,
it falls back to a basic SeedGroup.

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Seed]` | A list of Seed objects. |

**Returns:**

- `Sequence[SeedGroup]` — A list of SeedGroup or SeedAttackGroup objects, with seeds grouped by
- `Sequence[SeedGroup]` — prompt_group_id. Each group will be ordered by the sequence number of
- `Sequence[SeedGroup]` — the seeds, if available.

#### `render_template_value(kwargs: object = {}) → None`

Render seed values as templates using provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `object` | Key-value pairs to replace in the SeedDataset value. Defaults to `{}`. |

**Raises:**

- `ValueError` — If parameters are missing or invalid in the template.

## `class SeedGroup(YamlLoadable)`

A container for grouping prompts that need to be sent together.

This class handles:
- Grouping of SeedPrompt, SeedObjective, and SeedSimulatedConversation
- Consistent group IDs and roles across seeds
- Prepended conversation and next message extraction
- Validation of sequence overlaps between SeedPrompts and SeedSimulatedConversation

All prompts in the group share the same `prompt_group_id`.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Union[Seed, dict[str, Any]]]` | Sequence of seeds. Can include: - SeedObjective (or dict with seed_type="objective") - SeedSimulatedConversation (or dict with seed_type="simulated_conversation") - SeedPrompt for prompts (or dict with seed_type="prompt" or no seed_type) Note: is_objective and is_simulated_conversation are deprecated since 0.13.0. |

**Methods:**

#### `is_single_part_single_text_request() → bool`

Check if this is a single text prompt.

**Returns:**

- `bool` — True when there is exactly one prompt and it is text.

#### `is_single_request() → bool`

Check if all prompts are in a single sequence.

**Returns:**

- `bool` — True when all prompts share one sequence number.

#### `is_single_turn() → bool`

Check if this is a single-turn group (single request without objective).

**Returns:**

- `bool` — True when the group is a single request and has no objective.

#### `render_template_value(kwargs: Any = {}) → None`

Render seed values as templates with provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Any` | Key-value pairs to replace in seed values. Defaults to `{}`. |

#### `validate() → None`

Validate the seed group state.

This method can be called after external modifications to seeds
to ensure the group remains in a valid state. It is automatically
called during initialization.

**Raises:**

- `ValueError` — If validation fails.

## `class SeedObjective(Seed)`

Represents a seed objective with various attributes and metadata.

**Methods:**

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → SeedObjective
```

Load a Seed from a YAML file. Because SeedObjectives do not have any parameters, the required_parameters
and error_message arguments are unused.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the template. |
| `required_parameters` | `list[str]` | List of parameter names that must exist in the template. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. If None, a default message is used. Defaults to `None`. |

**Returns:**

- `SeedObjective` — The loaded and validated seed of the specific subclass type.

## `class SeedPrompt(Seed)`

Represents a seed prompt with various attributes and metadata.

**Methods:**

#### from_messages

```python
from_messages(messages: list[Message], starting_sequence: int = 0, prompt_group_id: Optional[uuid.UUID] = None) → list[SeedPrompt]
```

Convert a list of Messages to a list of SeedPrompts.

Each MessagePiece becomes a SeedPrompt. All pieces from the same message
share the same sequence number, preserving the grouping.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | List of Messages to convert. |
| `starting_sequence` | `int` | The starting sequence number. Defaults to 0. Defaults to `0`. |
| `prompt_group_id` | `Optional[uuid.UUID]` | Optional group ID to assign to all prompts. Defaults to None. Defaults to `None`. |

**Returns:**

- `list[SeedPrompt]` — List of SeedPrompts with incrementing sequence numbers per message.

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → SeedPrompt
```

Load a Seed from a YAML file and validate that it contains specific parameters.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the template. |
| `required_parameters` | `list[str]` | List of parameter names that must exist in the template. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. If None, a default message is used. Defaults to `None`. |

**Returns:**

- `SeedPrompt` — The loaded and validated SeedPrompt of the specific subclass type.

**Raises:**

- `ValueError` — If the template doesn't contain all required parameters.

#### `set_encoding_metadata() → None`

Set encoding metadata for the prompt within metadata dictionary. For images, this is just the
file format. For audio and video, this also includes bitrate (kBits/s as int), samplerate (samples/second
as int), bitdepth (as int), filesize (bytes as int), and duration (seconds as int) if the file type is
supported by TinyTag. Example supported file types include: MP3, MP4, M4A, and WAV.

## `class SeedSimulatedConversation(Seed)`

Configuration for generating a simulated conversation dynamically.

This class holds the paths and parameters needed to generate prepended conversation
content by running an adversarial chat against a simulated (compliant) target.

This is a pure configuration class. The actual generation is performed by
`generate_simulated_conversation_async` in the executor layer, which accepts
this config along with runtime dependencies (adversarial_chat target, scorer).

The `value` property returns a JSON serialization of the config for database
storage and deduplication.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `adversarial_chat_system_prompt_path` | `Union[str, Path]` | Path to YAML file containing the adversarial chat system prompt. |
| `simulated_target_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to YAML file containing the simulated target system prompt. Defaults to the compliant prompt. Defaults to `None`. |
| `next_message_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to YAML file containing the system prompt for generating a final user message. If provided, after the simulated conversation is generated, a single LLM call generates a user message that attempts to get the target to fulfill the objective. Defaults to None (no next message generation). Defaults to `None`. |
| `num_turns` | `int` | Number of conversation turns to generate. Defaults to 3. Defaults to `3`. |
| `sequence` | `int` | The starting sequence number for generated turns. When combined with static SeedPrompts, this determines where the simulated turns are inserted. Defaults to 0. Defaults to `0`. |
| `pyrit_version` | `Optional[str]` | PyRIT version for reproducibility tracking. Defaults to current version. Defaults to `None`. |
| `**kwargs` | `Any` | Additional arguments passed to the Seed base class. Defaults to `{}`. |

**Methods:**

#### `compute_hash() → str`

Compute a deterministic hash of this configuration.

**Returns:**

- `str` — A SHA256 hash string representing the configuration.

#### `from_dict(data: dict[str, Any]) → SeedSimulatedConversation`

Create a SeedSimulatedConversation from a dictionary, typically from YAML.

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict[str, Any]` | Dictionary containing the configuration. |

**Returns:**

- `SeedSimulatedConversation` — A new SeedSimulatedConversation instance.

**Raises:**

- `ValueError` — If required configuration fields are missing.

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → SeedSimulatedConversation
```

Load a SeedSimulatedConversation from a YAML file and validate required parameters.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the config. |
| `required_parameters` | `list[str]` | List of parameter names that must exist. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. Defaults to `None`. |

**Returns:**

- `SeedSimulatedConversation` — The loaded and validated SeedSimulatedConversation.

**Raises:**

- `ValueError` — If required parameters are missing.

#### `get_identifier() → dict[str, Any]`

Get an identifier dict capturing this configuration for comparison/storage.

**Returns:**

- `dict[str, Any]` — Dictionary with configuration details.

#### load_simulated_target_system_prompt

```python
load_simulated_target_system_prompt(objective: str, num_turns: int, simulated_target_system_prompt_path: Optional[Union[str, Path]] = None) → Optional[str]
```

Load and render the simulated target system prompt.

If no path is provided, returns None (no system prompt).
Validates that the template has required `objective` and `num_turns` parameters.

| Parameter | Type | Description |
|---|---|---|
| `objective` | `str` | The objective to render into the template. |
| `num_turns` | `int` | The number of turns to render into the template. |
| `simulated_target_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to the prompt YAML file. If None, no system prompt is used. Defaults to `None`. |

**Returns:**

- `Optional[str]` — The rendered system prompt string, or None if no path is provided.

**Raises:**

- `ValueError` — If the template doesn't have required parameters.

## `class SimulatedTargetSystemPromptPaths(enum.Enum)`

Enum for predefined simulated target system prompt paths.

## `class StorageIO(ABC)`

Abstract interface for storage systems (local disk, Azure Storage Account, etc.).

**Methods:**

#### `create_directory_if_not_exists(path: Union[Path, str]) → None`

Asynchronously creates a directory or equivalent in the storage system if it doesn't exist.

#### `is_file(path: Union[Path, str]) → bool`

Asynchronously checks if the path refers to a file (not a directory or container).

#### `path_exists(path: Union[Path, str]) → bool`

Asynchronously checks if a file or blob exists at the given path.

#### `read_file(path: Union[Path, str]) → bytes`

Asynchronously reads the file (or blob) from the given path.

#### `write_file(path: Union[Path, str], data: bytes) → None`

Asynchronously writes data to the given path.

## `class StrategyResult(ABC)`

Base class for all strategy results.

**Methods:**

#### `duplicate() → StrategyResultT`

Create a deep copy of the result.

**Returns:**

- `StrategyResultT` — A deep copy of the result.

## `class TextDataTypeSerializer(DataTypeSerializer)`

Serializer for text and text-like prompt values that stay in-memory.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt_text` | `str` | Prompt value. |
| `data_type` | `PromptDataType` | Text-like prompt data type. Defaults to `'text'`. |

**Methods:**

#### `data_on_disk() → bool`

Indicate whether this serializer persists data on disk.

**Returns:**

- `bool` — Always False for text serializers.

## `class UnvalidatedScore`

Score is an object that validates all the fields. However, we need a common
data class that can be used to store the raw score value before it is normalized and validated.

**Methods:**

#### `to_score(score_value: str, score_type: ScoreType) → Score`

Convert this unvalidated score into a validated Score.

| Parameter | Type | Description |
|---|---|---|
| `score_value` | `str` | Normalized score value. |
| `score_type` | `ScoreType` | Score type. |

**Returns:**

- `Score` — Validated score object.

## `class VideoPathDataTypeSerializer(DataTypeSerializer)`

Serializer for video path values stored on disk.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | The category or context for the data. |
| `prompt_text` | `Optional[str]` | The video path or identifier. Defaults to `None`. |
| `extension` | `Optional[str]` | The file extension, defaults to 'mp4'. Defaults to `None`. |

**Methods:**

#### `data_on_disk() → bool`

Indicate whether this serializer persists data on disk.

**Returns:**

- `bool` — Always True for video path serializers.
