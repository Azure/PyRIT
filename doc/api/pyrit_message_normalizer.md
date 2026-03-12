# pyrit.message_normalizer

Functionality to normalize messages into compatible formats for targets.

## `class ChatMessageNormalizer(MessageListNormalizer[ChatMessage], MessageStringNormalizer)`

Normalizer that converts a list of Messages to a list of ChatMessages.

This normalizer handles both single-part and multipart messages:
- Single piece messages: content is a simple string
- Multiple piece messages: content is a list of dicts with type/text or type/image_url

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `use_developer_role` | `bool` | If True, translates "system" role to "developer" role. Defaults to `False`. |
| `system_message_behavior` | `SystemMessageBehavior` | How to handle system messages. Defaults to "keep". Defaults to `'keep'`. |

**Methods:**

#### `normalize_async(messages: list[Message]) → list[ChatMessage]`

Convert a list of Messages to a list of ChatMessages.

For single-piece text messages, content is a string.
For multi-piece or non-text messages, content is a list of content dicts.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of Message objects to normalize. |

**Returns:**

- `list[ChatMessage]` — A list of ChatMessage objects.

**Raises:**

- `ValueError` — If the messages list is empty.

#### `normalize_string_async(messages: list[Message]) → str`

Convert a list of Messages to a JSON string representation.

This serializes the list of ChatMessages to JSON format.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of Message objects to normalize. |

**Returns:**

- `str` — A JSON string representation of the ChatMessages.

## `class ConversationContextNormalizer(MessageStringNormalizer)`

Normalizer that formats conversation history as turn-based text.

This is the standard format used by attacks like Crescendo and TAP
for including conversation context in adversarial chat prompts.
The output format is:

    Turn 1:
    User: <content>
    Assistant: <content>

    Turn 2:
    User: <content>
    ...

**Methods:**

#### `normalize_string_async(messages: list[Message]) → str`

Normalize a list of messages into a turn-based context string.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of Message objects to normalize. |

**Returns:**

- `str` — A formatted string with turn numbers and role prefixes.

**Raises:**

- `ValueError` — If the messages list is empty.

## `class GenericSystemSquashNormalizer(MessageListNormalizer[Message])`

Normalizer that combines the first system message with the first user message using generic instruction tags.

**Methods:**

#### `normalize_async(messages: list[Message]) → list[Message]`

Return messages with the first system message combined into the first user message.

The format uses generic instruction tags:
### Instructions ###
{system_content}
######
{user_content}

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of messages to normalize. |

**Returns:**

- `list[Message]` — A Message with the system message squashed into the first user message.

**Raises:**

- `ValueError` — If the messages list is empty.

## `class MessageListNormalizer(abc.ABC, Generic[T])`

Abstract base class for normalizers that return a list of items.

Subclasses specify the type T (e.g., Message, ChatMessage) that the list contains.
T must implement the DictConvertible protocol (have a to_dict() method).

**Methods:**

#### `normalize_async(messages: list[Message]) → list[T]`

Normalize the list of messages into a list of items.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of Message objects to normalize. |

**Returns:**

- `list[T]` — A list of normalized items of type T.

#### `normalize_to_dicts_async(messages: list[Message]) → list[dict[str, Any]]`

Normalize the list of messages into a list of dictionaries.

This method uses normalize_async and calls to_dict() on each item.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of Message objects to normalize. |

**Returns:**

- `list[dict[str, Any]]` — A list of dictionaries representing the normalized messages.

## `class MessageStringNormalizer(abc.ABC)`

Abstract base class for normalizers that return a string representation.

Use this for formatting messages into text for non-chat targets or context strings.

**Methods:**

#### `normalize_string_async(messages: list[Message]) → str`

Normalize the list of messages into a string representation.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The list of Message objects to normalize. |

**Returns:**

- `str` — A string representation of the messages.

## `class TokenizerTemplateNormalizer(MessageStringNormalizer)`

Enable application of the chat template stored in a Hugging Face tokenizer
to a list of messages. For more details, see
https://huggingface.co/docs/transformers/main/en/chat_templating.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tokenizer` | `PreTrainedTokenizerBase` | A Hugging Face tokenizer with a chat template. |
| `system_message_behavior` | `TokenizerSystemBehavior` | How to handle system messages. Options: - "keep": Keep system messages as-is (default) - "squash": Merge system into first user message - "ignore": Drop system messages entirely - "developer": Change system role to developer role Defaults to `'keep'`. |

**Methods:**

#### from_model

```python
from_model(model_name_or_alias: str, token: Optional[str] = None, system_message_behavior: Optional[TokenizerSystemBehavior] = None) → TokenizerTemplateNormalizer
```

Create a normalizer from a model name or alias.

This factory method simplifies creating a normalizer by handling tokenizer
loading automatically. Use aliases for common models or provide a full
HuggingFace model path.

| Parameter | Type | Description |
|---|---|---|
| `model_name_or_alias` | `str` | Either a full HuggingFace model name or an alias (e.g., 'chatml', 'phi3', 'llama3'). See MODEL_ALIASES for available aliases. |
| `token` | `Optional[str]` | Optional HuggingFace token for gated models. If not provided, falls back to HUGGINGFACE_TOKEN environment variable. Defaults to `None`. |
| `system_message_behavior` | `Optional[TokenizerSystemBehavior]` | Override how to handle system messages. If not provided, uses the model's default config. Defaults to `None`. |

**Returns:**

- `TokenizerTemplateNormalizer` — TokenizerTemplateNormalizer configured with the model's tokenizer.

**Raises:**

- `ValueError` — If the tokenizer doesn't have a chat_template.

#### `normalize_string_async(messages: list[Message]) → str`

Apply the chat template stored in the tokenizer to a list of messages.

Handles system messages based on the configured system_message_behavior:
- "keep": Pass system messages as-is
- "squash": Merge system into first user message
- "ignore": Drop system messages entirely
- "developer": Change system role to developer role

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | A list of Message objects. |

**Returns:**

- `str` — The formatted chat messages as a string.
