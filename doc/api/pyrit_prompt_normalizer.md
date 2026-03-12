# pyrit.prompt_normalizer

Prompt normalization components for standardizing and converting prompts.

This module provides tools for normalizing prompts before sending them to targets,
including converter configurations and request handling.

## `class NormalizerRequest`

Represents a single request sent to normalizer.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `message` | `Message` | The message to be normalized. |
| `request_converter_configurations` | `list[PromptConverterConfiguration]` | Configurations for converting the request. Defaults to an empty list. Defaults to `None`. |
| `response_converter_configurations` | `list[PromptConverterConfiguration]` | Configurations for converting the response. Defaults to an empty list. Defaults to `None`. |
| `conversation_id` | `Optional[str]` | The ID of the conversation. Defaults to None. Defaults to `None`. |

## `class PromptConverterConfiguration`

Represents the configuration for a prompt response converter.

The list of converters are applied to a response, which can have multiple response pieces.
indexes_to_apply are which pieces to apply to. By default, all indexes are applied.
prompt_data_types_to_apply are the types of the responses to apply the converters.

**Methods:**

#### from_converters

```python
from_converters(converters: list[PromptConverter]) → list[PromptConverterConfiguration]
```

Convert a list of converters into a list of PromptConverterConfiguration objects.
Each converter gets its own configuration with default settings.

| Parameter | Type | Description |
|---|---|---|
| `converters` | `list[PromptConverter]` | List of PromptConverters |

**Returns:**

- `list[PromptConverterConfiguration]` — List[PromptConverterConfiguration]: List of configurations, one per converter

## `class PromptNormalizer`

Handles normalization and processing of prompts before they are sent to targets.

**Methods:**

#### add_prepended_conversation_to_memory

```python
add_prepended_conversation_to_memory(conversation_id: str, should_convert: bool = True, converter_configurations: Optional[list[PromptConverterConfiguration]] = None, attack_identifier: Optional[ComponentIdentifier] = None, prepended_conversation: Optional[list[Message]] = None) → Optional[list[Message]]
```

Process the prepended conversation by converting it if needed and adding it to memory.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation ID to use for the message pieces |
| `should_convert` | `bool` | Whether to convert the prepended conversation Defaults to `True`. |
| `converter_configurations` | `Optional[list[PromptConverterConfiguration]]` | Configurations for converting the request Defaults to `None`. |
| `attack_identifier` | `Optional[ComponentIdentifier]` | Identifier for the attack Defaults to `None`. |
| `prepended_conversation` | `Optional[list[Message]]` | The conversation to prepend Defaults to `None`. |

**Returns:**

- `Optional[list[Message]]` — Optional[list[Message]]: The processed prepended conversation

#### convert_values

```python
convert_values(converter_configurations: list[PromptConverterConfiguration], message: Message) → None
```

Apply converter configurations to message pieces.

| Parameter | Type | Description |
|---|---|---|
| `converter_configurations` | `list[PromptConverterConfiguration]` | List of configurations specifying which converters to apply and to which message pieces. |
| `message` | `Message` | The message containing pieces to be converted. |

**Raises:**

- `Exception` — Any exception from converters propagates with execution context for error tracing.

#### send_prompt_async

```python
send_prompt_async(message: Message, target: PromptTarget, conversation_id: Optional[str] = None, request_converter_configurations: list[PromptConverterConfiguration] | None = None, response_converter_configurations: list[PromptConverterConfiguration] | None = None, labels: Optional[dict[str, str]] = None, attack_identifier: Optional[ComponentIdentifier] = None) → Message
```

Send a single request to a target.

| Parameter | Type | Description |
|---|---|---|
| `message` | `Message` | The message to be sent. |
| `target` | `PromptTarget` | The target to which the prompt is sent. |
| `conversation_id` | `str` | The ID of the conversation. Defaults to None. Defaults to `None`. |
| `request_converter_configurations` | `list[PromptConverterConfiguration]` | Configurations for converting the request. Defaults to an empty list. Defaults to `None`. |
| `response_converter_configurations` | `list[PromptConverterConfiguration]` | Configurations for converting the response. Defaults to an empty list. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | Labels associated with the request. Defaults to None. Defaults to `None`. |
| `attack_identifier` | `Optional[ComponentIdentifier]` | Identifier for the attack. Defaults to None. Defaults to `None`. |

**Returns:**

- `Message` — The response received from the target.

**Raises:**

- `Exception` — If an error occurs during the request processing.
- `ValueError` — If the message pieces are not part of the same sequence.

#### send_prompt_batch_to_target_async

```python
send_prompt_batch_to_target_async(requests: list[NormalizerRequest], target: PromptTarget, labels: Optional[dict[str, str]] = None, attack_identifier: Optional[ComponentIdentifier] = None, batch_size: int = 10) → list[Message]
```

Send a batch of prompts to the target asynchronously.

| Parameter | Type | Description |
|---|---|---|
| `requests` | `list[NormalizerRequest]` | A list of NormalizerRequest objects to be sent. |
| `target` | `PromptTarget` | The target to which the prompts are sent. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of labels to be included with the request. Defaults to None. Defaults to `None`. |
| `attack_identifier` | `Optional[ComponentIdentifier]` | The attack identifier. Defaults to None. Defaults to `None`. |
| `batch_size` | `int` | The number of prompts to include in each batch. Defaults to 10. Defaults to `10`. |

**Returns:**

- `list[Message]` — list[Message]: A list of Message objects representing the responses
received for each prompt.
