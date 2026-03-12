# pyrit.executor.attack.component

Attack components module.

## Functions

### `build_conversation_context_string_async(messages: list[Message]) → str`

Build a formatted context string from a list of messages.

This is a convenience function that uses ConversationContextNormalizer
to format messages into a "Turn N: User/Assistant" format suitable for
use in system prompts.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | The conversation messages to format. |

**Returns:**

- `str` — A formatted string representing the conversation context.
- `str` — Returns empty string if no messages provided.

### get_adversarial_chat_messages

```python
get_adversarial_chat_messages(prepended_conversation: list[Message], adversarial_chat_conversation_id: str, attack_identifier: ComponentIdentifier, adversarial_chat_target_identifier: ComponentIdentifier, labels: Optional[dict[str, str]] = None) → list[Message]
```

Transform prepended conversation messages for adversarial chat with swapped roles.

This function creates new Message objects with swapped roles for use in adversarial
chat conversations. From the adversarial chat's perspective:
- "user" messages become "assistant" (prompts it generated)
- "assistant" messages become "user" (responses it received)
- System messages are skipped (adversarial chat has its own system prompt)

All messages receive new UUIDs to distinguish them from the originals.

| Parameter | Type | Description |
|---|---|---|
| `prepended_conversation` | `list[Message]` | The original conversation messages to transform. |
| `adversarial_chat_conversation_id` | `str` | Conversation ID for the adversarial chat. |
| `attack_identifier` | `ComponentIdentifier` | Attack identifier to associate with messages. |
| `adversarial_chat_target_identifier` | `ComponentIdentifier` | Target identifier for the adversarial chat. |
| `labels` | `Optional[dict[str, str]]` | Optional labels to associate with the messages. Defaults to `None`. |

**Returns:**

- `list[Message]` — List of transformed messages with swapped roles and new IDs.

### `get_prepended_turn_count(prepended_conversation: Optional[list[Message]]) → int`

Count the number of turns (assistant responses) in a prepended conversation.

This is used to offset iteration counts so that executed_turns reflects
the total conversation depth including prepended messages.

| Parameter | Type | Description |
|---|---|---|
| `prepended_conversation` | `Optional[list[Message]]` | The prepended conversation messages, or None. |

**Returns:**

- `int` — The number of assistant messages in the prepended conversation.
Returns 0 if prepended_conversation is None or empty.

### `mark_messages_as_simulated(messages: Sequence[Message]) → list[Message]`

Mark assistant messages as simulated_assistant for traceability.

This function converts all assistant roles to simulated_assistant in the
provided messages. This is useful when loading conversations from YAML files
or other sources where the responses are not from actual targets.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `Sequence[Message]` | The messages to mark as simulated. |

**Returns:**

- `list[Message]` — List[Message]: The same messages with assistant roles converted to simulated_assistant.
Modifies the messages in place and also returns them for convenience.

## `class ConversationManager`

Manages conversations for attacks, handling message history,
system prompts, and conversation state.

This class provides methods to:
- Initialize attack context with prepended conversations
- Retrieve conversation history
- Set system prompts for chat targets

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `attack_identifier` | `ComponentIdentifier` | The identifier of the attack this manager belongs to. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Optional prompt normalizer for converting prompts. If not provided, a default PromptNormalizer instance will be created. Defaults to `None`. |

**Methods:**

#### add_prepended_conversation_to_memory_async

```python
add_prepended_conversation_to_memory_async(prepended_conversation: list[Message], conversation_id: str, request_converters: Optional[list[PromptConverterConfiguration]] = None, prepended_conversation_config: Optional[PrependedConversationConfig] = None, max_turns: Optional[int] = None) → int
```

Add prepended conversation messages to memory for a chat target.

This is a lower-level method that handles adding messages to memory without
modifying any attack context state. It can be called directly by attacks
that manage their own state (like TAP nodes) or internally by
initialize_context_async for standard attacks.

Messages are added with:
- Duplicated message objects (preserves originals)
- simulated_assistant role for assistant messages (for traceability)
- Converters applied based on config

| Parameter | Type | Description |
|---|---|---|
| `prepended_conversation` | `list[Message]` | Messages to add to memory. |
| `conversation_id` | `str` | Conversation ID to assign to all messages. |
| `request_converters` | `Optional[list[PromptConverterConfiguration]]` | Optional converters to apply to messages. Defaults to `None`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfig]` | Optional configuration for converter roles. Defaults to `None`. |
| `max_turns` | `Optional[int]` | If provided, validates that turn count doesn't exceed this limit. Defaults to `None`. |

**Returns:**

- `int` — The number of turns (assistant messages) added.

**Raises:**

- `ValueError` — If max_turns is exceeded by the prepended conversation.

#### `get_conversation(conversation_id: str) → list[Message]`

Retrieve a conversation by its ID.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The ID of the conversation to retrieve. |

**Returns:**

- `list[Message]` — A list of messages in the conversation, ordered by creation time.
- `list[Message]` — Returns empty list if no messages exist.

#### get_last_message

```python
get_last_message(conversation_id: str, role: Optional[ChatMessageRole] = None) → Optional[MessagePiece]
```

Retrieve the most recent message from a conversation.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The ID of the conversation to retrieve from. |
| `role` | `Optional[ChatMessageRole]` | If provided, return only the last message matching this role. Defaults to `None`. |

**Returns:**

- `Optional[MessagePiece]` — The last message piece, or None if no messages exist.

#### initialize_context_async

```python
initialize_context_async(context: AttackContext[Any], target: PromptTarget, conversation_id: str, request_converters: Optional[list[PromptConverterConfiguration]] = None, prepended_conversation_config: Optional[PrependedConversationConfig] = None, max_turns: Optional[int] = None, memory_labels: Optional[dict[str, str]] = None) → ConversationState
```

Initialize attack context with prepended conversation and merged labels.

This is the primary method for setting up an attack context. It:
1. Merges memory_labels from attack strategy with context labels
2. Processes prepended_conversation based on target type and config
3. Updates context.executed_turns for multi-turn attacks
4. Sets context.next_message if there's an unanswered user message

| Parameter | Type | Description |
|---|---|---|
| `context` | `AttackContext[Any]` | The attack context to initialize. |
| `target` | `PromptTarget` | The objective target for the conversation. |
| `conversation_id` | `str` | Unique identifier for the conversation. |
| `request_converters` | `Optional[list[PromptConverterConfiguration]]` | Converters to apply to messages. Defaults to `None`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfig]` | Configuration for handling prepended conversation. Defaults to `None`. |
| `max_turns` | `Optional[int]` | Maximum turns allowed (for validation and state tracking). Defaults to `None`. |
| `memory_labels` | `Optional[dict[str, str]]` | Labels from the attack strategy to merge with context labels. Defaults to `None`. |

**Returns:**

- `ConversationState` — ConversationState with turn_count and last_assistant_message_scores.

**Raises:**

- `ValueError` — If conversation_id is empty, or if prepended_conversation
requires a PromptChatTarget but target is not one.

#### set_system_prompt

```python
set_system_prompt(target: PromptChatTarget, conversation_id: str, system_prompt: str, labels: Optional[dict[str, str]] = None) → None
```

Set or update the system prompt for a conversation.

| Parameter | Type | Description |
|---|---|---|
| `target` | `PromptChatTarget` | The chat target to set the system prompt on. |
| `conversation_id` | `str` | Unique identifier for the conversation. |
| `system_prompt` | `str` | The system prompt text. |
| `labels` | `Optional[dict[str, str]]` | Optional labels to associate with the system prompt. Defaults to `None`. |

## `class ConversationState`

Container for conversation state data returned from context initialization.

## `class PrependedConversationConfig`

Configuration for controlling how prepended conversations are processed before
being sent to the objective target.

This class provides control over:
- Which message roles should have request converters applied
- How to normalize conversation history for non-chat objective targets
- What to do when the objective target is not a PromptChatTarget

**Methods:**

#### `default() → PrependedConversationConfig`

Create a default configuration with converters applied to all roles.

**Returns:**

- `PrependedConversationConfig` — A configuration that applies converters to all prepended messages,
- `PrependedConversationConfig` — raising an error for non-chat targets.

#### for_non_chat_target

```python
for_non_chat_target(message_normalizer: Optional[MessageStringNormalizer] = None, apply_converters_to_roles: Optional[list[ChatMessageRole]] = None) → PrependedConversationConfig
```

Create a configuration for use with non-chat targets.

This configuration normalizes the prepended conversation into a text block
that will be prepended to the first message sent to the target.

| Parameter | Type | Description |
|---|---|---|
| `message_normalizer` | `Optional[MessageStringNormalizer]` | Normalizer for formatting the prepended conversation into a string. Defaults to ConversationContextNormalizer if not provided. Defaults to `None`. |
| `apply_converters_to_roles` | `Optional[list[ChatMessageRole]]` | Roles to apply converters to before normalization. Defaults to all roles. Defaults to `None`. |

**Returns:**

- `PrependedConversationConfig` — A configuration that normalizes the prepended conversation for non-chat targets.

#### `get_message_normalizer() → MessageStringNormalizer`

Get the normalizer for objective target context, with a default fallback.

**Returns:**

- `MessageStringNormalizer` — The configured objective_target_context_normalizer, or a default
- `MessageStringNormalizer` — ConversationContextNormalizer if none was configured.
