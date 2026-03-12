# pyrit.analytics

Analytics module for PyRIT conversation and result analysis.

## Functions

### analyze_results

```python
analyze_results(attack_results: list[AttackResult]) → dict[str, AttackStats | dict[str, AttackStats]]
```

Analyze a list of AttackResult objects and return overall and grouped statistics.

**Returns:**

- `dict[str, AttackStats | dict[str, AttackStats]]` — A dictionary of AttackStats objects. The overall stats are accessible with the key
- `dict[str, AttackStats | dict[str, AttackStats]]` — "Overall", and the stats of any attack can be retrieved using "By_attack_identifier"
- `dict[str, AttackStats | dict[str, AttackStats]]` — followed by the identifier of the attack.

**Raises:**

- `ValueError` — if attack_results is empty.
- `TypeError` — if any element is not an AttackResult.

## `class ApproximateTextMatching(TextMatching)`

Approximate text matching using n-gram overlap.

This strategy computes the proportion of character n-grams from the target
that are present in the text. Useful for detecting partial matches, encoded
content, or text with variations.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `threshold` | `float` | The minimum n-gram overlap score (0.0 to 1.0) required for a match. Defaults to 0.5 (50% overlap). Defaults to `0.5`. |
| `n` | `int` | The length of character n-grams to use. Defaults to 3. Defaults to `3`. |
| `case_sensitive` | `bool` | Whether to perform case-sensitive matching. Defaults to False. Defaults to `False`. |

**Methods:**

#### `get_overlap_score(target: str, text: str) → float`

Get the n-gram overlap score without threshold comparison.

Useful for getting detailed scoring information.

| Parameter | Type | Description |
|---|---|---|
| `target` | `str` | The target string to match. |
| `text` | `str` | The text to search in. |

**Returns:**

- `float` — The n-gram overlap score between 0.0 and 1.0.

#### `is_match(target: str, text: str) → bool`

Check if target approximately matches text using n-gram overlap.

| Parameter | Type | Description |
|---|---|---|
| `target` | `str` | The string to search for. |
| `text` | `str` | The text to search in. |

**Returns:**

- `bool` — True if n-gram overlap score exceeds threshold, False otherwise.

## `class AttackStats`

Statistics for attack analysis results.

## `class ConversationAnalytics`

Handles analytics operations on conversation data, such as finding similar chat messages
based on conversation history or embedding similarity.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `memory_interface` | `MemoryInterface` | An instance of MemoryInterface for accessing conversation data. |

**Methods:**

#### get_prompt_entries_with_same_converted_content

```python
get_prompt_entries_with_same_converted_content(chat_message_content: str) → list[ConversationMessageWithSimilarity]
```

Retrieve chat messages that have the same converted content.

| Parameter | Type | Description |
|---|---|---|
| `chat_message_content` | `str` | The content of the chat message to find similar messages for. |

**Returns:**

- `list[ConversationMessageWithSimilarity]` — list[ConversationMessageWithSimilarity]: A list of ConversationMessageWithSimilarity objects representing
- `list[ConversationMessageWithSimilarity]` — the similar chat messages based on content.

#### get_similar_chat_messages_by_embedding

```python
get_similar_chat_messages_by_embedding(chat_message_embedding: list[float], threshold: float = 0.8) → list[EmbeddingMessageWithSimilarity]
```

Retrieve chat messages that are similar to the given embedding based on cosine similarity.

| Parameter | Type | Description |
|---|---|---|
| `chat_message_embedding` | `List[float]` | The embedding of the chat message to find similar messages for. |
| `threshold` | `float` | The similarity threshold for considering messages as similar. Defaults to 0.8. Defaults to `0.8`. |

**Returns:**

- `list[EmbeddingMessageWithSimilarity]` — List[ConversationMessageWithSimilarity]: A list of ConversationMessageWithSimilarity objects representing
- `list[EmbeddingMessageWithSimilarity]` — the similar chat messages based on embedding similarity.

## `class ExactTextMatching(TextMatching)`

Exact substring matching strategy.

Checks if the target string is present in the text as a substring.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `case_sensitive` | `bool` | Whether to perform case-sensitive matching. Defaults to False. Defaults to `False`. |
| `ignore_whitespace` | `bool` | Whether to ignore whitespace. Defaults to True. Defaults to `True`. |

**Methods:**

#### `is_match(target: str, text: str) → bool`

Check if target string is present in text.

| Parameter | Type | Description |
|---|---|---|
| `target` | `str` | The substring to search for. |
| `text` | `str` | The text to search in. |

**Returns:**

- `bool` — True if target is found in text, False otherwise.

## `class TextMatching(Protocol)`

Protocol for text matching strategies.

Classes implementing this protocol must provide an is_match method that
checks if a target string matches text according to some strategy.

**Methods:**

#### `is_match(target: str, text: str) → bool`

Check if target matches text according to the strategy.

| Parameter | Type | Description |
|---|---|---|
| `target` | `str` | The string to search for. |
| `text` | `str` | The text to search in. |

**Returns:**

- `bool` — True if target matches text according to the strategy, False otherwise.
