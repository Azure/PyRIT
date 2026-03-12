# pyrit.executor.attack.printer

Attack result printers module.

## `class AttackResultPrinter(ABC)`

Abstract base class for printing attack results.

This interface defines the contract for printing attack results in various formats.
Implementations can render results to console, logs, files, or other outputs.

**Methods:**

#### print_conversation_async

```python
print_conversation_async(result: AttackResult, include_scores: bool = False) → None
```

Print only the conversation history.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result containing the conversation to print |
| `include_scores` | `bool` | Whether to include scores in the output. Defaults to False. Defaults to `False`. |

#### print_result_async

```python
print_result_async(result: AttackResult, include_auxiliary_scores: bool = False, include_pruned_conversations: bool = False, include_adversarial_conversation: bool = False) → None
```

Print the complete attack result.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to print |
| `include_auxiliary_scores` | `bool` | Whether to include auxiliary scores in the output. Defaults to False. Defaults to `False`. |
| `include_pruned_conversations` | `bool` | Whether to include pruned conversations. For each pruned conversation, only the last message and its score are shown. Defaults to False. Defaults to `False`. |
| `include_adversarial_conversation` | `bool` | Whether to include the adversarial conversation (the red teaming LLM's reasoning). Only shown for successful attacks to avoid overwhelming output. Defaults to False. Defaults to `False`. |

#### `print_summary_async(result: AttackResult) → None`

Print a summary of the attack result without the full conversation.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to summarize |

## `class ConsoleAttackResultPrinter(AttackResultPrinter)`

Console printer for attack results with enhanced formatting.

This printer formats attack results for console display with optional color coding,
proper indentation, text wrapping, and visual separators. Colors can be disabled
for consoles that don't support ANSI characters.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `width` | `int` | Maximum width for text wrapping. Must be positive. Defaults to 100. Defaults to `100`. |
| `indent_size` | `int` | Number of spaces for indentation. Must be non-negative. Defaults to 2. Defaults to `2`. |
| `enable_colors` | `bool` | Whether to enable ANSI color output. When False, all output will be plain text without colors. Defaults to True. Defaults to `True`. |

**Methods:**

#### print_conversation_async

```python
print_conversation_async(result: AttackResult, include_scores: bool = False, include_reasoning_trace: bool = False) → None
```

Print the conversation history to console with enhanced formatting.

Displays the full conversation between user and assistant, including:
- Turn numbers
- Role indicators (USER/ASSISTANT)
- Original and converted values when different
- Images if present
- Scores for each response

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result containing the conversation_id. Must have a valid conversation_id attribute. |
| `include_scores` | `bool` | Whether to include scores in the output. Defaults to False. Defaults to `False`. |
| `include_reasoning_trace` | `bool` | Whether to include model reasoning trace in the output for applicable models. Defaults to False. Defaults to `False`. |

#### print_messages_async

```python
print_messages_async(messages: list[Any], include_scores: bool = False, include_reasoning_trace: bool = False) → None
```

Print a list of messages to console with enhanced formatting.

This method can be called directly with a list of Message objects,
without needing an AttackResult. Useful for printing prepended_conversation
or any other list of messages.

Displays:
- Turn numbers
- Role indicators (USER/ASSISTANT/SYSTEM)
- Original and converted values when different
- Images if present
- Scores for each response (if include_scores=True)

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list` | List of Message objects to print. |
| `include_scores` | `bool` | Whether to include scores in the output. Defaults to False. Defaults to `False`. |
| `include_reasoning_trace` | `bool` | Whether to include model reasoning trace in the output for applicable models. Defaults to False. Defaults to `False`. |

#### print_result_async

```python
print_result_async(result: AttackResult, include_auxiliary_scores: bool = False, include_pruned_conversations: bool = False, include_adversarial_conversation: bool = False) → None
```

Print the complete attack result to console.

This method orchestrates the printing of all components of an attack result,
including header, summary, conversation history, metadata, and footer.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to print. Must not be None. |
| `include_auxiliary_scores` | `bool` | Whether to include auxiliary scores in the output. Defaults to False. Defaults to `False`. |
| `include_pruned_conversations` | `bool` | Whether to include pruned conversations. For each pruned conversation, only the last message and its score are shown. Defaults to False. Defaults to `False`. |
| `include_adversarial_conversation` | `bool` | Whether to include the adversarial conversation (the red teaming LLM's reasoning). Only shown for successful attacks to avoid overwhelming output. Defaults to False. Defaults to `False`. |

#### `print_summary_async(result: AttackResult) → None`

Print a summary of the attack result with enhanced formatting.

Displays:
- Basic information (objective, attack type, conversation ID)
- Execution metrics (turns executed, execution time)
- Outcome information (status, reason)
- Final score if available

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to summarize. Must contain objective, attack_identifier, conversation_id, executed_turns, execution_time_ms, outcome, and optionally outcome_reason and last_score attributes. |

## `class MarkdownAttackResultPrinter(AttackResultPrinter)`

Markdown printer for attack results optimized for Jupyter notebooks.

This printer formats attack results as markdown, making them ideal for display
in Jupyter notebooks where LLM responses often contain code blocks and other
markdown formatting that should be properly rendered.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `display_inline` | `bool` | If True, uses IPython.display to render markdown inline in Jupyter notebooks. If False, prints markdown strings. Defaults to True. Defaults to `True`. |

**Methods:**

#### print_conversation_async

```python
print_conversation_async(result: AttackResult, include_scores: bool = False) → None
```

Print only the conversation history as formatted markdown.

Extracts and displays the conversation messages from the attack result
without the summary or metadata sections. Useful for focusing on the
actual interaction flow.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result containing the conversation to display. |
| `include_scores` | `bool` | Whether to include scores for each message. Defaults to False. Defaults to `False`. |

#### print_result_async

```python
print_result_async(result: AttackResult, include_auxiliary_scores: bool = False, include_pruned_conversations: bool = False, include_adversarial_conversation: bool = False) → None
```

Print the complete attack result as formatted markdown.

Generates a comprehensive markdown report including attack summary,
conversation history, scores, and metadata. The output is optimized
for display in Jupyter notebooks.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to print. |
| `include_auxiliary_scores` | `bool` | Whether to include auxiliary scores in the conversation display. Defaults to False. Defaults to `False`. |
| `include_pruned_conversations` | `bool` | Whether to include pruned conversations. For each pruned conversation, only the last message and its score are shown. Defaults to False. Defaults to `False`. |
| `include_adversarial_conversation` | `bool` | Whether to include the adversarial conversation (the red teaming LLM's reasoning). Only shown for successful attacks to avoid overwhelming output. Defaults to False. Defaults to `False`. |

#### `print_summary_async(result: AttackResult) → None`

Print a summary of the attack result as formatted markdown.

Displays key information about the attack including objective, outcome,
execution metrics, and final score without the full conversation history.
Useful for getting a quick overview of the attack results.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to summarize. |
