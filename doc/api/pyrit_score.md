# pyrit.score

Scoring functionality for evaluating AI model responses across various dimensions
including harm detection, objective completion, and content classification.

## Functions

### create_conversation_scorer

```python
create_conversation_scorer(scorer: Scorer, validator: Optional[ScorerPromptValidator] = None) → Scorer
```

Create a ConversationScorer that inherits from the same type as the wrapped scorer.

This factory dynamically creates a ConversationScorer class that inherits from the wrapped scorer's
base class (FloatScaleScorer or TrueFalseScorer), ensuring the returned scorer is an instance
of both ConversationScorer and the wrapped scorer's type.

| Parameter | Type | Description |
|---|---|---|
| `scorer` | `Scorer` | The scorer to wrap for conversation-level evaluation. Must be an instance of FloatScaleScorer or TrueFalseScorer. |
| `validator` | `Optional[ScorerPromptValidator]` | Optional validator override. If not provided, uses the wrapped scorer's validator. Defaults to `None`. |

**Returns:**

- `Scorer` — A ConversationScorer instance that is also an instance of the wrapped scorer's type.

**Raises:**

- `ValueError` — If the scorer is not an instance of FloatScaleScorer or TrueFalseScorer.

## `class BatchScorer`

A utility class for scoring prompts in batches in a parallelizable and convenient way.

This class provides functionality to score existing prompts stored in memory
without any target interaction, making it a pure scoring utility.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `batch_size` | `int` | The (max) batch size for sending prompts. Defaults to 10. Note: If using a scorer that takes a prompt target, and providing max requests per minute on the target, this should be set to 1 to ensure proper rate limit management. Defaults to `10`. |

**Methods:**

#### score_responses_by_filters_async

```python
score_responses_by_filters_async(scorer: Scorer, attack_id: Optional[str | uuid.UUID] = None, conversation_id: Optional[str | uuid.UUID] = None, prompt_ids: Optional[list[str] | list[uuid.UUID]] = None, labels: Optional[dict[str, str]] = None, sent_after: Optional[datetime] = None, sent_before: Optional[datetime] = None, original_values: Optional[list[str]] = None, converted_values: Optional[list[str]] = None, data_type: Optional[str] = None, not_data_type: Optional[str] = None, converted_value_sha256: Optional[list[str]] = None, objective: str = '') → list[Score]
```

Score the responses that match the specified filters.

| Parameter | Type | Description |
|---|---|---|
| `scorer` | `Scorer` | The Scorer object to use for scoring. |
| `attack_id` | `Optional[str | uuid.UUID]` | The ID of the attack. Defaults to None. Defaults to `None`. |
| `conversation_id` | `Optional[str | uuid.UUID]` | The ID of the conversation. Defaults to None. Defaults to `None`. |
| `prompt_ids` | `Optional[list[str] | list[uuid.UUID]]` | A list of prompt IDs. Defaults to None. Defaults to `None`. |
| `labels` | `Optional[dict[str, str]]` | A dictionary of labels. Defaults to None. Defaults to `None`. |
| `sent_after` | `Optional[datetime]` | Filter for prompts sent after this datetime. Defaults to None. Defaults to `None`. |
| `sent_before` | `Optional[datetime]` | Filter for prompts sent before this datetime. Defaults to None. Defaults to `None`. |
| `original_values` | `Optional[list[str]]` | A list of original values. Defaults to None. Defaults to `None`. |
| `converted_values` | `Optional[list[str]]` | A list of converted values. Defaults to None. Defaults to `None`. |
| `data_type` | `Optional[str]` | The data type to filter by. Defaults to None. Defaults to `None`. |
| `not_data_type` | `Optional[str]` | The data type to exclude. Defaults to None. Defaults to `None`. |
| `converted_value_sha256` | `Optional[list[str]]` | A list of SHA256 hashes of converted values. Defaults to None. Defaults to `None`. |
| `objective` | `str` | A task is used to give the scorer more context on what exactly to score. A task might be the request prompt text or the original attack model's objective. **Note: the same task is applied to all matched prompts.** Defaults to an empty string. Defaults to `''`. |

**Returns:**

- `list[Score]` — list[Score]: A list of Score objects for responses that match the specified filters.

**Raises:**

- `ValueError` — If no entries match the provided filters.

## `class ConsoleScorerPrinter(ScorerPrinter)`

Console printer for scorer information with enhanced formatting.

This printer formats scorer details for console display with optional color coding,
proper indentation, and visual hierarchy. Colors can be disabled for consoles
that don't support ANSI characters.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `indent_size` | `int` | Number of spaces for indentation. Must be non-negative. Defaults to 2. Defaults to `2`. |
| `enable_colors` | `bool` | Whether to enable ANSI color output. When False, all output will be plain text without colors. Defaults to True. Defaults to `True`. |

**Methods:**

#### print_harm_scorer

```python
print_harm_scorer(scorer_identifier: ComponentIdentifier, harm_category: str) → None
```

Print harm scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Harm evaluation metrics (MAE, Krippendorff alpha) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |
| `harm_category` | `str` | The harm category for looking up metrics (e.g., "hate_speech", "violence"). |

#### `print_objective_scorer(scorer_identifier: ComponentIdentifier) → None`

Print objective scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Objective evaluation metrics (accuracy, precision, recall, F1) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |

## `class ConversationScorer(Scorer, ABC)`

Scorer that evaluates entire conversation history rather than individual messages.

This scorer wraps another scorer (FloatScaleScorer or TrueFalseScorer) and evaluates
the full conversation context. Useful for multi-turn conversations where context matters
(e.g., psychosocial harms that emerge over time or persuasion/deception over many messages).

The ConversationScorer dynamically inherits from the same base class as the wrapped scorer,
ensuring proper type compatibility.

Note: This class cannot be instantiated directly. Use create_conversation_scorer() factory instead.

**Methods:**

#### `validate_return_scores(scores: list[Score]) → None`

Validate scores by delegating to the wrapped scorer's validation.

| Parameter | Type | Description |
|---|---|---|
| `scores` | `list[Score]` | The scores to validate. |

## `class Scorer(Identifiable, abc.ABC)`

Abstract base class for scorers.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `validator` | `ScorerPromptValidator` | Validator for message pieces and scorer configuration. |

**Methods:**

#### evaluate_async

```python
evaluate_async(file_mapping: Optional[ScorerEvalDatasetFiles] = None, num_scorer_trials: int = 3, update_registry_behavior: RegistryUpdateBehavior = None, max_concurrency: int = 10) → Optional[ScorerMetrics]
```

Evaluate this scorer against human-labeled datasets.

Uses file mapping to determine which datasets to evaluate and how to aggregate results.

| Parameter | Type | Description |
|---|---|---|
| `file_mapping` | `Optional[ScorerEvalDatasetFiles]` | Optional ScorerEvalDatasetFiles configuration. If not provided, uses the scorer's configured evaluation_file_mapping. Maps input file patterns to an output result file. Defaults to `None`. |
| `num_scorer_trials` | `int` | Number of times to score each response (for measuring variance). Defaults to 3. Defaults to `3`. |
| `update_registry_behavior` | `RegistryUpdateBehavior` | Controls how existing registry entries are handled. - SKIP_IF_EXISTS (default): Check registry for existing results. If found, return cached metrics. - ALWAYS_UPDATE: Always run evaluation and overwrite any existing registry entry. - NEVER_UPDATE: Always run evaluation but never write to registry (for debugging). Defaults to RegistryUpdateBehavior.SKIP_IF_EXISTS. Defaults to `None`. |
| `max_concurrency` | `int` | Maximum number of concurrent scoring requests. Defaults to 10. Defaults to `10`. |

**Returns:**

- `Optional[ScorerMetrics]` — The evaluation metrics, or None if no datasets found.

**Raises:**

- `ValueError` — If no file_mapping is provided and no evaluation_file_mapping is configured.

#### `get_eval_hash() → str`

Compute a behavioral equivalence hash for evaluation grouping.

Delegates to ``ScorerEvaluationIdentifier`` which filters target children
(prompt_target, converter_target) to behavioral params only, so the same
scorer configuration on different deployments produces the same eval hash.

**Returns:**

- `str` — A hex-encoded SHA256 hash suitable for eval registry keying.

#### `get_scorer_metrics() → Optional[ScorerMetrics]`

Get evaluation metrics for this scorer from the configured evaluation result file.

Looks up metrics by this scorer's identity hash in the JSONL result file.
The result file may contain entries for multiple scorer configurations.

Subclasses must implement this to return the appropriate metrics type:
- TrueFalseScorer subclasses should return ObjectiveScorerMetrics
- FloatScaleScorer subclasses should return HarmScorerMetrics

**Returns:**

- `Optional[ScorerMetrics]` — The metrics for this scorer, or None if not found or not configured.

#### `scale_value_float(value: float, min_value: float, max_value: float) → float`

Scales a value from 0 to 1 based on the given min and max values. E.g. 3 stars out of 5 stars would be .5.

| Parameter | Type | Description |
|---|---|---|
| `value` | `float` | The value to be scaled. |
| `min_value` | `float` | The minimum value of the range. |
| `max_value` | `float` | The maximum value of the range. |

**Returns:**

- `float` — The scaled value.

#### score_async

```python
score_async(message: Message, objective: Optional[str] = None, role_filter: Optional[ChatMessageRole] = None, skip_on_error_result: bool = False, infer_objective_from_request: bool = False) → list[Score]
```

Score the message, add the results to the database, and return a list of Score objects.

| Parameter | Type | Description |
|---|---|---|
| `message` | `Message` | The message to be scored. |
| `objective` | `Optional[str]` | The task or objective based on which the message should be scored. Defaults to None. Defaults to `None`. |
| `role_filter` | `Optional[ChatMessageRole]` | Only score messages with this exact stored role. Use "assistant" to score only real assistant responses, or "simulated_assistant" to score only simulated responses. Defaults to None (no filtering). Defaults to `None`. |
| `skip_on_error_result` | `bool` | If True, skip scoring if the message contains an error. Defaults to False. Defaults to `False`. |
| `infer_objective_from_request` | `bool` | If True, infer the objective from the message's previous request when objective is not provided. Defaults to False. Defaults to `False`. |

**Returns:**

- `list[Score]` — list[Score]: A list of Score objects representing the results.

**Raises:**

- `PyritException` — If scoring raises a PyRIT exception (re-raised with enhanced context).
- `RuntimeError` — If scoring raises a non-PyRIT exception (wrapped with scorer context).

#### score_image_async

```python
score_image_async(image_path: str, objective: Optional[str] = None) → list[Score]
```

Score the given image using the chat target.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The path to the image file to be scored. |
| `objective` | `Optional[str]` | The objective based on which the image should be scored. Defaults to None. Defaults to `None`. |

**Returns:**

- `list[Score]` — list[Score]: A list of Score objects representing the results.

#### score_image_batch_async

```python
score_image_batch_async(image_paths: Sequence[str], objectives: Optional[Sequence[str]] = None, batch_size: int = 10) → list[Score]
```

Score a batch of images asynchronously.

| Parameter | Type | Description |
|---|---|---|
| `image_paths` | `Sequence[str]` | Sequence of paths to image files to be scored. |
| `objectives` | `Optional[Sequence[str]]` | Optional sequence of objectives corresponding to each image. If provided, must match the length of image_paths. Defaults to None. Defaults to `None`. |
| `batch_size` | `int` | Maximum number of images to score concurrently. Defaults to 10. Defaults to `10`. |

**Returns:**

- `list[Score]` — list[Score]: A list of Score objects representing the scoring results for all images.

**Raises:**

- `ValueError` — If the number of objectives does not match the number of image_paths.

#### score_prompts_batch_async

```python
score_prompts_batch_async(messages: Sequence[Message], objectives: Optional[Sequence[str]] = None, batch_size: int = 10, role_filter: Optional[ChatMessageRole] = None, skip_on_error_result: bool = False, infer_objective_from_request: bool = False) → list[Score]
```

Score multiple prompts in batches using the provided objectives.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `Sequence[Message]` | The messages to be scored. |
| `objectives` | `Sequence[str]` | The objectives/tasks based on which the prompts should be scored. Must have the same length as messages. Defaults to `None`. |
| `batch_size` | `int` | The maximum batch size for processing prompts. Defaults to 10. Defaults to `10`. |
| `role_filter` | `Optional[ChatMessageRole]` | If provided, only score pieces with this role. Defaults to None (no filtering). Defaults to `None`. |
| `skip_on_error_result` | `bool` | If True, skip scoring pieces that have errors. Defaults to False. Defaults to `False`. |
| `infer_objective_from_request` | `bool` | If True and objective is empty, attempt to infer the objective from the request. Defaults to False. Defaults to `False`. |

**Returns:**

- `list[Score]` — list[Score]: A flattened list of Score objects from all scored prompts.

**Raises:**

- `ValueError` — If objectives is empty or if the number of objectives doesn't match
the number of messages.

#### score_response_async

```python
score_response_async(response: Message, objective_scorer: Optional[Scorer] = None, auxiliary_scorers: Optional[list[Scorer]] = None, role_filter: ChatMessageRole = 'assistant', objective: Optional[str] = None, skip_on_error_result: bool = True) → dict[str, list[Score]]
```

Score a response using an objective scorer and optional auxiliary scorers.

| Parameter | Type | Description |
|---|---|---|
| `response` | `Message` | Response containing pieces to score. |
| `objective_scorer` | `Optional[Scorer]` | The main scorer to determine success. Defaults to None. Defaults to `None`. |
| `auxiliary_scorers` | `Optional[List[Scorer]]` | List of auxiliary scorers to apply. Defaults to None. Defaults to `None`. |
| `role_filter` | `ChatMessageRole` | Only score pieces with this exact stored role. Defaults to "assistant" (real responses only, not simulated). Defaults to `'assistant'`. |
| `objective` | `Optional[str]` | Task/objective for scoring context. Defaults to None. Defaults to `None`. |
| `skip_on_error_result` | `bool` | If True, skip scoring pieces that have errors. Defaults to True. Defaults to `True`. |

**Returns:**

- `dict[str, list[Score]]` — Dict[str, List[Score]]: Dictionary with keys `auxiliary_scores` and `objective_scores`
containing lists of scores from each type of scorer.

**Raises:**

- `ValueError` — If response is not provided.

#### score_response_multiple_scorers_async

```python
score_response_multiple_scorers_async(response: Message, scorers: list[Scorer], role_filter: ChatMessageRole = 'assistant', objective: Optional[str] = None, skip_on_error_result: bool = True) → list[Score]
```

Score a response using multiple scorers in parallel.

This method applies each scorer to the first scorable response piece (filtered by role and error),
and returns all scores. This is typically used for auxiliary scoring where all results are needed.

| Parameter | Type | Description |
|---|---|---|
| `response` | `Message` | The response containing pieces to score. |
| `scorers` | `List[Scorer]` | List of scorers to apply. |
| `role_filter` | `ChatMessageRole` | Only score pieces with this exact stored role. Defaults to "assistant" (real responses only, not simulated). Defaults to `'assistant'`. |
| `objective` | `Optional[str]` | Optional objective description for scoring context. Defaults to `None`. |
| `skip_on_error_result` | `bool` | If True, skip scoring pieces that have errors (default: True). Defaults to `True`. |

**Returns:**

- `list[Score]` — List[Score]: All scores from all scorers

#### `score_text_async(text: str, objective: Optional[str] = None) → list[Score]`

Scores the given text based on the task using the chat target.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The text to be scored. |
| `objective` | `Optional[str]` | The task based on which the text should be scored Defaults to `None`. |

**Returns:**

- `list[Score]` — list[Score]: A list of Score objects representing the results.

#### `validate_return_scores(scores: list[Score]) → None`

Validate the scores returned by the scorer. Because some scorers may require
specific Score types or values.

| Parameter | Type | Description |
|---|---|---|
| `scores` | `list[Score]` | The scores to be validated. |

## `class ScorerPrinter(ABC)`

Abstract base class for printing scorer information.

This interface defines the contract for printing scorer details including
type information, nested sub-scorers, and evaluation metrics from the registry.
Implementations can render output to console, logs, files, or other outputs.

**Methods:**

#### print_harm_scorer

```python
print_harm_scorer(scorer_identifier: ComponentIdentifier, harm_category: str) → None
```

Print harm scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Harm evaluation metrics (MAE, Krippendorff alpha) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |
| `harm_category` | `str` | The harm category for looking up metrics (e.g., "hate_speech", "violence"). |

#### `print_objective_scorer(scorer_identifier: ComponentIdentifier) → None`

Print objective scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Objective evaluation metrics (accuracy, precision, recall, F1) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |

## `class ScorerPromptValidator`

Validates message pieces and scorer configurations.

This class provides validation for scorer inputs, ensuring that message pieces meet
required criteria such as data types, roles, and metadata requirements.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `supported_data_types` | `Optional[Sequence[PromptDataType]]` | Data types that the scorer supports. Defaults to all data types if not provided. Defaults to `None`. |
| `required_metadata` | `Optional[Sequence[str]]` | Metadata keys that must be present in message pieces. Defaults to empty list. Defaults to `None`. |
| `supported_roles` | `Optional[Sequence[ChatMessageRole]]` | Message roles that the scorer supports. Defaults to all roles if not provided. Defaults to `None`. |
| `max_pieces_in_response` | `Optional[int]` | Maximum number of pieces allowed in a response. Defaults to None (no limit). Defaults to `None`. |
| `max_text_length` | `Optional[int]` | Maximum character length for text data type pieces. Defaults to None (no limit). Defaults to `None`. |
| `enforce_all_pieces_valid` | `Optional[bool]` | Whether all pieces must be valid or just at least one. Defaults to False. Defaults to `False`. |
| `raise_on_no_valid_pieces` | `Optional[bool]` | Whether to raise ValueError when no pieces are valid. Defaults to False, allowing scorers to handle empty results gracefully (e.g., returning False for blocked responses). Set to True to raise an exception instead. Defaults to `False`. |
| `is_objective_required` | `bool` | Whether an objective must be provided for scoring. Defaults to False. Defaults to `False`. |

**Methods:**

#### `is_message_piece_supported(message_piece: MessagePiece) → bool`

Check if a message piece is supported by this validator.

| Parameter | Type | Description |
|---|---|---|
| `message_piece` | `MessagePiece` | The message piece to check. |

**Returns:**

- `bool` — True if the message piece meets all validation criteria, False otherwise.

#### `validate(message: Message, objective: str | None) → None`

Validate a message and objective against configured requirements.

| Parameter | Type | Description |
|---|---|---|
| `message` | `Message` | The message to validate. |
| `objective` | `str | None` | The objective string, if required. |

**Raises:**

- `ValueError` — If validation fails due to unsupported pieces, exceeding max pieces, or missing objective.
