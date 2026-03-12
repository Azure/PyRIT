# pyrit.executor.benchmark

Benchmark modules.

## `class FairnessBiasBenchmark(Strategy[FairnessBiasBenchmarkContext, AttackResult])`

Strategy for executing fairness bias benchmarks with required scorer.

This strategy evaluates target models for fairness and bias by:
1. Generating stories about different subjects (professions, roles, characters, etc.)
2. Extracting names from the generated content
3. Scoring responses for bias using a required scorer instance
4. Tracking results for bias analysis and reporting

The benchmark requires:
- A scorer instance provided in the attack config
- Subject and story type as strings
- Optionally, a custom objective (if not provided, a default will be generated)

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target model to test for fairness and bias |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Optional configuration for attack converters Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Optional configuration for attack scoring Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Optional normalizer for prompt processing Defaults to `None`. |
| `max_attempts_on_failure` | `int` | Maximum number of retry attempts on failure (default: 0) Defaults to `0`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → AttackResult`

Execute the benchmark strategy asynchronously with the provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `**kwargs` | `Any` | Keyword arguments containing: subject (str): The subject to test (profession, role, character, etc.) story_type (str): The type of story to generate num_experiments (int, optional): Number of experiments to run (default: 1) objective (str, optional): Custom objective prompt (default: auto-generated) prepended_conversation (List[Message], optional): Context conversation memory_labels (Dict[str, str], optional): Labels for memory tracking Defaults to `{}`. |

**Returns:**

- `AttackResult` — The result of the benchmark execution

#### `get_experiment_summary(context: FairnessBiasBenchmarkContext) → dict[str, Any]`

Get a summary of the experiment results.

| Parameter | Type | Description |
|---|---|---|
| `context` | `FairnessBiasBenchmarkContext` | The benchmark context containing experiment results |

**Returns:**

- `dict[str, Any]` — Dict[str, Any]: Summary dictionary of experiment

#### `get_last_context() → Optional[FairnessBiasBenchmarkContext]`

Get the context from the last execution.

**Returns:**

- `Optional[FairnessBiasBenchmarkContext]` — Optional[FairnessBiasBenchmarkContext]: The context from the most recent execution,
or None if no execution has occurred

## `class FairnessBiasBenchmarkContext(StrategyContext)`

Context for fairness bias benchmark execution.

## `class QuestionAnsweringBenchmark(Strategy[QuestionAnsweringBenchmarkContext, AttackResult])`

Strategy for executing question answering benchmarks.

This strategy evaluates target models on multiple choice questions by:
1. Formatting questions with their choices into prompts
2. Sending prompts to the target model via PromptSendingAttack
3. Evaluating responses using configured scorers
4. Tracking success/failure for benchmark reporting

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to evaluate. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |
| `objective_format_string` | `str` | Format string for objectives sent to scorers. Defaults to `_DEFAULT_OBJECTIVE_FORMAT`. |
| `question_asking_format_string` | `str` | Format string for questions sent to target. Defaults to `_DEFAULT_QUESTION_FORMAT`. |
| `options_format_string` | `str` | Format string for formatting answer choices. Defaults to `_DEFAULT_OPTIONS_FORMAT`. |
| `max_attempts_on_failure` | `int` | Maximum number of attempts on failure. Defaults to `0`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → AttackResult`

Execute the QA benchmark strategy asynchronously with the provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `question_answering_entry` | `QuestionAnsweringEntry` | The question answering entry to evaluate. |
| `prepended_conversation` | `Optional[List[Message]]` | Conversation to prepend. |
| `memory_labels` | `Optional[Dict[str, str]]` | Memory labels for the benchmark context. |
| `**kwargs` | `Any` | Additional parameters for the benchmark. Defaults to `{}`. |

**Returns:**

- `AttackResult` — The result of the benchmark execution.

## `class QuestionAnsweringBenchmarkContext(StrategyContext)`

Context for question answering benchmark execution.
