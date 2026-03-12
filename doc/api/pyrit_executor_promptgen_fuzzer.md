# pyrit.executor.promptgen.fuzzer

Fuzzer module for generating adversarial prompts through mutation and crossover operations.

## `class FuzzerContext(PromptGeneratorStrategyContext)`

Context for the Fuzzer prompt generation strategy.

This context contains all execution-specific state for a Fuzzer prompt generation instance,
ensuring thread safety by isolating state per execution.

## `class FuzzerConverter(PromptConverter)`

Base class for GPTFUZZER converters.

Adapted from GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.
Paper: https://arxiv.org/pdf/2309.10253 by Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing.
GitHub: https://github.com/sherdencooper/GPTFuzz/tree/master

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | Chat target used to perform fuzzing on user prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `prompt_template` | `SeedPrompt` | Template to be used instead of the default system prompt with instructions for the chat target. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

#### `input_supported(input_type: PromptDataType) → bool`

Check if the input type is supported.

**Returns:**

- `bool` — True if input type is text, False otherwise.

#### `output_supported(output_type: PromptDataType) → bool`

Check if the output type is supported.

**Returns:**

- `bool` — True if output type is text, False otherwise.

#### `send_prompt_async(request: Message) → str`

Send the message to the converter target and process the response.

| Parameter | Type | Description |
|---|---|---|
| `request` | `Message` | The message request to send. |

**Returns:**

- `str` — The output from the parsed JSON response.

**Raises:**

- `InvalidJsonException` — If the response is not valid JSON or missing required keys.

#### `update(kwargs: Any = {}) → None`

Update the converter with new parameters.

## `class FuzzerCrossOverConverter(FuzzerConverter)`

Uses multiple prompt templates to generate new prompts.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | Chat target used to perform fuzzing on user prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `None`. |
| `prompt_template` | `(SeedPrompt, Optional)` | Template to be used instead of the default system prompt with instructions for the chat target. Defaults to `None`. |
| `prompt_templates` | `(List[str], Optional)` | List of prompt templates to use in addition to the default one. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by combining it with a random prompt template from the list of available templates.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

#### `update(kwargs: Any = {}) → None`

Update the converter with new prompt templates.

## `class FuzzerExpandConverter(FuzzerConverter)`

Generates versions of a prompt with new, prepended sentences.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by generating versions of it with new, prepended sentences.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class FuzzerGenerator(PromptGeneratorStrategy[FuzzerContext, FuzzerResult], Identifiable)`

Implementation of the Fuzzer prompt generation strategy using Monte Carlo Tree Search (MCTS).

The Fuzzer generates diverse jailbreak prompts by systematically exploring and generating
prompt templates. It uses MCTS to balance exploration of new templates with
exploitation of promising ones, efficiently searching for effective prompt variations.

The generation flow consists of:
1. Selecting a template using MCTS-explore algorithm
2. Applying template converters to generate variations
3. Generating prompts from the selected/converted template
4. Testing prompts with the target and scoring responses
5. Updating rewards in the MCTS tree based on scores
6. Continuing until target jailbreak count reached or query limit reached

Note: While this is a prompt generator, it still requires scoring functionality
to provide feedback to the MCTS algorithm for effective template selection.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target to send the prompts to. |
| `template_converters` | `List[FuzzerConverter]` | The converters to apply on the selected jailbreak template. In each iteration, one converter is chosen at random. |
| `converter_config` | `Optional[StrategyConverterConfig]` | Configuration for prompt converters. Defaults to None. Defaults to `None`. |
| `scorer` | `Optional[Scorer]` | Configuration for scoring responses. Defaults to None. Defaults to `None`. |
| `scoring_success_threshold` | `float` | The score threshold to consider a jailbreak successful. Defaults to `0.8`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | The prompt normalizer to use. Defaults to None. Defaults to `None`. |
| `frequency_weight` | `float` | Constant that balances between high reward and selection frequency. Defaults to 0.5. Defaults to `_DEFAULT_FREQUENCY_WEIGHT`. |
| `reward_penalty` | `float` | Penalty that diminishes reward as path length increases. Defaults to 0.1. Defaults to `_DEFAULT_REWARD_PENALTY`. |
| `minimum_reward` | `float` | Minimal reward to prevent rewards from being too small. Defaults to 0.2. Defaults to `_DEFAULT_MINIMUM_REWARD`. |
| `non_leaf_node_probability` | `float` | Probability of selecting a non-leaf node. Defaults to 0.1. Defaults to `_DEFAULT_NON_LEAF_PROBABILITY`. |
| `batch_size` | `int` | The (max) batch size for sending prompts. Defaults to 10. Defaults to `_DEFAULT_BATCH_SIZE`. |
| `target_jailbreak_goal_count` | `int` | Target number of jailbreaks to find. Defaults to 1. Defaults to `_DEFAULT_TARGET_JAILBREAK_COUNT`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → FuzzerResult`

Execute the Fuzzer generation strategy asynchronously.

| Parameter | Type | Description |
|---|---|---|
| `prompts` | `List[str]` | The list of prompts to use for generation. |
| `prompt_templates` | `List[str]` | The list of prompt templates to use. |
| `max_query_limit` | `Optional[int]` | The maximum number of queries to execute. |
| `memory_labels` | `Optional[dict[str, str]]` | Optional labels to apply to the prompts. |
| `**kwargs` | `Any` | Additional keyword arguments. Defaults to `{}`. |

**Returns:**

- `FuzzerResult` — The result of the asynchronous execution.

#### with_default_scorer

```python
with_default_scorer(objective_target: PromptTarget, template_converters: list[FuzzerConverter], scoring_target: PromptChatTarget, converter_config: Optional[StrategyConverterConfig] = None, prompt_normalizer: Optional[PromptNormalizer] = None, frequency_weight: float = _DEFAULT_FREQUENCY_WEIGHT, reward_penalty: float = _DEFAULT_REWARD_PENALTY, minimum_reward: float = _DEFAULT_MINIMUM_REWARD, non_leaf_node_probability: float = _DEFAULT_NON_LEAF_PROBABILITY, batch_size: int = _DEFAULT_BATCH_SIZE, target_jailbreak_goal_count: int = _DEFAULT_TARGET_JAILBREAK_COUNT) → FuzzerGenerator
```

Create a FuzzerGenerator instance with default scoring configuration.

This factory method creates a FuzzerGenerator with a default scoring setup using
SelfAskScaleScorer with the Tree of Attacks scale, wrapped in a FloatScaleThresholdScorer
with a threshold of 0.8.

To use the returned generator, create a FuzzerContext with prompts and prompt_templates,
then pass it to execute_with_context_async().

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target to send the prompts to. |
| `template_converters` | `List[FuzzerConverter]` | The converters to apply on the selected jailbreak template. |
| `scoring_target` | `PromptChatTarget` | The chat target to use for scoring responses. |
| `converter_config` | `Optional[StrategyConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | The prompt normalizer to use. Defaults to `None`. |
| `frequency_weight` | `float` | Constant that balances between high reward and selection frequency. Defaults to `_DEFAULT_FREQUENCY_WEIGHT`. |
| `reward_penalty` | `float` | Penalty that diminishes reward as path length increases. Defaults to `_DEFAULT_REWARD_PENALTY`. |
| `minimum_reward` | `float` | Minimal reward to prevent rewards from being too small. Defaults to `_DEFAULT_MINIMUM_REWARD`. |
| `non_leaf_node_probability` | `float` | Probability of selecting a non-leaf node. Defaults to `_DEFAULT_NON_LEAF_PROBABILITY`. |
| `batch_size` | `int` | The (max) batch size for sending prompts. Defaults to `_DEFAULT_BATCH_SIZE`. |
| `target_jailbreak_goal_count` | `int` | Target number of jailbreaks to find. Defaults to `_DEFAULT_TARGET_JAILBREAK_COUNT`. |

**Returns:**

- `FuzzerGenerator` — A configured FuzzerGenerator instance with default scoring.

## `class FuzzerRephraseConverter(FuzzerConverter)`

Generates versions of a prompt with rephrased sentences.

## `class FuzzerResult(PromptGeneratorStrategyResult)`

Result of the Fuzzer prompt generation strategy execution.

This result includes the standard prompt generator result information with
fuzzer-specific concrete fields for tracking MCTS exploration and successful templates.

**Methods:**

#### `print_formatted(enable_colors: bool = True, width: int = 100) → None`

Print the result using FuzzerResultPrinter with custom formatting options.

| Parameter | Type | Description |
|---|---|---|
| `enable_colors` | `bool` | Whether to enable ANSI color output. Defaults to True. Defaults to `True`. |
| `width` | `int` | Maximum width for text wrapping. Defaults to 100. Defaults to `100`. |

#### `print_templates() → None`

Print only the successful templates (equivalent to original attack method).

## `class FuzzerResultPrinter`

Printer for Fuzzer generation strategy results with enhanced console formatting.

This printer displays fuzzer-specific information including successful templates,
jailbreak conversations, and execution statistics in a formatted, colorized output
similar to the original FuzzerAttack result display.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `width` | `int` | Maximum width for text wrapping. Must be positive. Defaults to 100. Defaults to `100`. |
| `indent_size` | `int` | Number of spaces for indentation. Must be non-negative. Defaults to 2. Defaults to `2`. |
| `enable_colors` | `bool` | Whether to enable ANSI color output. Defaults to True. Defaults to `True`. |

**Methods:**

#### `print_result(result: FuzzerResult) → None`

Print the complete fuzzer result to console.

| Parameter | Type | Description |
|---|---|---|
| `result` | `FuzzerResult` | The fuzzer result to print. |

#### `print_templates_only(result: FuzzerResult) → None`

Print only the successful templates (equivalent to original print_templates method).

| Parameter | Type | Description |
|---|---|---|
| `result` | `FuzzerResult` | The fuzzer result containing templates. |

## `class FuzzerShortenConverter(FuzzerConverter)`

Generates versions of a prompt with shortened sentences.

## `class FuzzerSimilarConverter(FuzzerConverter)`

Generates versions of a prompt with similar sentences.
