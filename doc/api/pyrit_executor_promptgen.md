# pyrit.executor.promptgen

Prompt generator strategy imports.

## `class AnecdoctorContext(PromptGeneratorStrategyContext)`

Context specific to Anecdoctor prompt generation.

Contains all parameters needed for executing Anecdoctor prompt generation,
including the evaluation data, language settings, and conversation ID.

## `class AnecdoctorGenerator(PromptGeneratorStrategy[AnecdoctorContext, AnecdoctorResult], Identifiable)`

Implementation of the Anecdoctor prompt generation strategy.

The Anecdoctor generator creates misinformation content by either:
1. Using few-shot examples directly (default mode when processing_model is not provided)
2. First extracting a knowledge graph from examples, then using it (when processing_model is provided)

This generator is designed to test a model's susceptibility to generating false or
misleading content when provided with examples in ClaimsReview format. The generator
can optionally use a processing model to extract a knowledge graph representation
of the examples before generating the final content.

The generation flow consists of:
1. (Optional) Extract knowledge graph from evaluation data using processing model
2. Format a system prompt based on language and content type
3. Send formatted examples (or knowledge graph) to target model
4. Return the generated content as the result

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptChatTarget` | The chat model to be used for prompt generation. |
| `processing_model` | `Optional[PromptChatTarget]` | The model used for knowledge graph extraction. If provided, the generator will extract a knowledge graph from the examples before generation. If None, the generator will use few-shot examples directly. Defaults to `None`. |
| `converter_config` | `Optional[StrategyConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → AnecdoctorResult`

Execute the prompt generation strategy asynchronously with the provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `content_type` | `str` | The type of content to generate (e.g., "viral tweet", "news article"). |
| `language` | `str` | The language of the content to generate (e.g., "english", "spanish"). |
| `evaluation_data` | `List[str]` | The data in ClaimsReview format to use in constructing the prompt. |
| `memory_labels` | `Optional[Dict[str, str]]` | Memory labels for the generation context. |
| `**kwargs` | `Any` | Additional parameters for the generation. Defaults to `{}`. |

**Returns:**

- `AnecdoctorResult` — The result of the anecdoctor generation.

## `class AnecdoctorResult(PromptGeneratorStrategyResult)`

Result of Anecdoctor prompt generation.

Contains the generated content from the misinformation prompt generation.

## `class PromptGeneratorStrategy(Strategy[PromptGeneratorStrategyContextT, PromptGeneratorStrategyResultT], ABC)`

Base class for all prompt generator strategies.
Provides a structure for implementing specific prompt generation strategies.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `context_type` | `type` | Type of the context used by the strategy. |
| `logger` | `logging.Logger` | Logger instance for logging events. Defaults to `logger`. |
| `event_handler` | `StrategyEventHandler` | Event handler for handling strategy events. Defaults to `None`. |

## `class PromptGeneratorStrategyContext(StrategyContext, ABC)`

Base class for all prompt generator strategy contexts.

## `class PromptGeneratorStrategyResult(StrategyResult, ABC)`

Base class for all prompt generator strategy results.
