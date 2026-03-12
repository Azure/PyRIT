# pyrit.setup.initializers

PyRIT initializers package.

## `class AIRTInitializer(PyRITInitializer)`

AIRT (AI Red Team) configuration initializer.

This initializer provides a unified setup for all AIRT components including:
- Converter targets with Azure OpenAI configuration
- Composite harm and objective scorers
- Adversarial target configurations for attacks

Required Environment Variables:
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT: Azure OpenAI endpoint for converters and targets
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL: Azure OpenAI model name for converters and targets
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2: Azure OpenAI endpoint for scoring
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL2: Azure OpenAI model name for scoring

Optional Environment Variables:
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY: API key for converter endpoint. If not set, Entra ID auth is used.
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2: API key for scorer endpoint. If not set, Entra ID auth is used.
- AZURE_CONTENT_SAFETY_API_KEY: API key for content safety. If not set, Entra ID auth is used.

This configuration is designed for full AI Red Team operations with:
- Separate endpoints for attack execution vs scoring (security isolation)
- Advanced composite scoring with harm detection and content filtering
- Production-ready Azure OpenAI integration

**Methods:**

#### `initialize_async() → None`

Execute the complete AIRT initialization.

Sets up:
1. Converter targets with Azure OpenAI
2. Composite harm and objective scorers
3. Adversarial target configurations
4. Default values for all attack types

## `class LoadDefaultDatasets(PyRITInitializer)`

Load default datasets for all registered scenarios.

**Methods:**

#### `initialize_async() → None`

Load default datasets from all registered scenarios.

## `class PyRITInitializer(ABC)`

Abstract base class for PyRIT configuration initializers.

PyRIT initializers provide a structured way to configure default values
and global settings during PyRIT initialization. They replace the need for
initialization scripts with type-safe, validated, and discoverable classes.

All initializers must implement the `name`, `description`, and `initialize`
properties/methods. The `validate` method can be overridden if custom
validation logic is needed.

**Methods:**

#### `get_dynamic_default_values_info_async() → dict[str, Any]`

Get information about what default values and global variables this initializer sets.
This is useful for debugging what default_values are set by an initializer.

Performs a sandbox run in isolation to discover what would be configured,
then restores the original state. This works regardless of whether the
initializer has been run before or which instance is queried.

**Returns:**

- `dict[str, Any]` — Dict[str, Any]: Information about what defaults and globals are set.

#### `get_info_async() → dict[str, Any]`

Get information about this initializer class.

This is a class method so it can be called without instantiating the class:
await SimpleInitializer.get_info_async() instead of SimpleInitializer().get_info_async()

**Returns:**

- `dict[str, Any]` — Dict[str, Any]: Dictionary containing name, description, class information, and default values.

#### `initialize_async() → None`

Execute the initialization logic asynchronously.

This method should contain all the configuration logic, including
calls to set_default_value() and set_global_variable() as needed.
All initializers must implement this as an async method.

#### `initialize_with_tracking_async() → None`

Execute initialization while tracking what changes are made.

This method runs initialize_async() and captures information about what
default values and global variables were set. The tracking information
is not cached - it's captured during the actual initialization run.

#### `validate() → None`

Validate the initializer configuration before execution.

This method checks that all required environment variables are set.
Subclasses should not override this method.

**Raises:**

- `ValueError` — If required environment variables are not set.

## `class ScenarioObjectiveListInitializer(PyRITInitializer)`

Configure default seed groups for use in PyRIT scenarios.

**Methods:**

#### `initialize_async() → None`

Set default objectives for scenarios that accept them (deprecated).

## `class ScenarioObjectiveTargetInitializer(PyRITInitializer)`

Configure a simple objective target for use in PyRIT scenarios.

**Methods:**

#### `initialize_async() → None`

Set default objective target for scenarios that accept them.

## `class ScorerInitializer(PyRITInitializer)`

Scorer Initializer for registering pre-configured scorers.

This initializer registers all evaluation scorers into the ScorerRegistry.
Targets are pulled from the TargetRegistry (populated by TargetInitializer),
so this initializer must run after the target initializer (enforced via execution_order).
Scorers that fail to initialize (e.g., due to missing targets) are skipped with a warning.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tags` | `list[ScorerTag] | None` | Tags for future filtering. Defaults to ["default"]. Defaults to `None`. |

**Methods:**

#### `initialize_async() → None`

Register available scorers using targets from the TargetRegistry.

**Raises:**

- `RuntimeError` — If the TargetRegistry is empty or hasn't been initialized.

## `class SimpleInitializer(PyRITInitializer)`

Complete simple configuration initializer.

This initializer provides a unified setup for basic PyRIT usage including:
- Converter targets with basic OpenAI configuration
- Simple objective scorer (no harm detection)
- Adversarial target configurations for attacks

Required Environment Variables:
- OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_MODEL

Optional Environment Variables:
- OPENAI_CHAT_KEY: API key. If not set, Entra ID auth is used for Azure endpoints.

This configuration is designed for simple use cases with:
- Basic OpenAI API integration
- Simplified scoring without harm detection or content filtering
- Minimal configuration requirements

**Methods:**

#### `initialize_async() → None`

Execute the complete simple initialization.

Sets up:
1. Converter targets with basic OpenAI configuration
2. Simple objective scorer (no harm detection)
3. Adversarial target configurations
4. Default values for attack types

## `class TargetInitializer(PyRITInitializer)`

Supported Endpoints by Category:

**OpenAI Chat Targets (OpenAIChatTarget):**
- PLATFORM_OPENAI_CHAT_* - Platform OpenAI Chat API
- AZURE_OPENAI_GPT4O_* - Azure OpenAI GPT-4o
- AZURE_OPENAI_INTEGRATION_TEST_* - Integration test endpoint
- AZURE_OPENAI_GPT3_5_CHAT_* - Azure OpenAI GPT-3.5
- AZURE_OPENAI_GPT4_CHAT_* - Azure OpenAI GPT-4
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_* - Azure OpenAI GPT-4o unsafe
- AZURE_OPENAI_GPT4O_UNSAFE_CHAT_*2 - Azure OpenAI GPT-4o unsafe secondary
- AZURE_FOUNDRY_DEEPSEEK_* - Azure AI Foundry DeepSeek
- AZURE_FOUNDRY_PHI4_* - Azure AI Foundry Phi-4
- AZURE_FOUNDRY_MISTRAL_LARGE_* - Azure AI Foundry Mistral Large
- GROQ_* - Groq API
- OPEN_ROUTER_* - OpenRouter API
- OLLAMA_* - Ollama local
- GOOGLE_GEMINI_* - Google Gemini (OpenAI-compatible)

**OpenAI Responses Targets (OpenAIResponseTarget):**
- AZURE_OPENAI_GPT5_RESPONSES_* - Azure OpenAI GPT-5 Responses
- AZURE_OPENAI_GPT5_RESPONSES_* (high reasoning) - Azure OpenAI GPT-5 Responses with high reasoning effort
- PLATFORM_OPENAI_RESPONSES_* - Platform OpenAI Responses
- AZURE_OPENAI_RESPONSES_* - Azure OpenAI Responses

**Realtime Targets (RealtimeTarget):**
- PLATFORM_OPENAI_REALTIME_* - Platform OpenAI Realtime
- AZURE_OPENAI_REALTIME_* - Azure OpenAI Realtime

**Image Targets (OpenAIImageTarget):**
- OPENAI_IMAGE_*1 - Azure OpenAI Image
- OPENAI_IMAGE_*2 - Platform OpenAI Image

**TTS Targets (OpenAITTSTarget):**
- OPENAI_TTS_*1 - Azure OpenAI TTS
- OPENAI_TTS_*2 - Platform OpenAI TTS

**Video Targets (OpenAIVideoTarget):**
- AZURE_OPENAI_VIDEO_* - Azure OpenAI Video

**Completion Targets (OpenAICompletionTarget):**
- OPENAI_COMPLETION_* - OpenAI Completion

**Azure ML Targets (AzureMLChatTarget):**
- AZURE_ML_PHI_* - Azure ML Phi

**Safety Targets (PromptShieldTarget):**
- AZURE_CONTENT_SAFETY_* - Azure Content Safety

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tags` | `list[TargetTag] | None` | Tags to filter which targets to register. If None, only "default" targets are registered. Defaults to `None`. |

**Methods:**

#### `initialize_async() → None`

Register available targets based on environment variables.

Scans for known endpoint environment variables and registers the
corresponding targets into the TargetRegistry. Only targets with
tags matching the configured tags are registered.
