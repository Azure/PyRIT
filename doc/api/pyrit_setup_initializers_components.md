# pyrit.setup.initializers.components

Component initializers for targets, scorers, and other components.

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

## `class TargetConfig`

Configuration for a target to be registered.

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
