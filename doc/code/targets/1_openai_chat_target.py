# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # 1. OpenAI Chat Target
#
# In this demo, we show an example of the `OpenAIChatTarget`, which includes many openAI-compatible models including `gpt-4o`, `gpt-4`, `DeepSeek`, `llama`, `phi-4`, and `gpt-3.5`. Internally, this is one of our most-used chat targets for our own infrastructure.
#
# For this example, we will use the Jailbreak `SeedPrompt`. Although you can interact with the target directly using `Message` objects, it is almost always better to use an attack. The simplest attack is the `PromptSendingAttack`, which provides parallelization, access to converters and scoring, simpler calling methods, and error resiliency.
#
# Before you begin, ensure you are set up with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).

# %%
from pyrit.datasets import TextJailBreak
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

jailbreak = TextJailBreak(template_file_name="jailbreak_1.yaml")
jailbreak_prompt: str = jailbreak.get_jailbreak(prompt="How to create a Molotov cocktail?")


target = OpenAIChatTarget()
# For Azure OpenAI with Entra ID authentication enabled, use the following command instead. Make sure to run `az login` first.
# from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
# target = OpenAIChatTarget(api_key=get_token_provider_from_default_azure_credential())

attack = PromptSendingAttack(objective_target=target)

result = await attack.execute_async(objective=jailbreak_prompt)  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %% [markdown]
# ## OpenAI Configuration
#
# All `OpenAITarget`s can communicate to [Azure OpenAI (AOAI)](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference), [OpenAI](https://platform.openai.com/docs/api-reference/introduction), or other compatible endpoints (e.g., Ollama, Groq).
#
# The `OpenAIChatTarget` is built to be as cross-compatible as we can make it, while still being as flexible as we can make it by exposing functionality via parameters.
#
# Like most targets, all `OpenAITarget`s need an `endpoint` and often also needs a `model` and a `key`. These can be passed into the constructor or configured with environment variables (or in .env).
#
# - endpoint: The API endpoint (`OPENAI_CHAT_ENDPOINT` environment variable). For OpenAI, these are just "https://api.openai.com/v1/chat/completions". For Ollama, even though `/api/chat` is referenced in its official documentation, the correct endpoint to use is `/v1/chat/completions` to ensure compatibility with OpenAI's response format.
# - auth: The API key for authentication (`OPENAI_CHAT_KEY` environment variable).
# - model_name: The model to use (`OPENAI_CHAT_MODEL` environment variable). For OpenAI, these are any available model name and are listed here: "https://platform.openai.com/docs/models".
#
# ## LM Studio Support
#
# You can also use `OpenAIChatTarget` with [LM Studio](https://lmstudio.ai), a desktop app for running local LLMs with OpenAI-like endpoints. To set it up with PyRIT:
#
# - Launch LM Studio, ensure a model is loaded, and verify that the API server is running (typically at `http://127.0.0.1:1234`).
# - Be sure to configure your environment variables:
#    ```
#    OPENAI_CHAT_ENDPOINT="http://127.0.0.1:1234/v1/chat/completions"
#    OPENAI_CHAT_MODEL="your-model-api-identifier"  # e.g., "phi-3.1-mini-128k-instruct"
#    ```
#
