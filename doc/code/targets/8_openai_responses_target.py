# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 8. OpenAI Responses Target
#
# In this demo, we show an example of the `OpenAIResponseTarget`. [Responses](https://platform.openai.com/docs/api-reference/responses) is a newer protocol than chat completions and provides additional functionality with a somewhat modified API. The allowed input types include text, image, web search, file search, functions, and reasoning.
#
# Before you begin, ensure you are set up with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).
# ## OpenAI Configuration
#
# Like most targets, all `OpenAITarget`s need an `endpoint` and often also needs a `model` and a `key`. These can be passed into the constructor or configured with environment variables (or in .env).
#
# - endpoint: The API endpoint (`OPENAI_RESPONSES_ENDPOINT` environment variable). For OpenAI, these are just "https://api.openai.com/v1/responses".
# - auth: The API key for authentication (`OPENAI_RESPONSES_KEY` environment variable).
# - model_name: The model to use (`OPENAI_RESPONSES_MODEL` environment variable). For OpenAI, these are any available model name and are listed here: "https://platform.openai.com/docs/models".

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIResponseTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

target = OpenAIResponseTarget()

orchestrator = PromptSendingOrchestrator(objective_target=target)

response = await orchestrator.run_attack_async(objective="Tell me a joke")  # type: ignore
await response.print_conversation_async()  # type: ignore
