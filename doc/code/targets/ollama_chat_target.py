# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: pyrt_env
#     language: python
#     name: pyrt_env
# ---

# %% [markdown]
# # OllamaChatTarget
#
# This example demonstrates how to use the `OllamaChatTarget` class in PyRIT to send a prompt
# to Ollama and retrieve a response.
#
# ## Setup
# Before running this example, you need to set the following environment variables:
# ```
# export OLLAMA_ENDPOINT="http://127.0.0.1:11434/api/chat"
# export OLLAMA_MODEL_NAME="llama3:8b"
# ```
# Alternatively, you can pass these values as arguments when initializing `OllamaChatTarget`:
# ```python
# groq_target = OllamaChatTarget(model_name="llama3:8b", endpoint="http://127.0.0.1:11434/api/chat")
# ```
# ## Example
# The following code initializes `OllamaChatTarget`, sends a prompt using `PromptSendingOrchestrator`,
# and retrieves a response.
# %%

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OllamaChatTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

groq_target = OllamaChatTarget()

prompt = "Why is the sky blue ?"

orchestrator = PromptSendingOrchestrator(objective_target=groq_target)

response = await orchestrator.send_prompts_async(prompt_list=[prompt])  # type: ignore
await orchestrator.print_conversations_async()  # type: ignore
