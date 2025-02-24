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
# # GroqChatTarget
#
# This example demonstrates how to use the `GroqChatTarget` class in PyRIT to send a prompt
# to a Groq model and retrieve a response.
#
# ## Setup
# Before running this example, you need to set the following environment variables:
#
# ```
# export GROQ_API_KEY="your_api_key_here"
# export GROQ_MODEL_NAME="llama3-8b-8192"
# ```
#
# Alternatively, you can pass these values as arguments when initializing `GroqChatTarget`:
#
# ```python
# groq_target = GroqChatTarget(model_name="llama3-8b-8192", api_key="your_api_key_here")
# ```
#
# You can also limit the request rate using `max_requests_per_minute`.
#
# ## Example
# The following code initializes `GroqChatTarget`, sends a prompt using `PromptSendingOrchestrator`,
# and retrieves a response.
# %%

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import GroqChatTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

groq_target = GroqChatTarget()

prompt = "Why is the sky blue ?"

orchestrator = PromptSendingOrchestrator(objective_target=groq_target)

response = await orchestrator.send_prompts_async(prompt_list=[prompt])  # type: ignore
await orchestrator.print_conversations_async()  # type: ignore
