# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # OpenAI Completions - optional
#
# Before you begin, ensure you are setup with the correct version of PyRIT and have the applicable secrets configured as described [here](../../setup/populating_secrets.md).
#
# Once you are configured, then you will be able to get completions for your text.

from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAICompletionTarget

# %%
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# Note that max_tokens will default to 16 for completions, so you may want to set the upper limit of allowed tokens for a longer response.
target = OpenAICompletionTarget(max_tokens=2048)

attack = PromptSendingAttack(objective_target=target)
result = await attack.execute_async(objective="Hello! Who are you?")  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore
