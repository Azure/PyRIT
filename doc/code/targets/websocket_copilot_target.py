# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # WebSocket Copilot Target
#
# The `WebSocketCopilotTarget` is an alternative to the `PlaywrightCopilotTarget` that is designed to be more reliable by minimizing dependence on browser automation. Instead of driving the Copilot UI, it communicates directly with Copilot over a WebSocket connection, using Playwright only for authentication.
#
# Before using this target, ensure you have:
#
# 1. A licensed Microsoft 365 Copilot account (the free version is not supported)
# 2. Playwright installed: `pip install playwright && playwright install chromium`
# 3. Set the following environment variables:
#    - `COPILOT_USERNAME`: Your Microsoft account username/email
#    - `COPILOT_PASSWORD`: Your Microsoft account password
#
# Note:
# The `WebSocketCopilotTarget` uses `CopilotAuthenticator` under the hood, which launches a headless browser once to obtain authentication tokens. These tokens are then cached for subsequent requests and refreshed as needed.

# %% [markdown]
# ## Basic Usage with `PromptSendingAttack`
#
# The simplest way to interact with the `WebSocketCopilotTarget` is through the `PromptSendingAttack` class.

# %%
import asyncio
import sys

from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import WebSocketCopilotTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

await initialize_pyrit_async(memory_db_type=IN_MEMORY)

target = WebSocketCopilotTarget()
attack = PromptSendingAttack(objective_target=target)

objective = "Tell me a joke about AI"

result = await attack.execute_async(objective=objective)
await ConsoleAttackResultPrinter().print_conversation_async(result=result)

# %% [markdown]
# ## Multi-Turn Conversations
#
# The `WebSocketCopilotTarget` supports multi-turn conversations by leveraging Copilot's server-side conversation management. It automatically generates consistent `session_id` and `conversation_id` values for each PyRIT conversation, enabling Copilot to maintain context across multiple turns.
#
# However, this target does not support setting a system prompt nor modifying conversation history. As a result, it cannot be used with attack strategies that require altering prior messages (such as PAIR, TAP, or flip attack) or in contexts where a `PromptChatTarget` is required.
#
# Here is a simple multi-turn conversation example:

# %%
from pyrit.executor.attack import MultiPromptSendingAttack
from pyrit.models import Message

prompts = [
    "I'm thinking of a number between 1 and 10.",
    "It's greater than 5.",
    "It's an even number.",
    "What number am I thinking of?",
]

messages = [Message.from_prompt(prompt=p, role="user") for p in prompts]
multi_turn_attack = MultiPromptSendingAttack(objective_target=target)

result = await multi_turn_attack.execute_async(
    objective="Engage in a multi-turn conversation about a number guessing game",
    messages=messages,
)

await ConsoleAttackResultPrinter().print_conversation_async(result=result)

# %%
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
