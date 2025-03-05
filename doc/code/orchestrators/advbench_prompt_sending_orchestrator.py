# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AdvBench PromptSendingOrchestrator - optional
#
# This demo is uses prompts from the AdvBench dataset to try against a target. It includes the ways you can send the prompts,
# and how you can view results. Before starting, import the necessary libraries.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).
#
# The first example is as simple as it gets.
#
# The results and intermediate interactions will be saved to memory according to the environment settings. For details, see the [Memory Configuration Guide](../memory/0_memory.md).

# %%
import time

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_adv_bench_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget

initialize_pyrit(memory_db_type=IN_MEMORY)
target = OpenAIChatTarget()

orchestrator = PromptSendingOrchestrator(objective_target=target)

adv_bench_prompts = fetch_adv_bench_dataset()

start = time.time()
await orchestrator.send_prompts_async(prompt_list=adv_bench_prompts.get_values(first=3))  # type: ignore
end = time.time()

print(f"Elapsed time for operation: {end-start}")

await orchestrator.print_conversations_async()  # type: ignore

orchestrator.dispose_db_engine()
