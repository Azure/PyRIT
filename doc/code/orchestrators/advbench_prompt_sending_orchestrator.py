# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-311
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
import uuid

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter
from pyrit.datasets import fetch_adv_bench_dataset


default_values.load_environment_files()
target = OpenAIChatTarget()

# %%
# You could optionally pass memory labels to orchestrators, which will be associated with each prompt and assist in retrieving or scoring later.
test_op_name = str(uuid.uuid4())
test_user_name = str(uuid.uuid4())
memory_labels = {"op_name": test_op_name, "user_name": test_user_name}
with PromptSendingOrchestrator(prompt_target=target, memory_labels=memory_labels) as orchestrator:
    adv_bench_prompts = fetch_adv_bench_dataset()

    start = time.time()
    await orchestrator.send_prompts_async(prompt_list=adv_bench_prompts.prompts[:3])  # type: ignore
    end = time.time()

    print(f"Elapsed time for operation: {end-start}")

    orchestrator.print_conversations()

# %% [markdown]
# ## Adding Converters
#
# Additionally, we can make it more interesting by initializing the orchestrator with different types of prompt converters.
# This variation takes the original example, but converts the text to base64 before sending it to the target.

# %%
with PromptSendingOrchestrator(
    prompt_target=target, prompt_converters=[Base64Converter()], batch_size=1
) as orchestrator:

    adv_bench_prompts = fetch_adv_bench_dataset()

    # this is run in a Jupyter notebook, so we can use await
    await orchestrator.send_prompts_async(prompt_list=adv_bench_prompts.prompts[:3])  # type: ignore

    orchestrator.print_conversations()
